#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Loading Utilities for Pipeline Parallel Training

This module provides comprehensive data loading utilities for pipeline parallel training
with data parallelism across nodes.

Key Features:
- Pipeline-aware data loading and distribution
- Data parallel sharding across nodes with proper synchronization
- Micro-batch preparation for pipeline parallelism
- Support for both first-stage data loading and intermediate stage activation passing
- Cross-node data parallel coordination
- Memory-efficient data handling

Training Iteration Flow:
    1. context_ids → LLM (no grad) → memory_states
    2. memory_states → Hypernetwork (with grad) → loradict
    3. conversation_ids + loradict → LLM (with grad) → hidden_states
    4. hidden_states + labels → cross entropy loss → backward

Dataset Format:
    Each sample contains:
    - context_ids:                (context_seq_len,)  token ids for context
    - conversation_ids:           (conv_seq_len,)     token ids for conversation
    - labels:                     (conv_seq_len,)     labels for loss computation
    - context_attention_mask:     (context_seq_len,)  attention mask for context (unused by collator)
    - conversation_attention_mask:(conv_seq_len,)     attention mask for conversation (unused by collator)
"""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
import torch.distributed as dist
from typing import Dict, List, Any, Optional, Iterator, Tuple
import logging
import numpy as np
import os
import time
import random
from utils.mytokenizer import create_tokenizer
import importlib

logger = logging.getLogger(__name__)


_special_token_cache: Dict[str, dict] = {}


def resolve_special_token_id(model_path: str, *, tokenizer_cfg=None) -> dict:
    """
    Resolve all special token ids from the tokenizer at ``model_path``.

    Reads the ``added_tokens_decoder`` section from the tokenizer config
    and builds a dict mapping each token string to its integer id.
    Each token is verified to encode to exactly **one** token id via the
    tokenizer; if tokenization fails or produces more than one id, a
    ``ValueError`` is raised.

    Results are cached per ``model_path`` so repeated calls are free.

    Args:
        model_path: Path to the pretrained model / tokenizer directory.
        tokenizer_cfg: Required on first call (when cache is empty). Either
            a DictConfig object (from Hydra cfg.tokenizer) or a path string
            to a tokenizer YAML config file. Subsequent calls with the same
            model_path use the cache and don't need this parameter.

    Returns:
        A dict ``{token_string: token_id}`` for every entry in
        ``added_tokens_decoder``.  For example::

            {"<|endoftext|>": 248044, "<|im_start|>": 248045, ...}

    Raises:
        ValueError: If any token does not map to exactly one id.
        FileNotFoundError: If ``tokenizer_config.json`` is not found.
        TypeError: If tokenizer_cfg is None and result is not cached.
    """
    abs_path = os.path.abspath(model_path)
    if abs_path in _special_token_cache:
        return _special_token_cache[abs_path]

    if tokenizer_cfg is None:
        raise TypeError(
            f"tokenizer_cfg must be provided on first call to resolve_special_token_id "
            f"for model_path='{model_path}' (result not yet cached). "
            f"Pass either a DictConfig (from Hydra cfg.tokenizer) or a YAML file path."
        )
    import json as _json

    config_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"tokenizer_config.json not found at '{model_path}'. "
            f"Cannot resolve special token ids."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        tokenizer_config = _json.load(f)

    added_tokens = tokenizer_config.get("added_tokens_decoder", {})
    if not added_tokens:
        raise ValueError(
            f"No 'added_tokens_decoder' found in tokenizer_config.json at "
            f"'{model_path}'."
        )

    tokenizer = create_tokenizer(model_path, tokenizer_cfg=tokenizer_cfg)

    special_token_dict = {}
    for expected_id_str, token_info in added_tokens.items():
        token_str = token_info["content"]
        expected_id = int(expected_id_str)

        # Verify via tokenizer encoding
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Special token '{token_str}' does not map to exactly one id "
                f"in the tokenizer at '{model_path}'. "
                f"Got {len(token_ids)} token(s): {token_ids}."
            )
        if token_ids[0] != expected_id:
            raise ValueError(
                f"Special token '{token_str}' encodes to id {token_ids[0]} "
                f"but expected {expected_id} from added_tokens_decoder."
            )

        special_token_dict[token_str] = expected_id

    # Also include extra special tokens defined in tokenizer config
    from utils.mytokenizer import get_extra_token_ids
    extra_ids = get_extra_token_ids(tokenizer_cfg=tokenizer_cfg)
    special_token_dict.update(extra_ids)

    logger.info(
        f"Resolved {len(special_token_dict)} special token ids "
        f"from tokenizer at '{model_path}' "
        f"(including {len(extra_ids)} extra tokens from config)"
    )
    _special_token_cache[abs_path] = special_token_dict
    return special_token_dict


def resolve_pad_token_id(model_path: str, pad_token: str = "<|endoftext|>", *, tokenizer_cfg=None) -> int:
    """
    Convenience wrapper: resolve just the pad_token_id.

    Calls :func:`resolve_special_token_id` and returns the id for
    ``pad_token``.

    Args:
        model_path: Path to the pretrained model / tokenizer directory.
        pad_token:  The special token string to look up.  Default
                    ``"<|endoftext|>"``.
        tokenizer_cfg: Required. Either a DictConfig object (from Hydra
            cfg.tokenizer) or a path string to a tokenizer YAML config file.

    Returns:
        The integer token id for ``pad_token``.
    """
    special_tokens = resolve_special_token_id(model_path, tokenizer_cfg=tokenizer_cfg)
    if pad_token not in special_tokens:
        raise ValueError(
            f"'{pad_token}' not found in special tokens. "
            f"Available: {list(special_tokens.keys())}"
        )
    pad_token_id = special_tokens[pad_token]
    logger.info(
        f"Resolved pad_token_id: '{pad_token}' → {pad_token_id} "
        f"(tokenizer: {model_path})"
    )
    return pad_token_id


def resolve_dataset_seed(cfg=None) -> int:
    """
    Resolve the canonical dataset seed from config.

    Priority:
        1. cfg.seed.dataset (if cfg is provided and has the field)
        2. configs/base.yaml seed.dataset (loaded from disk)

    This ensures ALL dataset-related randomness uses the single
    authoritative seed defined in configs/base.yaml.

    Args:
        cfg: Optional Hydra DictConfig. If provided and contains
             seed.dataset, that value is used directly.

    Returns:
        The integer dataset seed.
    """
    # Try cfg first
    if cfg is not None:
        try:
            return int(cfg.seed.dataset)
        except Exception:
            pass

    # Fallback: load from base.yaml on disk
    _base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "base.yaml"
    )
    if os.path.exists(_base_path):
        import yaml
        with open(_base_path, "r") as f:
            base_data = yaml.safe_load(f)
        if base_data and "seed" in base_data and "dataset" in base_data["seed"]:
            return int(base_data["seed"]["dataset"])

    # Should never reach here if base.yaml exists
    raise RuntimeError(
        "Cannot resolve dataset seed: cfg.seed.dataset not available and "
        "configs/base.yaml not found or missing seed.dataset field."
    )


# ---------------------------------------------------------------------------
# PipelineDataLoader
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int):
    """Deterministic worker seeding for DataLoader reproducibility.

    PyTorch sets each worker's seed to ``base_seed + worker_id`` where
    ``base_seed`` comes from the DataLoader's ``generator``.  We propagate
    this to Python's ``random`` and NumPy so that any randomness inside
    ``__getitem__`` or the collator is also reproducible.
    """
    import random, numpy as np
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class _ContiguousBlockSampler:
    """
    A sampler that assigns contiguous blocks of indices to each DP rank.

    Unlike DistributedSampler (which interleaves indices round-robin),
    this sampler gives each rank a contiguous slice of the dataset:
        rank 0 -> [0, 1, ..., block_size-1]
        rank 1 -> [block_size, block_size+1, ..., 2*block_size-1]
        ...

    This preserves the dataset's natural ordering within each rank,
    which is essential for datasets that should NOT be shuffled during
    training (e.g., sequential curriculum data).
    """

    def __init__(self, dataset_size: int, num_replicas: int, rank: int, drop_last: bool = True):
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        if drop_last:
            # Each rank gets exactly floor(dataset_size / num_replicas) samples
            self.block_size = dataset_size // num_replicas
        else:
            # Each rank gets ceil(dataset_size / num_replicas) samples
            self.block_size = (dataset_size + num_replicas - 1) // num_replicas

        self.start_idx = self.rank * self.block_size
        self.end_idx = min(self.start_idx + self.block_size, self.dataset_size)
        self.num_samples = self.end_idx - self.start_idx

    def __iter__(self):
        return iter(range(self.start_idx, self.end_idx))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """No-op for compatibility with DistributedSampler interface."""
        pass


class PipelineDataLoader:
    """
    Advanced data loader designed for pipeline parallel training with data parallelism.

    This loader handles:
    - Data parallel sharding across multiple nodes
    - Micro-batch preparation for pipeline stages
    - Proper data distribution to different pipeline stages
    - Cross-node synchronization for data parallel groups
    - Memory-efficient data loading

    The collator returns List[Dict], where each dict is a "sub-item" representing
    one forward pass. Currently all datasets return a list of length 1.
    Each sub-item dict has keys:
        context_ids, conversation_ids, labels, context_lengths
    and optionally:
        distill: {conversation_ids, labels}  (for distillation)

    Each micro-batch is a dict with the same keys as a sub-item.
    """

    TENSOR_KEYS = [
        "context_ids", "conversation_ids", "labels",
        "context_lengths",
    ]

    DISTILL_TENSOR_KEYS = [
        "conversation_ids", "labels",
    ]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        micro_batch_size: int,
        data_parallel_rank: int = 0,
        data_parallel_size: int = 1,
        pipeline_stage: int = 0,
        total_pipeline_stages: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn=None,
        batches_per_epoch: int = -1,
        seed: int = 42,
    ):
        """
        Initialize advanced pipeline data loader.

        Args:
            dataset: The dataset to load from
            batch_size: Global batch size (across all data parallel groups)
            micro_batch_size: Size of each micro-batch for pipeline
            data_parallel_rank: Rank in data parallel group
            data_parallel_size: Size of data parallel group
            pipeline_stage: Current pipeline stage index
            total_pipeline_stages: Total number of pipeline stages
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle the data
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop incomplete batches
            collate_fn: Custom collate function (required — provided by dataset module)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.pipeline_stage = pipeline_stage
        self.total_pipeline_stages = total_pipeline_stages
        self.is_first_stage = (pipeline_stage == 0)
        self.is_last_stage = (pipeline_stage == total_pipeline_stages - 1)

        # Calculate local batch size for this data parallel group
        self.local_batch_size = batch_size // data_parallel_size

        # Verify batch sizes are compatible
        if self.local_batch_size % micro_batch_size != 0:
            raise ValueError(
                f"Local batch size ({self.local_batch_size}) must be divisible by "
                f"micro_batch_size ({micro_batch_size})"
            )

        # Collator must be provided by the dataset module
        if collate_fn is None:
            raise ValueError(
                "collate_fn is required. Each dataset module must provide "
                "its own collator via create_dataset_and_collator()."
            )

        # Only first pipeline stage needs to actually load data.
        # Other stages receive activations from the previous stage.
        if self.is_first_stage:
            # Setup distributed sampler for data parallelism.
            if shuffle:
                # Shuffle mode: DistributedSampler with random shuffling.
                # The seed ensures all DP replicas use the same shuffling
                # order (each replica then takes its own shard).
                self.sampler = DistributedSampler(
                    dataset,
                    num_replicas=data_parallel_size,
                    rank=data_parallel_rank,
                    shuffle=True,
                    seed=seed,
                )
            else:
                # Sequential mode: each DP rank gets a contiguous block of
                # data in order. E.g. with 4 ranks and 16 samples:
                #   rank 0 -> [0,1,2,3], rank 1 -> [4,5,6,7], etc.
                # This preserves the dataset's natural ordering within each rank.
                self.sampler = _ContiguousBlockSampler(
                    dataset_size=len(dataset),
                    num_replicas=data_parallel_size,
                    rank=data_parallel_rank,
                    drop_last=drop_last,
                )

            # Create a deterministic generator for the DataLoader so that
            # worker random states are reproducible across resume.
            self._dl_generator = torch.Generator()
            self._dl_generator.manual_seed(seed)

            # Create base data loader
            self.data_loader = DataLoader(
                dataset,
                batch_size=self.local_batch_size,
                sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
                generator=self._dl_generator,
                worker_init_fn=_seed_worker,
            )

            logger.info(
                f"PipelineDataLoader initialized for first stage "
                f"(DP rank {data_parallel_rank}/{data_parallel_size}, "
                f"local batch size: {self.local_batch_size})"
            )
        else:
            self.sampler = None
            self.data_loader = None
            logger.info(
                f"PipelineDataLoader initialized for stage {pipeline_stage} "
                f"(receives activations, no data loading)"
            )

        self.iter = None
        self.current_epoch = 0
        self._seed = seed  # Store seed for reproducible resume

        # Snapshot of _dl_generator state taken BEFORE each iter() call.
        # This is the state that determines worker seeding.  On resume we
        # restore this snapshot so that iter() produces identical worker
        # seeds as the original training run.
        self._dl_generator_pre_iter_state = None
        if hasattr(self, '_dl_generator'):
            self._dl_generator_pre_iter_state = self._dl_generator.get_state().clone()

        # For non-first stages: track batches_per_epoch so they can raise
        # StopIteration at the same boundary as the first stage.
        # If batches_per_epoch <= 0, non-first stages never raise StopIteration
        # (legacy behaviour).
        self._batches_per_epoch = batches_per_epoch
        self._batch_counter = 0

    def __iter__(self):
        """Initialize iterator for the data loader.

        If ``_batches_to_skip`` > 0 (set by ``load_state_dict``), fast-forwards
        the underlying DataLoader iterator by consuming (and discarding) that
        many batches so that training resumes from the exact position.
        """
        if not self.is_first_stage:
            # Reset batch counter so non-first stages can track epoch boundary
            self._batch_counter = 0
            return self

        if self.data_loader is None:
            raise RuntimeError("Data loader not initialized for first stage")

        # Snapshot generator state BEFORE iter() consumes it.
        # This is the state that determines worker seeding; we save it
        # so that on resume, restoring this state + calling iter() will
        # produce identical worker random states.
        if hasattr(self, '_dl_generator'):
            self._dl_generator_pre_iter_state = self._dl_generator.get_state().clone()

        self.iter = iter(self.data_loader)

        # Fast-forward: skip batches already processed before checkpoint
        _skip = getattr(self, '_batches_to_skip', 0)
        if _skip > 0:
            for _ in range(_skip):
                try:
                    next(self.iter)
                except StopIteration:
                    break
            # _batch_counter stays at the restored value (already accounts for skipped batches)
            self._batches_to_skip = 0  # Only skip once after resume
        else:
            # Normal epoch start: reset batch counter
            self._batch_counter = 0

        return self

    def __next__(self):
        """Get next batch and prepare micro-batches for pipeline."""
        if not self.is_first_stage:
            # If batches_per_epoch is set, raise StopIteration at epoch boundary
            # so all pipeline stages stay synchronized.
            if self._batches_per_epoch > 0:
                if self._batch_counter >= self._batches_per_epoch:
                    self._batch_counter = 0
                    self.current_epoch += 1
                    raise StopIteration
                self._batch_counter += 1
            return {
                "stage": self.pipeline_stage,
                "is_first": False,
                "is_last": self.is_last_stage,
                "micro_batches": None,
                "data_parallel_rank": self.data_parallel_rank,
            }

        if self.iter is None:
            self.__iter__()

        try:
            batch = next(self.iter)
            self._batch_counter += 1  # Track position for checkpoint resume

            # Collator returns List[Dict] (list of sub-items).
            # Each sub-item is a dict with standard tensor keys.
            # Currently all datasets return exactly 1 sub-item.
            if isinstance(batch, list):
                num_sub_items = len(batch)
                # For now, we only support num_sub_items == 1 in the
                # training loop. Future multi-loradict will handle > 1.
                assert num_sub_items >= 1, (
                    f"Collator returned empty list. Expected at least 1 sub-item."
                )
                primary_batch = batch[0]
            else:
                # Backward compatibility: if collator returns a plain dict
                num_sub_items = 1
                primary_batch = batch

            # Split primary sub-item into micro-batches for pipeline
            micro_batches = self._split_into_micro_batches(primary_batch)

            return {
                "stage": self.pipeline_stage,
                "is_first": True,
                "is_last": self.is_last_stage,
                "micro_batches": micro_batches,
                "num_sub_items": num_sub_items,
                "data_parallel_rank": self.data_parallel_rank,
                "batch_info": {
                    "global_batch_size": self.batch_size,
                    "local_batch_size": self.local_batch_size,
                    "micro_batch_size": self.micro_batch_size,
                    "num_micro_batches": len(micro_batches),
                },
            }

        except StopIteration:
            self.iter = None
            self.current_epoch += 1
            if self.sampler is not None:
                self.sampler.set_epoch(self.current_epoch)
            # Propagate epoch to dataset for per-epoch reshuffling
            _ds = self.dataset
            if hasattr(_ds, "dataset"):
                _ds = _ds.dataset
            if hasattr(_ds, "set_epoch") and callable(_ds.set_epoch):
                _ds.set_epoch(self.current_epoch)
            raise

    def _split_into_micro_batches(self, batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Split a batch into micro-batches for pipeline parallelism.

        Args:
            batch: The full batch dict with keys in TENSOR_KEYS.
                   May also contain a "distill" sub-dict with its own tensors.
                   May also contain "extra_info" (a list of dicts) that should
                   be sliced per micro-batch.

        Returns:
            List of micro-batch dicts, each with the same keys.
            If "distill" is present, each micro-batch will also have a
            "distill" sub-dict with the corresponding slice.
            If "extra_info" is present, each micro-batch will have the
            corresponding slice of the list.
        """
        # Infer batch size from the first tensor key found
        first_key = next(k for k in self.TENSOR_KEYS if k in batch)
        total = batch[first_key].size(0)
        num_micro_batches = total // self.micro_batch_size

        micro_batches: List[Dict[str, torch.Tensor]] = []
        for i in range(num_micro_batches):
            start = i * self.micro_batch_size
            end = start + self.micro_batch_size
            mb: Dict[str, torch.Tensor] = {}
            for key, value in batch.items():
                if key == "distill":
                    # Handle nested distill dict
                    if value is not None and isinstance(value, dict):
                        mb["distill"] = {}
                        for dk, dv in value.items():
                            if isinstance(dv, torch.Tensor):
                                mb["distill"][dk] = dv[start:end]
                            else:
                                mb["distill"][dk] = dv
                    else:
                        mb["distill"] = None
                elif key == "extra_info":
                    # Handle extra_info: a list of dicts, slice by micro-batch
                    if value is not None and isinstance(value, list):
                        mb["extra_info"] = value[start:end]
                    else:
                        mb["extra_info"] = value
                elif isinstance(value, torch.Tensor):
                    mb[key] = value[start:end]
                else:
                    mb[key] = value
            micro_batches.append(mb)

        return micro_batches

    def set_epoch(self, epoch: int):
        """Set epoch for the sampler and dataset (for reproducibility and per-epoch reshuffling)."""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        # Allow dataset to reshuffle per epoch (e.g. trajectory_distill_transfer).
        # Handle torch.utils.data.Subset wrapping: propagate to underlying dataset.
        _ds = self.dataset
        if hasattr(_ds, "dataset"):
            # Subset wraps the real dataset in .dataset attribute
            _ds = _ds.dataset
        if hasattr(_ds, "set_epoch") and callable(_ds.set_epoch):
            _ds.set_epoch(epoch)
        self.current_epoch = epoch

    def state_dict(self) -> Dict[str, Any]:
        """Return serialisable state for checkpoint save.

        Saves the generator state snapshot taken BEFORE the current epoch's
        ``iter()`` call.  On resume, restoring this state ensures that
        ``iter()`` will seed workers identically to the original run.
        """
        state = {
            "current_epoch": self.current_epoch,
            "batch_counter": self._batch_counter,
            "seed": getattr(self, '_seed', 42),
        }
        # Save the pre-iter snapshot (NOT the current generator state).
        # This is the state that was captured right before iter() was
        # called for the current epoch, so restoring it + calling iter()
        # will produce identical worker seeds.
        if self._dl_generator_pre_iter_state is not None:
            state["dl_generator_pre_iter_state"] = self._dl_generator_pre_iter_state
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore state from a checkpoint.

        Must be called **before** ``set_epoch()`` / ``iter()`` for the
        resumed epoch so that the generator state is correct.

        Restores the generator to the state it had BEFORE ``iter()`` was
        called for the saved epoch.  When the training loop subsequently
        calls ``iter()``, the generator will seed workers identically to
        the original run, and fast-forward past already-processed batches.
        """
        self.current_epoch = state.get("current_epoch", 0)
        self._batch_counter = state.get("batch_counter", 0)
        # Store how many batches to skip on next iter() call
        self._batches_to_skip = self._batch_counter
        if "dl_generator_pre_iter_state" in state and hasattr(self, "_dl_generator"):
            self._dl_generator.set_state(state["dl_generator_pre_iter_state"])
            self._dl_generator_pre_iter_state = state["dl_generator_pre_iter_state"].clone()


# ---------------------------------------------------------------------------
# Dataset module validation
# ---------------------------------------------------------------------------

# Every dataset module under mydatasets/ MUST expose these three callables.
REQUIRED_DATASET_FUNCTIONS = ("create_dataset_and_collator", "preprocess", "debug")


def _validate_dataset_module(module, module_path: str) -> None:
    """
    Verify that *module* exposes all required dataset interface functions.

    Every ``mydatasets/*.py`` module MUST define:
        - ``create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)``
        - ``preprocess(cfg, model_path)``
        - ``debug(cfg, model_path)``

    Raises:
        AttributeError: listing all missing functions at once.
    """
    missing = [
        name for name in REQUIRED_DATASET_FUNCTIONS
        if not hasattr(module, name) or not callable(getattr(module, name))
    ]
    if missing:
        raise AttributeError(
            f"Dataset module '{module_path}' is missing required function(s): "
f"{missing}. Every dataset module under mydatasets/ must define: "
            f"{list(REQUIRED_DATASET_FUNCTIONS)}."
        )


# ---------------------------------------------------------------------------
# Validation split utilities
# ---------------------------------------------------------------------------

def parse_validation_num(validation_num, total_samples: int) -> int:
    """
    Parse the ``validation_num`` config value into an actual sample count.

    Supported formats:
      - ``-1``                → disabled (returns 0)
      - integer ``x`` (>= 1)  → exactly x validation samples
      - float ``y`` (0 < y < 1) → fraction of total samples

    Args:
        validation_num: Raw config value (int or float).
        total_samples: Total number of samples in the full dataset.

    Returns:
        Number of validation samples (0 if disabled).
    """
    if isinstance(validation_num, int):
        if validation_num == -1:
            return 0
        if validation_num >= 1:
            if validation_num >= total_samples:
                raise ValueError(
                    f"validation_num ({validation_num}) >= total dataset size "
                    f"({total_samples}). Must be smaller than the dataset."
                )
            return validation_num
        raise ValueError(
            f"validation_num must be -1 (disabled), a positive integer, "
            f"or a float in (0, 1). Got int: {validation_num}"
        )
    elif isinstance(validation_num, float):
        if 0.0 < validation_num < 1.0:
            val_count = max(1, int(total_samples * validation_num))
            if val_count >= total_samples:
                val_count = total_samples - 1
            return val_count
        raise ValueError(
            f"validation_num as float must be in (0, 1). Got: {validation_num}"
        )
    else:
        raise ValueError(
            f"validation_num must be int or float. "
            f"Got {type(validation_num).__name__}: {validation_num}"
        )


def split_train_val(dataset: Dataset, val_size: int, seed: int = 42):
    """
    Split a dataset into train and validation subsets.

    Selects a random contiguous block of ``val_size`` samples as the
    validation set. This preserves the dataset's natural ordering (e.g.,
    repo-contiguous ordering in trajectory_distill_transfer) within both
    the train and validation subsets.

    Uses a fixed random seed for reproducibility across ranks.

    Args:
        dataset: The full dataset.
        val_size: Number of samples for the validation set.
        seed: Random seed for reproducible split.

    Returns:
        (train_subset, val_subset): Both are ``torch.utils.data.Subset``.
    """
    total = len(dataset)
    assert 0 < val_size < total, (
        f"val_size ({val_size}) must be in (0, {total})"
    )

    rng = random.Random(seed)
    # Pick a random start position for the contiguous validation block
    max_start = total - val_size
    val_start = rng.randint(0, max_start)
    val_end = val_start + val_size

    val_indices = list(range(val_start, val_end))
    train_indices = list(range(0, val_start)) + list(range(val_end, total))

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_dataset_from_config(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create dataset and collator from the dataset module specified in cfg.data.

    Each data config YAML (configs/data/{pretrain,sft}/<name>.yaml) MUST contain a
    ``dataset_module`` field pointing to a Python module under ``mydatasets/``
    (e.g. ``mydatasets.pretrain.oldpretrain``).  That module MUST expose a factory:

        create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
            -> (Dataset, collator_callable)

    If ``cfg.data.validation_split_num`` (pretrain) or
    ``cfg.data.validation_subset_num`` (sft) is set and not -1, the dataset
    is split into train and validation subsets.

    The global ``cfg.seed.dataset`` seed is used for all random operations
    (train/val split, data shuffling, DP sharding) to guarantee
    reproducibility and cross-node consistency.

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data with dataset_module).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (train_dataset, val_dataset_or_None, collator)

    Raises:
        ValueError: If cfg.data.dataset_module is missing or empty.
    """
    # Resolve the global dataset seed
    dataset_seed = cfg.seed.dataset

    dataset_module_path = cfg.data.get("dataset_module", None)
    if not dataset_module_path:
        raise ValueError(
            "cfg.data.dataset_module is required but not set. "
            "Every data config YAML must specify a dataset_module "
            "(e.g. 'mydatasets.pretrain.oldpretrain')."
        )

    data_name = cfg.data.get("name", None)
    if not data_name:
        raise ValueError(
            "cfg.data.name is required but not set. "
            "Every data config YAML must explicitly specify a 'name' field."
        )
    logger.info(
        f"Loading dataset module: {dataset_module_path} "
        f"(name={data_name}, dataset_seed={dataset_seed})"
    )

    dataset_mod = importlib.import_module(dataset_module_path)
    _validate_dataset_module(dataset_mod, dataset_module_path)

    dataset, collator = dataset_mod.create_dataset_and_collator(
        cfg, model_path, pad_token_id, num_mem_token,
    )

    total_samples = len(dataset)
    logger.info(f"Dataset '{data_name}' loaded: {total_samples} samples")

    # --- Validation: prefer module's own create_val_dataset if available ---
    if hasattr(dataset_mod, "create_val_dataset") and callable(dataset_mod.create_val_dataset):
        val_dataset = dataset_mod.create_val_dataset(cfg, model_path, pad_token_id, num_mem_token)
        if val_dataset is not None:
            logger.info(
                f"Dataset '{data_name}': using module-provided validation set "
                f"({len(val_dataset)} samples)"
            )
            return dataset, val_dataset, collator
        else:
            # Fall through to split logic below (val file may not exist,
            # but we can still try splitting the training set)
            logger.info(
                f"Dataset '{data_name}': module create_val_dataset returned None, "
                f"falling through to split logic"
            )

    # --- Validation split (uses global dataset seed) ---
    # Support both new names: validation_split_num (pretrain) and validation_subset_num (sft)
    validation_num = cfg.data.get(
        "validation_split_num",
        cfg.data.get("validation_subset_num", -1)
    )
    val_size = parse_validation_num(validation_num, total_samples)

    if val_size > 0:
        train_dataset, val_dataset = split_train_val(dataset, val_size, seed=dataset_seed)
        logger.info(
            f"Dataset '{data_name}' split: "
            f"train={len(train_dataset)}, val={len(val_dataset)} "
            f"(seed={dataset_seed})"
        )
        return train_dataset, val_dataset, collator
    else:
        logger.info(f"Dataset '{data_name}': validation disabled (validation_num={validation_num})")
        return dataset, None, collator


# ---------------------------------------------------------------------------
# Generic debug utility — used by every dataset module
# ---------------------------------------------------------------------------

def debug_dataset(
    dataset,
    collator,
    tokenizer,
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    num_samples: int = 5,
    num_mem_token: int = 10,
    pad_token_id: int = 0,
    output_dir: Optional[str] = None,
    strict_shape_check: bool = True,
):
    """
    Sample the first ``num_samples`` data points, run them through the
    collator, and print a per-token aligned table for each sample.

    Output is written to ``{output_dir}/{dataset_name}.txt`` (and also
    printed to stdout).  If ``output_dir`` is ``None``, the file is placed
next to the calling dataset module (``mydatasets/`` directory).

    For every sample the table has one row per token position.  Columns:

        Idx | Ctx Token | Ctx ID | Label ID | Conv Token | Conv ID | Note

    Special tokens (pad, mem-placeholder, -100 labels) are annotated in
    the *Note* column.

    Args:
        dataset:      The Dataset instance.
        collator:     The collator callable.
        tokenizer:    A HuggingFace tokenizer (for id→token decoding).
        dataset_name: Human-readable name printed in the header.
        metadata:     Optional dict of extra info to print (e.g. seq lengths).
        num_samples:  How many samples to inspect.
        num_mem_token: Number of memory-token placeholders (for annotation).
        pad_token_id: Token id used for padding.
        output_dir:   Directory to write ``{dataset_name}.txt``.  Defaults to
                      the directory of the **caller** module (e.g.
                      ``mydatasets/pretrain/`` or ``mydatasets/sft/``).
    """
    # Determine output file path — default to the caller's directory so that
    # pretrain debug files land in mydatasets/pretrain/ and sft debug files
    # land in mydatasets/sft/.
    if output_dir is None:
        import inspect
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        output_dir = os.path.dirname(os.path.abspath(caller_file))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_debug.txt")

    n = min(num_samples, len(dataset))
    samples = [dataset[i] for i in range(n)]
    batch_output = collator(samples)

    # Handle both List[Dict] (new format) and plain Dict (legacy) collator output
    if isinstance(batch_output, list):
        batch = batch_output[0]
    else:
        batch = batch_output

    ctx_ids_batch = batch["context_ids"]        # (B, ctx_total_len)
    conv_ids_batch = batch["conversation_ids"]  # (B, conv_len)
    labels_batch = batch["labels"]              # (B, conv_len)
    ctx_lengths = batch["context_lengths"]      # (B,)

    # ---- Shape validation ----
    conv_len = conv_ids_batch.size(1)
    label_len = labels_batch.size(1)
    ctx_total_len = ctx_ids_batch.size(1)
    if conv_len != label_len:
        raise ValueError(
            f"conversation_ids length ({conv_len}) != labels length ({label_len}). "
            f"They must be identical."
        )
    if ctx_total_len != conv_len + num_mem_token:
        if strict_shape_check:
            raise ValueError(
                f"context_ids length ({ctx_total_len}) != conversation_ids length ({conv_len}) "
                f"+ num_mem_token ({num_mem_token}) = {conv_len + num_mem_token}. "
                f"ctx_len must equal conv_len + num_mem_token."
            )
        else:
            logger.warning(
                f"context_ids length ({ctx_total_len}) != conversation_ids length ({conv_len}) "
                f"+ num_mem_token ({num_mem_token}) = {conv_len + num_mem_token}. "
                f"Skipping strict shape check (SFT mode)."
            )

    # Open file for real-time writing (flushed after every _print call)
    _log_file = open(output_path, "w", encoding="utf-8")

    def _print(*args, **kwargs):
        """Print to both the file (flushed immediately) and stdout."""
        import builtins
        builtins.print(*args, **kwargs, file=_log_file)
        _log_file.flush()
        builtins.print(*args, **kwargs)

    # Extract distill data if available
    distill_data = batch.get("distill")
    has_distill = (distill_data is not None and isinstance(distill_data, dict)
                   and "conversation_ids" in distill_data and "labels" in distill_data)
    if has_distill:
        distill_conv_batch = distill_data["conversation_ids"]  # (B, conv_len)
        distill_labels_batch = distill_data["labels"]          # (B, conv_len)

    # ---- Header ----
    sep = "=" * 120
    _print(sep)
    _print(f"  DEBUG — Dataset: {dataset_name}")
    _print(sep)
    _print(f"  Total samples : {len(dataset)}")
    _print(f"  Inspecting    : {n} samples")
    _print(f"  Batch shape   : context_ids={list(ctx_ids_batch.shape)}, "
           f"conversation_ids={list(conv_ids_batch.shape)}, "
           f"labels={list(labels_batch.shape)}")
    if has_distill:
        _print(f"  Distill shape : conversation_ids={list(distill_conv_batch.shape)}, "
               f"labels={list(distill_labels_batch.shape)}")
    else:
        _print(f"  Distill       : None")
    _print(f"  num_mem_token : {num_mem_token}")
    _print(f"  pad_token_id  : {pad_token_id}")
    if metadata:
        for k, v in metadata.items():
            _print(f"  {k:14s}: {v}")
    _print(sep)

    for si in range(n):
        ctx_ids = ctx_ids_batch[si]    # (ctx_total_len,)
        conv_ids = conv_ids_batch[si]  # (conv_len,)
        labels = labels_batch[si]      # (conv_len,)
        ctx_len = ctx_lengths[si].item()

        max_pos = max(ctx_ids.size(0), conv_ids.size(0))

        _print(f"\n{'─'*120}")
        _print(f"  Sample {si}  (context valid tokens: {ctx_len})")
        _print(f"{'─'*120}")

        # Column widths
        w_idx = 5
        w_tok = 20
        w_id = 10
        w_note = 20

        header = (f"{'Idx':>{w_idx}} | "
                  f"{'Ctx Token':<{w_tok}} | {'Ctx ID':>{w_id}} | "
                  f"{'Conv Token':<{w_tok}} | {'Conv ID':>{w_id}} | "
                  f"{'Label Token':<{w_tok}} | {'Label ID':>{w_id}} | {'Note':<{w_note}}")
        _print(header)
        _print("-" * len(header))

        for pos in range(max_pos):
            notes = []
            is_mem_only = (pos >= conv_ids.size(0) and pos >= ctx_len
                          and pos < ctx_len + num_mem_token)

            # Context
            if pos < ctx_ids.size(0):
                cid = ctx_ids[pos].item()
                if tokenizer is not None:
                    ctok = tokenizer.decode([cid])
                    ctok = repr(ctok) if ctok.strip() == "" else ctok
                else:
                    ctok = str(cid)
                cid_str = str(cid)
                # Annotate special positions
                if pos >= ctx_len and pos < ctx_len + num_mem_token:
                    notes.append(f"[MEM] (default={num_mem_token})")
                elif pos >= ctx_len + num_mem_token and cid == pad_token_id:
                    notes.append("[CTX_PAD]")
            else:
                ctok = ""
                cid_str = ""

            # Conversation / Labels
            if is_mem_only:
                # mem-token-only rows: leave conv/label columns empty
                vtok = ""
                vid_str = ""
                ltok = ""
                lid_str = ""
            elif pos < conv_ids.size(0):
                vid = conv_ids[pos].item()
                if tokenizer is not None:
                    vtok = tokenizer.decode([vid])
                    vtok = repr(vtok) if vtok.strip() == "" else vtok
                else:
                    vtok = str(vid)
                vid_str = str(vid)
                lid = labels[pos].item()
                lid_str = str(lid)
                if lid == -100:
                    ltok = ""
                    notes.append("[MASKED]")
                else:
                    if tokenizer is not None:
                        ltok = tokenizer.decode([lid])
                        ltok = repr(ltok) if ltok.strip() == "" else ltok
                    else:
                        ltok = str(lid)
                if vid == pad_token_id:
                    notes.append("[CONV_PAD]")
            else:
                vtok = ""
                vid_str = ""
                lid_str = ""
                ltok = ""

            # Truncate long tokens for display
            ctok = ctok[:w_tok] if len(ctok) > w_tok else ctok
            vtok = vtok[:w_tok] if len(vtok) > w_tok else vtok
            ltok = ltok[:w_tok] if len(ltok) > w_tok else ltok
            note_str = " ".join(notes)

            _print(f"{pos:>{w_idx}} | "
                   f"{ctok:<{w_tok}} | {cid_str:>{w_id}} | "
                   f"{vtok:<{w_tok}} | {vid_str:>{w_id}} | "
                   f"{ltok:<{w_tok}} | {lid_str:>{w_id}} | {note_str:<{w_note}}")

        # ---- Distillation data for this sample ----
        if has_distill:
            distill_conv = distill_conv_batch[si]    # (conv_len,)
            distill_labels = distill_labels_batch[si]  # (conv_len,)

            _print(f"\n  {'·'*60}")
            _print(f"  Distill data for Sample {si}")
            _print(f"  {'·'*60}")

            d_header = (f"{'Idx':>{w_idx}} | "
                        f"{'Distill Conv Token':<{w_tok}} | {'Distill Conv ID':>{w_id}} | "
                        f"{'Distill Label Token':<{w_tok}} | {'Distill Label ID':>{w_id}} | {'Note':<{w_note}}")
            _print(d_header)
            _print("-" * len(d_header))

            for pos in range(distill_conv.size(0)):
                d_notes = []
                d_vid = distill_conv[pos].item()
                if tokenizer is not None:
                    d_vtok = tokenizer.decode([d_vid])
                    d_vtok = repr(d_vtok) if d_vtok.strip() == "" else d_vtok
                else:
                    d_vtok = str(d_vid)
                d_vid_str = str(d_vid)

                d_lid = distill_labels[pos].item()
                d_lid_str = str(d_lid)
                if d_lid == -100:
                    d_ltok = ""
                    d_notes.append("[MASKED]")
                else:
                    if tokenizer is not None:
                        d_ltok = tokenizer.decode([d_lid])
                        d_ltok = repr(d_ltok) if d_ltok.strip() == "" else d_ltok
                    else:
                        d_ltok = str(d_lid)

                if d_vid == pad_token_id:
                    d_notes.append("[PAD]")

                # Truncate long tokens for display
                d_vtok = d_vtok[:w_tok] if len(d_vtok) > w_tok else d_vtok
                d_ltok = d_ltok[:w_tok] if len(d_ltok) > w_tok else d_ltok
                d_note_str = " ".join(d_notes)

                _print(f"{pos:>{w_idx}} | "
                       f"{d_vtok:<{w_tok}} | {d_vid_str:>{w_id}} | "
                       f"{d_ltok:<{w_tok}} | {d_lid_str:>{w_id}} | {d_note_str:<{w_note}}")

    _print(f"\n{sep}")
    _print(f"  DEBUG complete — {n} samples inspected (distill={'yes' if has_distill else 'no'})")
    _print(f"  Output saved to: {output_path}")
    _print(sep)

    _log_file.close()


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Data Loading Utilities for Pipeline Parallel Training")
    print("=" * 60)
    print("Available classes and functions:")
    print("- PipelineDataLoader: Pipeline-aware data loader with micro-batching")
    print("- resolve_special_token_id(): Resolve all special token ids from tokenizer")
    print("- resolve_pad_token_id(): Convenience wrapper for pad token id")
    print("- create_dataset_from_config(): Create dataset & collator from config")
    print("- debug_dataset(): Generic debug utility for all dataset modules")
    print()
    print("Batch keys: context_ids, conversation_ids, labels, context_lengths")
    print("=" * 60)