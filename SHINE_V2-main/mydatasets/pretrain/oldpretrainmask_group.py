#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Old Pretrain Mask Group Dataset Module for SHINE_V2

This module extends oldpretrainmask by supporting multi-chunk data points.
It reuses the same preprocessing and cache as oldpretrainmask (same token stream),
but partitions chunks into groups where each data point spans 1 or more
consecutive chunks.

Configuration:
    group_dict: Mapping of {num_chunks: proportion_of_data_points}.
    Example: {1: 0.5, 2: 0.3, 3: 0.2} means:
        - 50% of data points span 1 chunk each
        - 30% of data points span 2 consecutive chunks each
        - 20% of data points span 3 consecutive chunks each
    Proportions are by DATA POINT count, not chunk count.

    The dataset dynamically partitions the token stream into groups at
    construction time based on group_dict and global_batch_size. Each group
    contains data points of the same chunk count, and the number of data points
    in each group is aligned to global_batch_size (so that each batch contains
    only data points of the same chunk count).

Dataset:
    Returns a list of dicts (length = num_chunks for that data point).
    Each dict has the same format as oldpretrainmask's single-chunk output.

Collator:
    Processes each chunk in the list independently (same masking logic as
    oldpretrainmask), returns a list of dicts with length = num_chunks.

Key Design:
    - Reuses oldpretrainmask's cache (token_stream.npy) — no re-preprocessing needed.
    - Supports dynamic global_batch_size changes without re-preprocessing.
    - global_batch_size = local_batch_size * num_nodes (DP across nodes, PP within node).
    - Each batch contains only data points with the same chunk count.
"""

from __future__ import annotations

import os
import json
import random
import logging
import time
import fcntl
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.base import BaseDataset, BaseCollator
from mydatasets.pretrain.oldpretrainmask import (
    preprocess,  # Reuse same preprocessing
    _check_verified_marker,
    _find_text_content_ranges,
    _random_mask,
    _span_mask,
    OldPretrainMaskCollator,
)
from utils.mytokenizer import create_tokenizer

logger = logging.getLogger(__name__)

SHARD_PREFIX = "shard_"
MANIFEST_FILE = "manifest.json"
VERIFIED_FILE = "VERIFIED"


# ---------------------------------------------------------------------------
# Group partitioning logic
# ---------------------------------------------------------------------------

def _compute_group_layout(
    total_chunks: int,
    group_dict: Dict[int, float],
    global_batch_size: int,
) -> List[Dict[str, Any]]:
    """
    Partition chunks into groups based on group_dict and global_batch_size.

    Each group contains data points of the same chunk count. The number of
    data points in each group is aligned to global_batch_size.

    Args:
        total_chunks: Total number of chunks in the token stream.
        group_dict: {num_chunks_per_point: proportion_of_data_points}
            e.g., {1: 0.5, 2: 0.3, 3: 0.2}
        global_batch_size: Total batch size across all nodes.

    Returns:
        List of group dicts, each containing:
            - num_chunks_per_point: int
            - num_data_points: int (aligned to global_batch_size)
            - chunk_start: int (starting chunk index in the stream)
            - chunk_end: int (ending chunk index, exclusive)
    """
    # Normalize proportions
    total_prop = sum(group_dict.values())
    normalized = {k: v / total_prop for k, v in group_dict.items()}

    # Sort by num_chunks (ascending) for sequential layout
    sorted_groups = sorted(normalized.items(), key=lambda x: x[0])

    # Step 1: Compute raw data point counts from proportions
    # Total data points = total_chunks / weighted_avg_chunks_per_point
    # But we need to solve: sum(n_i * c_i) <= total_chunks
    # where n_i = proportion_i * N, c_i = chunks_per_point_i
    # So: N * sum(proportion_i * c_i) <= total_chunks
    # N <= total_chunks / sum(proportion_i * c_i)
    weighted_avg = sum(prop * num_c for num_c, prop in sorted_groups)
    max_total_points = int(total_chunks / weighted_avg)

    # Step 2: Align each group's data point count to global_batch_size
    groups = []
    for num_c, prop in sorted_groups:
        raw_points = int(max_total_points * prop)
        # Align down to global_batch_size
        aligned_points = (raw_points // global_batch_size) * global_batch_size
        if aligned_points == 0:
            # Ensure at least one batch if there are enough chunks
            if raw_points >= global_batch_size:
                aligned_points = global_batch_size
            else:
                # Skip this group if not enough for even one batch
                continue
        groups.append({
            "num_chunks_per_point": num_c,
            "num_data_points": aligned_points,
            "total_chunks_used": aligned_points * num_c,
        })

    # Step 3: Assign chunk ranges sequentially
    chunk_offset = 0
    for g in groups:
        g["chunk_start"] = chunk_offset
        g["chunk_end"] = chunk_offset + g["total_chunks_used"]
        chunk_offset += g["total_chunks_used"]

    # Verify we don't exceed total chunks
    total_used = sum(g["total_chunks_used"] for g in groups)
    if total_used > total_chunks:
        # Scale down proportionally
        scale = total_chunks / total_used
        chunk_offset = 0
        for g in groups:
            raw_points = int(g["num_data_points"] * scale)
            aligned_points = (raw_points // global_batch_size) * global_batch_size
            g["num_data_points"] = aligned_points
            g["total_chunks_used"] = aligned_points * g["num_chunks_per_point"]
            g["chunk_start"] = chunk_offset
            g["chunk_end"] = chunk_offset + g["total_chunks_used"]
            chunk_offset += g["total_chunks_used"]

    # Remove empty groups
    groups = [g for g in groups if g["num_data_points"] > 0]

    return groups


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OldPretrainMaskGroupDataset(BaseDataset):
    """
    Chunked pretrain dataset with multi-chunk data point support.

    Reuses the same token_stream.npy as oldpretrainmask but partitions chunks
    into groups where each data point spans 1 or more consecutive chunks.

    The dataset is designed for SEQUENTIAL access (shuffle=false) with
    contiguous DP sharding. Data points within the same group have the same
    chunk count, ensuring uniform batch shapes.
    """

    # Minimum distance (in chunks) between primary and distill chunks
    DISTILL_MIN_DISTANCE = 1000

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        chunk_length: int,
        group_dict: Dict[int, float],
        global_batch_size: int,
    ):
        """
        Args:
            model_path: Path to the pretrained model / tokenizer directory.
            cache_dir: Directory containing the preprocessed token cache.
            chunk_length: Number of tokens per chunk.
            group_dict: {num_chunks_per_point: proportion} mapping.
            global_batch_size: Total batch size across all nodes.
        """
        super().__init__(model_path)
        self.chunk_length = chunk_length
        self.cache_dir = cache_dir
        self.group_dict = group_dict
        self.global_batch_size = global_batch_size

        _check_verified_marker(cache_dir)

        manifest_path = os.path.join(cache_dir, MANIFEST_FILE)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        num_shards = manifest["num_shards"]

        eos_id = self.special_tokens["<|endoftext|>"]
        mmap_path = os.path.join(cache_dir, "token_stream.npy")
        lock_path = os.path.join(cache_dir, "token_stream.lock")
        verified_path = os.path.join(cache_dir, VERIFIED_FILE)

        t0 = time.time()

        # Build or load mmap (same logic as oldpretrainmask)
        def _mmap_is_valid():
            if not os.path.exists(mmap_path):
                return False
            try:
                m = np.load(mmap_path, mmap_mode="r")
                mt = os.path.getmtime(mmap_path)
                vt = os.path.getmtime(verified_path)
                ok = m.dtype == np.int32 and m.ndim == 1 and m.shape[0] > 0 and mt >= vt
                del m
                return ok
            except Exception:
                return False

        if not _mmap_is_valid():
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    if not _mmap_is_valid():
                        logger.info(f"[oldpretrainmask_group] Building mmap from {num_shards} shards ...")
                        shard_infos = []
                        total_stream_len = 0
                        for si in range(num_shards):
                            sp = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
                            data = np.load(sp)
                            lengths = data["lengths"]
                            n_tokens = int(lengths.astype(np.int64).sum())
                            n_nonempty = int((lengths > 0).sum())
                            shard_infos.append((sp, n_tokens, n_nonempty))
                            total_stream_len += n_tokens + n_nonempty
                            del data, lengths

                        import socket
                        mmap_tmp = f"{mmap_path}.tmp.{socket.gethostname()}.{os.getpid()}"
                        fp = np.lib.format.open_memmap(mmap_tmp, mode="w+", dtype=np.int32, shape=(total_stream_len,))

                        write_pos = 0
                        for si, (sp, n_tokens, n_nonempty) in enumerate(shard_infos):
                            if n_tokens == 0:
                                continue
                            data = np.load(sp)
                            tokens = data["tokens"]
                            lengths = data["lengths"]
                            read_pos = 0
                            for length in lengths:
                                length = int(length)
                                if length == 0:
                                    continue
                                fp[write_pos:write_pos + length] = tokens[read_pos:read_pos + length]
                                write_pos += length
                                fp[write_pos] = eos_id
                                write_pos += 1
                                read_pos += length
                            del data, tokens, lengths
                            if (si + 1) % 100 == 0:
                                fp.flush()

                        assert write_pos == total_stream_len
                        fp.flush()
                        del fp
                        os.rename(mmap_tmp, mmap_path)
                        logger.info(f"[oldpretrainmask_group] Mmap built: {total_stream_len:,} tokens")
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        token_stream = np.load(mmap_path, mmap_mode="r")
        total_stream_len = token_stream.shape[0]
        logger.info(f"[oldpretrainmask_group] Loaded mmap: {total_stream_len:,} tokens ({time.time()-t0:.1f}s)")

        # Chunk the token stream
        total_chunks = total_stream_len // chunk_length
        if total_chunks == 0:
            raise ValueError(f"Total tokens ({total_stream_len}) < chunk_length ({chunk_length})")

        used_tokens = total_chunks * chunk_length
        self._all_chunks = token_stream[:used_tokens].reshape(total_chunks, chunk_length)
        self._token_stream_mmap = token_stream
        self._total_chunks = total_chunks

        # Compute group layout
        self._groups = _compute_group_layout(total_chunks, group_dict, global_batch_size)

        # Build flat index: each entry is (group_idx, point_idx_within_group)
        self._flat_index: List[Tuple[int, int]] = []
        for gi, g in enumerate(self._groups):
            for pi in range(g["num_data_points"]):
                self._flat_index.append((gi, pi))

        # Compute distill offset per group
        for g in self._groups:
            num_points = g["num_data_points"]
            g["distill_offset"] = min(self.DISTILL_MIN_DISTANCE, num_points // 2)

        # Log summary
        total_points = len(self._flat_index)
        total_chunks_used = sum(g["total_chunks_used"] for g in self._groups)
        logger.info(
            f"[oldpretrainmask_group] {total_points:,} data points across {len(self._groups)} groups, "
            f"using {total_chunks_used:,}/{total_chunks:,} chunks, "
            f"global_batch_size={global_batch_size}"
        )
        for g in self._groups:
            logger.info(
                f"  Group: {g['num_chunks_per_point']} chunks/point, "
                f"{g['num_data_points']:,} points, "
                f"chunks [{g['chunk_start']}:{g['chunk_end']}], "
                f"distill_offset={g['distill_offset']}"
            )

    def __len__(self) -> int:
        return len(self._flat_index)

    def get_num_chunks_for_idx(self, idx: int) -> int:
        """Return the number of chunks for the data point at the given index."""
        gi, _ = self._flat_index[idx]
        return self._groups[gi]["num_chunks_per_point"]

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get a multi-chunk data point.

        Returns:
            List of dicts, length = num_chunks_per_point for this data point.
            Each dict contains:
                - chunk: (chunk_length,) LongTensor of unmasked tokens
                - distill_chunk: (chunk_length,) LongTensor for distillation
        """
        gi, pi = self._flat_index[idx]
        g = self._groups[gi]
        num_c = g["num_chunks_per_point"]
        chunk_start = g["chunk_start"]
        distill_offset = g["distill_offset"]
        num_points = g["num_data_points"]

        # The consecutive chunks for this data point
        first_chunk_idx = chunk_start + pi * num_c

        # Distill: use a distant data point within the same group
        distill_pi = (pi + distill_offset) % num_points
        distill_first_chunk_idx = chunk_start + distill_pi * num_c

        result = []
        for c_offset in range(num_c):
            chunk_idx = first_chunk_idx + c_offset
            distill_chunk_idx = distill_first_chunk_idx + c_offset

            chunk = torch.from_numpy(np.array(self._all_chunks[chunk_idx], dtype=np.int64))
            distill_chunk = torch.from_numpy(np.array(self._all_chunks[distill_chunk_idx], dtype=np.int64))

            result.append({
                "chunk": chunk,
                "distill_chunk": distill_chunk,
            })

        return result


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class OldPretrainMaskGroupCollator(BaseCollator):
    """
    Collator for OldPretrainMaskGroupDataset.

    Processes each chunk in the multi-chunk data point independently using
    the same masking logic as OldPretrainMaskCollator.

    Returns a list of dicts (length = num_chunks_per_point), where each dict
    has the same format as OldPretrainMaskCollator's output.
    """

    def __init__(
        self,
        model_path: str,
        context_max_length: int,
        conversation_max_length: int,
        pad_token_id: int,
        num_mem_token: int = 0,
        mask_ratio_start: float = 0.0,
        mask_ratio_end: float = 0.0,
        mask_ratio_std: float = 0.0,
        mask_strategy: str = "random",
        span_mean_length: int = 3,
        span_max_length: int = 10,
        tokenizer_cfg=None,
    ):
        super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)
        if context_max_length != conversation_max_length:
            raise ValueError(
                f"context_max_length ({context_max_length}) must equal "
                f"conversation_max_length ({conversation_max_length})"
            )
        self.chunk_length = context_max_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

        # Dynamic mask ratio schedule
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.mask_ratio_std = mask_ratio_std
        self.mask_strategy = mask_strategy
        self.span_mean_length = span_mean_length
        self.span_max_length = span_max_length

        # Training progress state
        self._current_step = 0
        self._max_steps = 1
        self._is_eval = False

        # Get special token IDs
        if "<MASK>" not in self.special_tokens:
            raise ValueError("<MASK> token not found in special_tokens.")
        self._mask_token_id = self.special_tokens["<MASK>"]
        self._end_think_id = self.special_tokens["</think>"]
        self._im_end_id = self.special_tokens["<|im_end|>"]
        self._im_start_id = self.special_tokens["<|im_start|>"]
        self._eos_id = self.special_tokens["<|endoftext|>"]

        _tokenizer = create_tokenizer(model_path, tokenizer_cfg=tokenizer_cfg)
        _nn_ids = _tokenizer.encode("\n\n", add_special_tokens=False)
        if len(_nn_ids) == 1:
            self._newline_newline_id = _nn_ids[0]
        else:
            self._newline_newline_id = 271
            logger.warning(
                f"[OldPretrainMaskGroupCollator] '\\n\\n' encodes to {_nn_ids}, "
                f"using fallback ID 271"
            )
        del _tokenizer

        logger.info(
            f"[OldPretrainMaskGroupCollator] mask_ratio={mask_ratio_start}->{mask_ratio_end}, "
            f"std={mask_ratio_std}, strategy={mask_strategy}"
        )

    def set_training_progress(self, current_step: int, max_steps: int):
        """Update training progress for dynamic mask ratio scheduling."""
        self._current_step = current_step
        self._max_steps = max(1, max_steps)

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode. When True, no masking is applied."""
        self._is_eval = is_eval

    def _get_current_mask_ratio(self) -> float:
        """Compute the mask ratio for the current step."""
        progress = min(1.0, max(0.0, self._current_step / self._max_steps))
        mean_ratio = self.mask_ratio_start + (self.mask_ratio_end - self.mask_ratio_start) * progress
        if self.mask_ratio_std > 0:
            sampled_ratio = random.gauss(mean_ratio, self.mask_ratio_std)
        else:
            sampled_ratio = mean_ratio
        return max(0.0, min(1.0, sampled_ratio))

    def _process_single_chunk(
        self,
        chunk: torch.Tensor,
        distill_chunk: torch.Tensor,
        effective_mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single chunk: compute context_ids, labels, distill_labels.

        Returns:
            (context_ids_row, conversation_ids_row, labels_row,
             distill_conversation_ids_row, distill_labels_row)
        """
        L = self.chunk_length
        num_mem = self.num_mem_token
        context_total_len = L + num_mem

        # Context and conversation
        context_row = torch.full((context_total_len,), self.pad_token_id, dtype=torch.long)
        labels_row = torch.full((L,), -100, dtype=torch.long)

        # Find text content ranges
        content_ranges = _find_text_content_ranges(
            chunk, self._end_think_id, self._newline_newline_id,
            self._im_end_id, self._im_start_id, self._eos_id,
        )

        # Labels: only text content + trailing <|im_end|>
        for start, end in content_ranges:
            labels_row[start:end] = chunk[start:end]
            if end < L and chunk[end].item() == self._im_end_id:
                labels_row[end] = chunk[end]

        # Context: masked chunk
        if effective_mask_ratio > 0 and content_ranges:
            if self.mask_strategy == "random":
                masked_chunk = _random_mask(chunk, content_ranges, self._mask_token_id, effective_mask_ratio)
            elif self.mask_strategy == "span":
                masked_chunk = _span_mask(
                    chunk, content_ranges, self._mask_token_id,
                    effective_mask_ratio, self.span_mean_length, self.span_max_length
                )
            else:
                raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")
            context_row[:L] = masked_chunk
        else:
            context_row[:L] = chunk

        # Distill labels
        distill_labels_row = torch.full((L,), -100, dtype=torch.long)
        distill_content_ranges = _find_text_content_ranges(
            distill_chunk, self._end_think_id, self._newline_newline_id,
            self._im_end_id, self._im_start_id, self._eos_id,
        )
        for start, end in distill_content_ranges:
            distill_labels_row[start:end] = distill_chunk[start:end]
            if end < L and distill_chunk[end].item() == self._im_end_id:
                distill_labels_row[end] = distill_chunk[end]

        return context_row, chunk, labels_row, distill_chunk, distill_labels_row

    def __call__(self, samples: List[List[Dict[str, torch.Tensor]]]) -> List[Dict[str, torch.Tensor]]:
        """
        Collate a batch of multi-chunk data points.

        Args:
            samples: List of data points, each is a list of chunk dicts.
                All data points in the batch must have the same number of chunks.

        Returns:
            List of dicts (length = num_chunks_per_point), where each dict contains:
                - context_ids: (B, context_total_len)
                - conversation_ids: (B, chunk_length)
                - labels: (B, chunk_length)
                - context_lengths: (B,)
                - distill: {conversation_ids: (B, chunk_length), labels: (B, chunk_length)}
        """
        batch_size = len(samples)
        if batch_size == 0:
            return []

        num_chunks_per_point = len(samples[0])
        # Verify all samples have the same number of chunks
        for s in samples:
            assert len(s) == num_chunks_per_point, (
                f"All samples in a batch must have the same number of chunks. "
                f"Expected {num_chunks_per_point}, got {len(s)}"
            )

        L = self.chunk_length
        num_mem = self.num_mem_token
        context_total_len = L + num_mem

        # Determine mask ratio
        if self._is_eval:
            effective_mask_ratio = 0.0
        else:
            effective_mask_ratio = self._get_current_mask_ratio()

        # Process each chunk position independently
        result_list = []
        for c_idx in range(num_chunks_per_point):
            context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
            conversation_ids = torch.empty((batch_size, L), dtype=torch.long)
            labels = torch.full((batch_size, L), -100, dtype=torch.long)
            context_lengths = torch.full((batch_size,), L, dtype=torch.long)
            distill_conversation_ids = torch.empty((batch_size, L), dtype=torch.long)
            distill_labels = torch.full((batch_size, L), -100, dtype=torch.long)

            for i in range(batch_size):
                chunk_dict = samples[i][c_idx]
                chunk = chunk_dict["chunk"]
                distill_chunk = chunk_dict["distill_chunk"]

                ctx_row, conv_row, lbl_row, d_conv_row, d_lbl_row = self._process_single_chunk(
                    chunk, distill_chunk, effective_mask_ratio
                )

                context_ids[i] = ctx_row
                conversation_ids[i] = conv_row
                labels[i] = lbl_row
                distill_conversation_ids[i] = d_conv_row
                distill_labels[i] = d_lbl_row

            result_list.append({
                "context_ids": context_ids,
                "conversation_ids": conversation_ids,
                "labels": labels,
                "context_lengths": context_lengths,
                "distill": {
                    "conversation_ids": distill_conversation_ids,
                    "labels": distill_labels,
                },
            })

        return result_list


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create OldPretrainMaskGroupDataset and OldPretrainMaskGroupCollator.

    Requires cfg.data.group_dict and cfg.training.pp_batchsize.local_batch_size to be set.
    global_batch_size is computed as local_batch_size * num_nodes.
    """
    data_cfg = cfg.data
    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length

    if context_seq_len != conv_seq_len:
        raise ValueError(
            f"context_seq_length ({context_seq_len}) must equal conv_seq_length ({conv_seq_len})"
        )

    _raw_cache = data_cfg.get("cache_dir", "cache/oldpretrainmask_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)

    # Parse group_dict from config
    group_dict_cfg = data_cfg.group_dict
    group_dict = {}
    for k, v in group_dict_cfg.items():
        group_dict[int(k)] = float(v)

    # Compute global_batch_size = local_batch_size * num_nodes
    training_cfg = cfg.training
    # Support both new nested config (pp_batchsize.local_batch_size) and legacy flat config
    if hasattr(training_cfg, "pp_batchsize") and training_cfg.pp_batchsize is not None:
        local_batch_size = training_cfg.pp_batchsize.local_batch_size
    else:
        local_batch_size = training_cfg.local_batch_size

    # num_nodes = world_size / gpus_per_node
    # In our setup: PP within node, DP across nodes
    import torch.distributed as dist
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gpus_per_node = cfg.parallel.total_gpus
        num_nodes = world_size // gpus_per_node
    else:
        num_nodes = 1

    global_batch_size = local_batch_size * num_nodes

    # Read mask settings
    mask_cfg = data_cfg.mask
    mask_ratio_start = mask_cfg.ratio_start
    mask_ratio_end = mask_cfg.ratio_end
    mask_ratio_std = mask_cfg.ratio_std
    mask_strategy = mask_cfg.strategy
    span_mean_length = mask_cfg.get("span_mean_length", 3)
    span_max_length = mask_cfg.get("span_max_length", 10)

    logger.info(
        f"[oldpretrainmask_group] Creating dataset: chunk_length={context_seq_len}, "
        f"cache='{cache_dir}', group_dict={group_dict}, "
        f"global_batch_size={global_batch_size} (local={local_batch_size} x nodes={num_nodes}), "
        f"mask_ratio={mask_ratio_start}->{mask_ratio_end}"
    )

    train_ds = OldPretrainMaskGroupDataset(
        model_path=model_path,
        cache_dir=cache_dir,
        chunk_length=context_seq_len,
        group_dict=group_dict,
        global_batch_size=global_batch_size,
    )

    collator = OldPretrainMaskGroupCollator(
        model_path=model_path,
        context_max_length=context_seq_len,
        conversation_max_length=conv_seq_len,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
        mask_ratio_start=mask_ratio_start,
        mask_ratio_end=mask_ratio_end,
        mask_ratio_std=mask_ratio_std,
        mask_strategy=mask_strategy,
        span_mean_length=span_mean_length,
        span_max_length=span_max_length,
        tokenizer_cfg=cfg.tokenizer,
    )

    return train_ds, collator


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Debug: inspect samples from each group.

    For each group, produces one file: {dataset_name}_group{gi}_{num_c}chunks_debug.txt
    Within each file, data points are shown sequentially, and for each data point
    all chunk positions (0, 1, 2, ...) are printed in order with per-token tables.
    """
    from utils.mydata import resolve_pad_token_id
    from omegaconf import OmegaConf

    if "model" not in cfg:
        cfg = OmegaConf.merge(cfg, {"model": {"path": model_path}})

    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    num_mem_token = 10

    dataset, collator = create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer)

    mask_cfg = cfg.data.mask
    dataset_name = cfg.data.get("name", "oldpretrainmask_group")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    num_samples_per_group = 3

    for gi, g in enumerate(dataset._groups):
        num_c = g["num_chunks_per_point"]

        # Find first data point index in this group
        start_idx = sum(
            dataset._groups[j]["num_data_points"]
            for j in range(gi)
        )

        # Open output file for this group
        file_name = f"{dataset_name}_group{gi}_{num_c}chunks_debug.txt"
        output_path = os.path.join(output_dir, file_name)
        _log_file = open(output_path, "w", encoding="utf-8")

        def _print(*args, **kwargs):
            import builtins
            builtins.print(*args, **kwargs, file=_log_file)
            _log_file.flush()
            builtins.print(*args, **kwargs)

        sep = "=" * 120
        _print(sep)
        _print(f"  DEBUG — Dataset: {dataset_name}, Group {gi} ({num_c} chunks/point)")
        _print(sep)
        _print(f"  Total data points in group: {g['num_data_points']}")
        _print(f"  Chunk range: [{g['chunk_start']}:{g['chunk_end']}]")
        _print(f"  Inspecting: {min(num_samples_per_group, g['num_data_points'])} data points")
        _print(f"  num_mem_token : {num_mem_token}")
        _print(f"  pad_token_id  : {pad_token_id}")
        _print(f"  context_seq_len: {cfg.data.context_seq_length}")
        _print(f"  conv_seq_len  : {cfg.data.conv_seq_length}")
        _print(f"  mask_ratio_start: {mask_cfg.ratio_start}")
        _print(f"  mask_ratio_end: {mask_cfg.ratio_end}")
        _print(f"  mask_strategy : {mask_cfg.strategy}")
        _print(sep)

        n = min(num_samples_per_group, g["num_data_points"])

        for sample_offset in range(n):
            idx = start_idx + sample_offset
            sample = dataset[idx]  # List[Dict], length = num_c

            # Collate this single sample
            collated = collator([sample])  # List[Dict], length = num_c

            _print(f"\n{'━' * 120}")
            _print(f"  Data Point {sample_offset} (dataset index {idx}), {num_c} chunk(s)")
            _print(f"{'━' * 120}")

            for c_idx in range(num_c):
                batch = collated[c_idx]
                ctx_ids = batch["context_ids"][0]        # (ctx_total_len,)
                conv_ids = batch["conversation_ids"][0]  # (conv_len,)
                labels = batch["labels"][0]              # (conv_len,)
                ctx_len = batch["context_lengths"][0].item()

                # Distill data
                distill_data = batch.get("distill")
                has_distill = (distill_data is not None and isinstance(distill_data, dict)
                               and "conversation_ids" in distill_data and "labels" in distill_data)

                _print(f"\n  {'─' * 100}")
                _print(f"  Chunk {c_idx}/{num_c}  |  context_ids={list(ctx_ids.shape)}, "
                       f"conversation_ids={list(conv_ids.shape)}, labels={list(labels.shape)}, "
                       f"context valid tokens={ctx_len}")
                if has_distill:
                    _print(f"  Distill: conversation_ids={list(distill_data['conversation_ids'][0].shape)}, "
                           f"labels={list(distill_data['labels'][0].shape)}")
                _print(f"  {'─' * 100}")

                # Per-token table
                max_pos = max(ctx_ids.size(0), conv_ids.size(0))
                w_idx = 5
                w_tok = 20
                w_id = 10
                w_note = 20

                header = (f"{'Idx':>{w_idx}} | "
                          f"{'Ctx Token':<{w_tok}} | {'Ctx ID':>{w_id}} | "
                          f"{'Conv Token':<{w_tok}} | {'Conv ID':>{w_id}} | "
                          f"{'Label Token':<{w_tok}} | {'Label ID':>{w_id}} | {'Note':<{w_note}}")
                _print(f"  {header}")
                _print(f"  {'-' * len(header)}")

                for pos in range(max_pos):
                    notes = []

                    # Context column
                    if pos < ctx_ids.size(0):
                        cid = ctx_ids[pos].item()
                        ctok = tokenizer.decode([cid])
                        ctok = repr(ctok) if ctok.strip() == "" else ctok
                        cid_str = str(cid)
                        if pos >= ctx_len and pos < ctx_len + num_mem_token:
                            notes.append("[MEM]")
                        elif pos >= ctx_len + num_mem_token and cid == pad_token_id:
                            notes.append("[CTX_PAD]")
                    else:
                        ctok = ""
                        cid_str = ""

                    # Conversation / Labels columns
                    is_mem_only = (pos >= conv_ids.size(0) and pos >= ctx_len
                                   and pos < ctx_len + num_mem_token)
                    if is_mem_only:
                        vtok = ""
                        vid_str = ""
                        ltok = ""
                        lid_str = ""
                    elif pos < conv_ids.size(0):
                        vid = conv_ids[pos].item()
                        vtok = tokenizer.decode([vid])
                        vtok = repr(vtok) if vtok.strip() == "" else vtok
                        vid_str = str(vid)
                        lid = labels[pos].item()
                        lid_str = str(lid)
                        if lid == -100:
                            ltok = ""
                            notes.append("[MASKED]")
                        else:
                            ltok = tokenizer.decode([lid])
                            ltok = repr(ltok) if ltok.strip() == "" else ltok
                        if vid == pad_token_id:
                            notes.append("[CONV_PAD]")
                    else:
                        vtok = ""
                        vid_str = ""
                        ltok = ""
                        lid_str = ""

                    # Truncate long tokens
                    ctok = ctok[:w_tok] if len(ctok) > w_tok else ctok
                    vtok = vtok[:w_tok] if len(vtok) > w_tok else vtok
                    ltok = ltok[:w_tok] if len(ltok) > w_tok else ltok
                    note_str = " ".join(notes)

                    _print(f"  {pos:>{w_idx}} | "
                           f"{ctok:<{w_tok}} | {cid_str:>{w_id}} | "
                           f"{vtok:<{w_tok}} | {vid_str:>{w_id}} | "
                           f"{ltok:<{w_tok}} | {lid_str:>{w_id}} | {note_str:<{w_note}}")

                # Distill table for this chunk
                if has_distill:
                    distill_conv = distill_data["conversation_ids"][0]
                    distill_labels = distill_data["labels"][0]

                    _print(f"\n  {'·' * 60}")
                    _print(f"  Distill data for Chunk {c_idx}")
                    _print(f"  {'·' * 60}")

                    d_header = (f"{'Idx':>{w_idx}} | "
                                f"{'Distill Conv Token':<{w_tok}} | {'Distill Conv ID':>{w_id}} | "
                                f"{'Distill Label Token':<{w_tok}} | {'Distill Label ID':>{w_id}} | {'Note':<{w_note}}")
                    _print(f"  {d_header}")
                    _print(f"  {'-' * len(d_header)}")

                    for pos in range(distill_conv.size(0)):
                        d_notes = []
                        d_vid = distill_conv[pos].item()
                        d_vtok = tokenizer.decode([d_vid])
                        d_vtok = repr(d_vtok) if d_vtok.strip() == "" else d_vtok
                        d_vid_str = str(d_vid)

                        d_lid = distill_labels[pos].item()
                        d_lid_str = str(d_lid)
                        if d_lid == -100:
                            d_ltok = ""
                            d_notes.append("[MASKED]")
                        else:
                            d_ltok = tokenizer.decode([d_lid])
                            d_ltok = repr(d_ltok) if d_ltok.strip() == "" else d_ltok

                        if d_vid == pad_token_id:
                            d_notes.append("[PAD]")

                        d_vtok = d_vtok[:w_tok] if len(d_vtok) > w_tok else d_vtok
                        d_ltok = d_ltok[:w_tok] if len(d_ltok) > w_tok else d_ltok
                        d_note_str = " ".join(d_notes)

                        _print(f"  {pos:>{w_idx}} | "
                               f"{d_vtok:<{w_tok}} | {d_vid_str:>{w_id}} | "
                               f"{d_ltok:<{w_tok}} | {d_lid_str:>{w_id}} | {d_note_str:<{w_note}}")

        _print(f"\n{sep}")
        _print(f"  DEBUG complete — Group {gi} ({num_c} chunks/point), {n} data points inspected")
        _print(f"  Output saved to: {output_path}")
        _print(sep)

        _log_file.close()
        print(f"\n  [Group {gi}] Debug output written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="oldpretrainmask_group dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true")
    group.add_argument("--preprocess", action="store_true")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/oldpretrainmask_group.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    data_cfg = OmegaConf.load(args.config)
    _base_yaml = os.path.join("configs", "base.yaml")
    _base_cfg = OmegaConf.load(_base_yaml) if os.path.exists(_base_yaml) else OmegaConf.create({})
    _dataset_seed = _base_cfg.get("seed", {}).get("dataset", 42)

    # For debug, we need training config to compute global_batch_size
    # Load from main_pretrain.yaml defaults
    training_cfg = OmegaConf.create({"pp_batchsize": {"local_batch_size": 8}})
    parallel_cfg = OmegaConf.create({"total_gpus": 8, "pipeline_parallel_size": 8})
    for _cfg_name in ["main_pretrain.yaml"]:
        _main_yaml = os.path.join("configs", _cfg_name)
        if os.path.exists(_main_yaml):
            main_cfg = OmegaConf.load(_main_yaml)
            for d in main_cfg.get("defaults", []):
                if isinstance(d, dict) and "training" in d:
                    t_cfg_path = os.path.join("configs", "training", f"{d['training']}.yaml")
                    if os.path.exists(t_cfg_path):
                        training_cfg = OmegaConf.load(t_cfg_path)
            break

    cfg = OmegaConf.create({
        "data": data_cfg,
        "seed": {"dataset": _dataset_seed},
        "training": training_cfg,
        "parallel": parallel_cfg,
    })

    model_path = args.model_path
    if model_path is None:
        for _cfg_name in ["main_pretrain.yaml", "main_sft.yaml"]:
            _main_yaml = os.path.join("configs", _cfg_name)
            if os.path.exists(_main_yaml):
                main_cfg = OmegaConf.load(_main_yaml)
                for d in main_cfg.get("defaults", []):
                    if isinstance(d, dict) and "model" in d:
                        model_cfg_path = os.path.join("configs", "model", f"{d['model']}.yaml")
                        if os.path.exists(model_cfg_path):
                            model_path = OmegaConf.load(model_cfg_path).get("path", None)
                if model_path is not None:
                    break

    if model_path is not None:
        cfg = OmegaConf.merge(cfg, {"model": {"path": model_path}})

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.preprocess:
        if model_path is None:
            print("ERROR: --model_path required for --preprocess")
            sys.exit(1)
        # Reuse oldpretrainmask's preprocess (same cache format)
        preprocess(cfg, model_path)
    elif args.debug:
        if model_path is None:
            print("ERROR: --model_path required for --debug")
            sys.exit(1)
        debug(cfg, model_path)
