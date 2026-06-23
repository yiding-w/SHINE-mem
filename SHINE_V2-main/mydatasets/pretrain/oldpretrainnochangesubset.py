#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Old Pretrain No-Change Subset Dataset Module for SHINE_V2

This module provides a random subset of the oldpretrainnochange dataset for
overfitting tests and quick debugging. It reuses the same GroupTextDataset and
GroupPretrainNoChangeCollator from oldpretrainnochange, but only exposes a
random subset of `num_samples` groups.

Use this to verify training correctness by overfitting on a small subset of
real data (instead of synthetic random tokens).

Unified factory interface:
    create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
        -> (SubsetGroupTextDataset, GroupPretrainNoChangeCollator)
"""

import os
import random
import logging
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

# Ensure project root is on sys.path so that local package imports work
import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.pretrain.oldpretrainnochange import (
    GroupTextDataset,
    GroupPretrainNoChangeCollator,
    _get_train_dataset,
)
from utils.mytokenizer import create_tokenizer, NOTHINKING_CHAT_TEMPLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subset wrapper
# ---------------------------------------------------------------------------

class SubsetGroupTextDataset(Dataset):
    """
    A random subset of GroupTextDataset.

    Takes the first `num_samples` groups after shuffling with a fixed seed,
    ensuring reproducibility across ranks.
    """

    def __init__(self, full_dataset: GroupTextDataset, num_samples: int, seed: int = 42):
        """
        Args:
            full_dataset: The full GroupTextDataset instance.
            num_samples: Number of groups to keep in the subset.
            seed: Random seed for reproducible subset selection
                  (should be cfg.seed.dataset).
        """
        self.full_dataset = full_dataset
        total = len(full_dataset)

        if num_samples > total:
            logger.warning(
                f"[oldpretrainnochangesubset] Requested num_samples={num_samples} "
                f"exceeds total groups={total}. Using all {total} groups."
            )
            num_samples = total

        # Deterministic random subset selection
        rng = random.Random(seed)
        all_indices = list(range(total))
        rng.shuffle(all_indices)
        self.subset_indices = sorted(all_indices[:num_samples])

        logger.info(
            f"[oldpretrainnochangesubset] Selected {len(self.subset_indices)} "
            f"groups out of {total} (seed={seed})"
        )

    def __len__(self) -> int:
        return len(self.subset_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.subset_indices[idx]
        return self.full_dataset[real_idx]


# ---------------------------------------------------------------------------
# Factory function — unified interface
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a SubsetGroupTextDataset and GroupPretrainNoChangeCollator.

    This loads the full oldpretrainnochange dataset, then takes a random
    subset of `num_samples` groups for overfitting / debugging.

    Parameters from cfg.data (oldpretrainnochangesubset.yaml):
        - num_samples:          number of groups to keep (small for overfitting)
        - seed:                 random seed for subset selection
        - context_seq_length:   sequence length for context
        - conv_seq_length:      sequence length for conversation
        - data_path:            path to the dataset
        - data_format:          format of the dataset
        - cache_dir:            cache directory for group_idx
        - cache_name:           cache file prefix
        - completion_freq:      probability of <COMP> task
        - max_completion_ratio: max ratio for completion split
        - min_completion_ratio: min ratio for completion split

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (SubsetGroupTextDataset, GroupPretrainNoChangeCollator)
    """
    data_cfg = cfg.data

    num_samples = data_cfg.get("num_samples", 40)
    dataset_seed = cfg.seed.dataset
    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length

    # Cache settings
    _raw_cache = data_cfg.get("cache_dir", "old_data/transmla_pretrain_6B_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)
    cache_name = data_cfg.get("cache_name", "train")

    # Pretrain task settings
    completion_freq = data_cfg.get("completion_freq", 0.5)
    max_completion_ratio = data_cfg.get("max_completion_ratio", 0.3)
    min_completion_ratio = data_cfg.get("min_completion_ratio", 0.1)

    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer, chat_template=NOTHINKING_CHAT_TEMPLATE)

    logger.info(
        f"[oldpretrainnochangesubset] Loading data and creating subset dataset: "
        f"num_samples={num_samples}, conv_seq_len={conv_seq_len}, seed={dataset_seed}"
    )

    # ---- Load data ----
    train_texts, val_texts, num_train = _get_train_dataset(data_cfg, dataset_seed=dataset_seed)
    logger.info(f"[oldpretrainnochangesubset] Full train texts: {num_train}")

    # ---- Build full dataset ----
    full_ds = GroupTextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        conversation_max_len=conv_seq_len,
        cache_dir=cache_dir,
        cache_name=cache_name,
        map_num_proc=data_cfg.get("map_num_proc", 16),
        map_batch_size=data_cfg.get("map_batch_size", 2048),
        num_cache=data_cfg.get("num_cache", 100),
        preprocess_mode=False,
        overwrite=False,
    )

    logger.info(f"[oldpretrainnochangesubset] Full dataset: {len(full_ds)} groups")

    # ---- Take random subset ----
    subset_ds = SubsetGroupTextDataset(full_ds, num_samples=num_samples, seed=dataset_seed)

    # ---- Build collator ----
    collator = GroupPretrainNoChangeCollator(
        model_path=model_path,
        tokenizer=tokenizer,
        context_max_length=context_seq_len,
        conversation_max_length=conv_seq_len,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
        completion_freq=completion_freq,
        max_completion_ratio=max_completion_ratio,
        min_completion_ratio=min_completion_ratio,
    )

    return subset_ds, collator


# ---------------------------------------------------------------------------
# Preprocess — delegates to oldpretrainnochange (same cache)
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Preprocess delegates to oldpretrainnochange since the subset uses the
    same group_idx cache. Just calls the parent's preprocess.
    """
    import io
    from mydatasets.pretrain.oldpretrainnochange import preprocess as parent_preprocess

    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "oldpretrainnochangesubset")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{dataset_name}_preprocess.txt")

    buf = io.StringIO()

    def _print(*args, **kwargs):
        import builtins
        builtins.print(*args, **kwargs, file=buf)
        builtins.print(*args, **kwargs)

    sep = "=" * 80
    _print(sep)
    _print(f"  PREPROCESS — Dataset: {dataset_name}")
    _print(sep)
    _print(f"  This dataset reuses the oldpretrainnochange group_idx cache.")
    _print(f"  Delegating to oldpretrainnochange.preprocess()...")
    _print(sep)

    # Delegate to parent preprocess
    parent_preprocess(cfg, model_path)

    _print(f"\n{sep}")
    _print(f"  Subset preprocess complete (cache shared with oldpretrainnochange).")
    _print(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[oldpretrainnochangesubset] Preprocess log: {output_path}")


# ---------------------------------------------------------------------------
# Debug — inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Create the subset dataset + collator, then call the generic
    ``debug_dataset`` utility to print aligned per-token tables.
    """
    from utils.mydata import resolve_pad_token_id, debug_dataset

    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    num_mem_token = 10

    dataset, collator = create_dataset_and_collator(
        cfg, model_path, pad_token_id, num_mem_token,
    )

    data_cfg = cfg.data
    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer, chat_template=NOTHINKING_CHAT_TEMPLATE)

    metadata = {
        "context_seq_len": data_cfg.context_seq_length,
        "conv_seq_len": data_cfg.conv_seq_length,
        "num_samples": data_cfg.get("num_samples", 40),
        "seed": cfg.seed.dataset,
        "num_groups_subset": len(dataset),
        "note": "Random subset of oldpretrainnochange for overfitting test",
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("name", "oldpretrainnochangesubset"),
        metadata=metadata,
        num_samples=5,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/oldpretrainnochangesubset.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="oldpretrainnochangesubset dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Build group_idx cache (delegates to parent)")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/oldpretrainnochangesubset.yaml",
                        help="Path to data config YAML")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model directory (for tokenizer and special tokens)")
    args = parser.parse_args()

    # Load config
    data_cfg = OmegaConf.load(args.config)
    # Load seed from base.yaml (single source of truth)
    _base_yaml = os.path.join("configs", "base.yaml")
    _base_cfg = OmegaConf.load(_base_yaml) if os.path.exists(_base_yaml) else OmegaConf.create({})
    _dataset_seed = _base_cfg.get("seed", {}).get("dataset", 42)
    cfg = OmegaConf.create({"data": data_cfg, "seed": {"dataset": _dataset_seed}})

    # Resolve model_path from config -> model config if not provided
    model_path = args.model_path
    if model_path is None:
        for _cfg_name in ["main_pretrain.yaml", "main_sft.yaml"]:
            _main_yaml = os.path.join("configs", _cfg_name)
            if os.path.exists(_main_yaml):
                main_cfg = OmegaConf.load(_main_yaml)
                model_cfg_name = None
                for d in main_cfg.get("defaults", []):
                    if isinstance(d, dict) and "model" in d:
                        model_cfg_name = d["model"]
                        break
                if model_cfg_name:
                    model_cfg_path = os.path.join("configs", "model", f"{model_cfg_name}.yaml")
                    if os.path.exists(model_cfg_path):
                        model_cfg = OmegaConf.load(model_cfg_path)
                        model_path = model_cfg.get("path", None)
                if model_path is not None:
                    break

    if model_path is not None:
        cfg = OmegaConf.merge(cfg, {"model": {"path": model_path}})

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.preprocess:
        if model_path is None:
            print("ERROR: --model_path is required for --preprocess (or set in configs/model/*.yaml)")
            sys.exit(1)
        preprocess(cfg, model_path)
    elif args.debug:
        if model_path is None:
            print("ERROR: --model_path is required for --debug mode (or set in configs/model/*.yaml).")
            sys.exit(1)
        debug(cfg, model_path)
