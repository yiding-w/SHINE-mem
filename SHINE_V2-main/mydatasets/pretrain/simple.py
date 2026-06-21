#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple (Overfitting) Dataset Module for SHINE_V2

Designed for verifying code correctness by overfitting on a small dataset.
Each data point has:
  - Identical context and conversation token sequences (same random tokens)
  - Forced equal context_length and conv_length (configurable)
  - Configurable number of samples

All tuneable parameters come from configs/data/pretrain/simple.yaml.

Unified factory interface:
    create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
        -> (SimpleOverfitDataset, SimpleOverfitCollator)
"""

import os
import random
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

# Ensure project root is on sys.path so that local package imports work
import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.base import BaseDataset, BaseCollator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple Overfitting Dataset
# ---------------------------------------------------------------------------

class SimpleOverfitDataset(BaseDataset):
    """
    Dataset for overfitting tests.

    Each sample has IDENTICAL context and conversation token sequences
    (same random tokens). This makes it trivial for the model to learn
    the mapping, allowing quick verification of training correctness.

    Each sample is a dict with:
        - context_ids:      LongTensor (seq_length,)
        - conversation_ids: LongTensor (seq_length,)
        - labels:           LongTensor (seq_length,)
        - context_lengths:  int (always == seq_length, no padding needed)
    """

    def __init__(
        self,
        model_path: str,
        num_samples: int = 32,
        seq_length: int = 256,
        vocab_size: int = 50000,
        seed: int = 42,
    ):
        super().__init__(model_path)
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Use a fixed seed for reproducibility across ranks
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Pre-generate all samples
        self.data: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_samples):
            # Generate one random token sequence
            tokens = torch.randint(0, vocab_size, (seq_length,), generator=gen)

            # Context and conversation are IDENTICAL
            context_ids = tokens.clone()
            conversation_ids = tokens.clone()

            # Labels: same as conversation_ids (next-token prediction)
            # Mask out the first token position (no prediction target)
            labels = tokens.clone()
            labels[0] = -100

            self.data.append({
                "context_ids": context_ids,
                "conversation_ids": conversation_ids,
                "labels": labels,
                "context_lengths": seq_length,
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class SimpleOverfitCollator(BaseCollator):
    """
    Collator for the simple overfitting dataset.

    Since all samples have the same fixed seq_length, no padding is needed.
    Simply stacks samples into a batch.

    Produced batch keys:
        context_ids:      (B, context_total_len)  where context_total_len = seq_length + num_mem_token
        conversation_ids: (B, seq_length)
        labels:           (B, seq_length)
        context_lengths:  (B,)  always == seq_length
    """

    def __init__(
        self,
        model_path: str,
        seq_length: int = 256,
        pad_token_id: int = 0,
        num_mem_token: int = 0,
    ):
        super().__init__(model_path)
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        batch_size = len(samples)
        seq_len = self.seq_length
        num_mem = self.num_mem_token

        # context_total_len includes space for mem_token placeholders
        context_total_len = seq_len + num_mem

        # Pre-allocate tensors
        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        context_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        for i, s in enumerate(samples):
            # Layout: [valid_tokens | mem_placeholders (pad)]
            context_ids[i, :seq_len] = s["context_ids"][:seq_len]
            conversation_ids[i, :seq_len] = s["conversation_ids"][:seq_len]
            labels[i, :seq_len] = s["labels"][:seq_len]

        # Distillation: use a rotated version of the batch
        shift = random.randint(1, max(1, batch_size - 1))
        distill_indices = [(i + shift) % batch_size for i in range(batch_size)]
        distill_conversation_ids = conversation_ids[distill_indices].clone()
        distill_labels = labels[distill_indices].clone()

        return [{
            "context_ids": context_ids,
            "conversation_ids": conversation_ids,
            "labels": labels,
            "context_lengths": context_lengths,
            "distill": {
                "conversation_ids": distill_conversation_ids,
                "labels": distill_labels,
            },
        }]


# ---------------------------------------------------------------------------
# Factory function — unified interface
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a SimpleOverfitDataset and its collator.

    Parameters from cfg.data (simple.yaml):
        - num_samples:        number of samples (small for overfitting)
        - context_seq_length: sequence length (used for both context and conversation)
        - conv_seq_length:    must equal context_seq_length (enforced)
        - vocab_size:         vocabulary size for random token generation
        - seed:               random seed for reproducibility

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (SimpleOverfitDataset, SimpleOverfitCollator)
    """
    data_cfg = cfg.data

    num_samples = data_cfg.get("num_samples", 32)
    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length
    vocab_size = data_cfg.get("vocab_size", 50000)
    dataset_seed = cfg.seed.dataset

    # Enforce equal lengths for this overfitting dataset
    if context_seq_len != conv_seq_len:
        raise ValueError(
            f"[simple] context_seq_length ({context_seq_len}) must equal "
            f"conv_seq_length ({conv_seq_len}) for the simple overfitting dataset."
        )

    seq_length = context_seq_len

    logger.info(
        f"[simple] Creating overfitting dataset: "
        f"num_samples={num_samples}, seq_length={seq_length}, "
        f"vocab_size={vocab_size}, seed={dataset_seed}"
    )

    dataset = SimpleOverfitDataset(
        model_path=model_path,
        num_samples=num_samples,
        seq_length=seq_length,
        vocab_size=vocab_size,
        seed=dataset_seed,
    )

    collator = SimpleOverfitCollator(
        model_path=model_path,
        seq_length=seq_length,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
    )

    return dataset, collator


# ---------------------------------------------------------------------------
# Preprocess — nothing to do for this synthetic dataset
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """No preprocessing needed for simple overfitting data."""
    import io
    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "simple")

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
    _print(f"  Config file   : configs/data/pretrain/{dataset_name}.yaml")
    _print(f"  Status        : No preprocessing needed (synthetic overfitting data)")
    _print(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[simple] Overfitting dataset requires no preprocessing. Output: {output_path}")


# ---------------------------------------------------------------------------
# Debug — inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Create the simple dataset + collator, then call the generic
    ``debug_dataset`` utility to print aligned per-token tables.
    """
    from utils.mydata import resolve_pad_token_id, debug_dataset
    from utils.mytokenizer import create_tokenizer

    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    num_mem_token = 10

    dataset, collator = create_dataset_and_collator(
        cfg, model_path, pad_token_id, num_mem_token,
    )

    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer)

    data_cfg = cfg.data
    metadata = {
        "seq_length": data_cfg.context_seq_length,
        "num_samples": data_cfg.get("num_samples", 32),
        "vocab_size": data_cfg.get("vocab_size", 50000),
        "seed": cfg.seed.dataset,
        "note": "context == conversation (identical tokens)",
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("name", "simple"),
        metadata=metadata,
        num_samples=5,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/simple.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Simple overfitting dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Preprocess (no-op)")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/simple.yaml",
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.preprocess:
        preprocess(cfg, model_path)
    elif args.debug:
        if model_path is None:
            print("ERROR: --model_path is required for --debug mode (or set in configs/model/*.yaml).")
            sys.exit(1)
        debug(cfg, model_path)
