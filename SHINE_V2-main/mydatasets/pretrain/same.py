#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Same Dataset Module for SHINE_V2 — PP vs TP Debug

Every sample in this dataset is IDENTICAL:
  - Same context_ids
  - Same conversation_ids (equal to context_ids)
  - Same labels

This guarantees that regardless of batch size, DP sharding, or micro-batch
splitting, every forward pass sees exactly the same data. Useful for
verifying that PP and TP produce identical results.

Unified factory interface:
    create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
        -> (SameDataset, SameCollator)
"""

import os
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset

import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.base import BaseDataset, BaseCollator

logger = logging.getLogger(__name__)


class SameDataset(BaseDataset):
    """
    Dataset where EVERY sample is identical.

    A single random token sequence is generated once (with a fixed seed),
    and every __getitem__ returns the same data. Context and conversation
    are also identical to each other.

    This ensures that no matter how data is distributed across ranks,
    every rank sees the exact same tokens.
    """

    def __init__(
        self,
        model_path: str,
        num_samples: int = 2048,
        seq_length: int = 1120,
        vocab_size: int = 50000,
        seed: int = 42,
    ):
        super().__init__(model_path)
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Generate ONE fixed token sequence
        gen = torch.Generator()
        gen.manual_seed(seed)
        # Use token ids in range [1, vocab_size) to avoid pad_token_id=0
        self.tokens = torch.randint(1, vocab_size, (seq_length,), generator=gen)

        # Context and conversation are IDENTICAL
        self.context_ids = self.tokens.clone()
        self.conversation_ids = self.tokens.clone()

        # Labels: same as conversation_ids, mask first 4 tokens
        self.labels = self.tokens.clone()
        self.labels[:4] = -100

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Every sample is the same
        return {
            "context_ids": self.context_ids.clone(),
            "conversation_ids": self.conversation_ids.clone(),
            "labels": self.labels.clone(),
            "context_lengths": self.seq_length,
        }


class SameCollator(BaseCollator):
    """
    Collator for the same dataset.

    Since all samples are identical and have the same fixed seq_length,
    no padding is needed. Simply stacks samples into a batch.

    Produced batch keys:
        context_ids:      (B, context_total_len) where context_total_len = seq_length + num_mem_token
        conversation_ids: (B, seq_length)
        labels:           (B, seq_length)
        context_lengths:  (B,) always == seq_length
    """

    def __init__(
        self,
        model_path: str,
        seq_length: int = 1120,
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

        return [{
            "context_ids": context_ids,
            "conversation_ids": conversation_ids,
            "labels": labels,
            "context_lengths": context_lengths,
        }]


# ---------------------------------------------------------------------------
# Factory function — unified interface
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a SameDataset and its collator.

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (SameDataset, SameCollator)
    """
    data_cfg = cfg.data

    num_samples = data_cfg.get("num_samples", 2048)
    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length
    vocab_size = data_cfg.get("vocab_size", 50000)
    dataset_seed = cfg.seed.dataset

    # Enforce equal lengths
    if context_seq_len != conv_seq_len:
        raise ValueError(
            f"[same] context_seq_length ({context_seq_len}) must equal "
            f"conv_seq_length ({conv_seq_len}) for the same dataset."
        )

    seq_length = context_seq_len

    logger.info(
        f"[same] Creating identical-sample dataset: "
        f"num_samples={num_samples}, seq_length={seq_length}, "
        f"vocab_size={vocab_size}, seed={dataset_seed}"
    )

    dataset = SameDataset(
        model_path=model_path,
        num_samples=num_samples,
        seq_length=seq_length,
        vocab_size=vocab_size,
        seed=dataset_seed,
    )

    collator = SameCollator(
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
    """No preprocessing needed for same dataset."""
    import io
    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "same")

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
    _print(f"  Status        : No preprocessing needed (all samples identical)")
    _print(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[same] Dataset requires no preprocessing. Output: {output_path}")


# ---------------------------------------------------------------------------
# Debug — inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Create the same dataset + collator, then call the generic
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
        "num_samples": data_cfg.get("num_samples", 2048),
        "vocab_size": data_cfg.get("vocab_size", 50000),
        "seed": cfg.seed.dataset,
        "note": "ALL samples identical; context == conversation",
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("name", "same"),
        metadata=metadata,
        num_samples=3,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )
