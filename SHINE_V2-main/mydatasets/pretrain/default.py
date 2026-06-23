#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Default (Synthetic) Dataset Module for SHINE_V2

Provides a synthetic dataset for debugging and testing purposes.
All tuneable parameters come from configs/data/pretrain/default.yaml.

Unified factory interface:
    create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
        -> (ContextConversationDataset, ContextConversationCollator)
"""

import os
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

# Ensure project root is on sys.path so that local package imports
# (e.g. ``mydatasets.base``, ``utils.mydata``) work when this file is
# executed directly (python mydatasets/default.py ...).
import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.base import BaseDataset, BaseCollator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------------------------

class ContextConversationDataset(BaseDataset):
    """
    Dataset that produces (context, conversation, label) triplets.

    Each sample is a dict with:
        - context_ids:                LongTensor (context_seq_len,)
        - conversation_ids:           LongTensor (conv_seq_len,)
        - labels:                     LongTensor (conv_seq_len,)
        - context_attention_mask:     LongTensor (context_seq_len,)  (unused by collator)
        - conversation_attention_mask:LongTensor (conv_seq_len,)     (unused by collator)

    For demonstration / testing purposes this class generates synthetic data.
    """

    def __init__(
        self,
        model_path: str,
        num_samples: int = 1000,
        context_seq_length: int = 1024,
        conv_seq_length: int = 1024,
        vocab_size: int = 50000,
        seed: int = 42,
    ):
        super().__init__(model_path)
        self.num_samples = num_samples
        self.context_seq_length = context_seq_length
        self.conv_seq_length = conv_seq_length
        self.vocab_size = vocab_size

        # Use a fixed seed for reproducibility across ranks
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Pre-generate synthetic data
        self.data: List[Dict[str, torch.Tensor]] = []
        for _ in range(num_samples):
            ctx_ids = torch.randint(0, vocab_size, (context_seq_length,), generator=gen)
            conv_ids = torch.randint(0, vocab_size, (conv_seq_length,), generator=gen)
            # Labels: shifted conversation_ids (teacher forcing)
            # Use -100 for positions that should be ignored in loss
            labels = conv_ids.clone()
            # Mask out the first token as a simple convention
            labels[0] = -100

            ctx_mask = torch.ones(context_seq_length, dtype=torch.long)
            conv_mask = torch.ones(conv_seq_length, dtype=torch.long)

            self.data.append({
                "context_ids": ctx_ids,
                "conversation_ids": conv_ids,
                "labels": labels,
                "context_attention_mask": ctx_mask,
                "conversation_attention_mask": conv_mask,
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class ContextConversationCollator(BaseCollator):
    """
    Collator that pads / truncates samples and stacks them into a batch.

    Produced batch keys:
        context_ids:        (B, context_total_len)  where context_total_len = max_context_seq_length + num_mem_token
        conversation_ids:   (B, conv_seq_len)
        labels:             (B, conv_seq_len)
        context_lengths:    (B,)  actual number of valid tokens per context sample (before mem tokens)

    **Both context and conversation are right-padded to fixed lengths** so
    that no ``attention_mask`` is needed.  SDPA can use ``is_causal=True``
    and trigger the Flash backend for both forward passes.

    **Scheme B layout for context_ids**:
        [valid_tok_0 ... valid_tok_{N-1} | mem_placeholder × num_mem_token | PAD ... PAD]
        where N = context_lengths[i], total length = max_context_seq_length + num_mem_token.

    **Layout for conversation_ids**:
        [valid_tok_0 ... valid_tok_{M-1} | PAD ... PAD]
        total length = max_conv_seq_length.  Labels use -100 for padding
        positions so the loss function ignores them automatically.

    Padding value for ids is ``pad_token_id``, for labels is ``-100``.
    """

    def __init__(
        self,
        model_path: str,
        max_context_seq_length: Optional[int] = None,
        max_conv_seq_length: Optional[int] = None,
        pad_token_id: int = 0,
        num_mem_token: int = 0,
    ):
        super().__init__(model_path)
        self.max_context_seq_length = max_context_seq_length
        self.max_conv_seq_length = max_conv_seq_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        ctx_len = self.max_context_seq_length
        conv_len = self.max_conv_seq_length

        batch_size = len(samples)
        num_mem = self.num_mem_token

        # context_total_len includes space for mem_token placeholders
        context_total_len = ctx_len + num_mem

        # Pre-allocate tensors (right-padded)
        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.full((batch_size, conv_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, conv_len), -100, dtype=torch.long)
        context_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, s in enumerate(samples):
            # Context — truncate or copy, then insert mem placeholders right after valid tokens
            c_len = min(s["context_ids"].size(0), ctx_len)
            # Layout: [valid_tokens | mem_placeholders | padding]
            context_ids[i, :c_len] = s["context_ids"][:c_len]
            context_lengths[i] = c_len

            # Conversation — right-pad to conv_len
            v_len = min(s["conversation_ids"].size(0), conv_len)
            conversation_ids[i, :v_len] = s["conversation_ids"][:v_len]
            labels[i, :v_len] = s["labels"][:v_len]

        # --- Distillation: use a rotated version of the batch ---
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
    Create a synthetic ContextConversationDataset and its collator.

    All tuneable parameters are read from ``cfg.data`` (default.yaml):
        - num_samples:        number of synthetic samples to generate
        - context_seq_length: sequence length for context input
        - conv_seq_length:    sequence length for conversation input
        - vocab_size:         vocabulary size for random token generation

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory (unused for synthetic data).
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (ContextConversationDataset, ContextConversationCollator)
    """
    data_cfg = cfg.data

    num_samples = data_cfg.get("num_samples", 10000)
    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length
    vocab_size = data_cfg.get("vocab_size", 50000)
    dataset_seed = cfg.seed.dataset

    logger.info(
        f"[default] Creating synthetic dataset: "
        f"num_samples={num_samples}, context_seq_len={context_seq_len}, "
        f"conv_seq_len={conv_seq_len}, vocab_size={vocab_size}, seed={dataset_seed}"
    )

    dataset = ContextConversationDataset(
        model_path=model_path,
        num_samples=num_samples,
        context_seq_length=context_seq_len,
        conv_seq_length=conv_seq_len,
        vocab_size=vocab_size,
        seed=dataset_seed,
    )

    collator = ContextConversationCollator(
        model_path=model_path,
        max_context_seq_length=context_seq_len,
        max_conv_seq_length=conv_seq_len,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
    )

    return dataset, collator


# ---------------------------------------------------------------------------
# Preprocess — nothing to do for synthetic data
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """No preprocessing needed for synthetic data."""
    import io
    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "default")

    # Output directory: same as this file's directory (mydatasets/)
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
    _print(f"  Status        : No preprocessing needed (synthetic data)")
    _print(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[default] Synthetic dataset requires no preprocessing. Output: {output_path}")


# ---------------------------------------------------------------------------
# Debug — inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Create the synthetic dataset + collator, then call the generic
    ``debug_dataset`` utility to print aligned per-token tables.

    Args:
        cfg: Hydra config.
        model_path: Path to the model / tokenizer directory.
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
        "context_seq_len": data_cfg.context_seq_length,
        "conv_seq_len": data_cfg.conv_seq_length,
        "num_samples": data_cfg.get("num_samples", 10000),
        "vocab_size": data_cfg.get("vocab_size", 50000),
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("name", "default"),
        metadata=metadata,
        num_samples=5,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/default.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Default (synthetic) dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Preprocess (no-op for synthetic)")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/default.yaml",
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
