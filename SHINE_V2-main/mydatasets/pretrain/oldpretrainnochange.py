#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Old Pretrain No-Change Dataset Module for SHINE_V2

This module provides a grouped text dataset for pretraining where texts are
grouped together by token length to maximize packing efficiency, and a collator
that applies chat-template formatting with reconstruction/completion tasks.

Dataset:
    GroupTextDataset groups raw texts by token length into bins that fit within
    conversation_max_len.  Supports optional data warmup with progressive
    length buckets.  Preprocessing builds and caches the group index (JSON).

Collator:
    GroupPretrainNoChangeCollator applies chat-template formatting:
      - Each text is randomly assigned as <RECON> (reconstruction) or <COMP>
        (completion) task.
      - Evidence is the concatenation of (possibly truncated) texts.
      - Conversation is the chat-template formatted multi-turn dialogue.
      - Labels mask non-assistant tokens with -100.

Unified batch format (output of collator):
    - context_ids:      (B, context_max_length + num_mem_token)  evidence tokens + mem placeholders
    - conversation_ids: (B, conversation_max_length)
    - labels:           (B, conversation_max_length)  non-assistant tokens masked with -100
    - context_lengths:  (B,)  actual evidence token count per sample
"""

from __future__ import annotations

import os
import json
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

# Ensure project root is on sys.path so that local package imports
# (e.g. ``mydatasets.base``, ``utils.mydata``) work when this file is
# executed directly (python mydatasets/oldpretrainnochange.py ...).
import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from datasets import load_dataset, load_from_disk
from mydatasets.base import BaseDataset, BaseCollator
from utils.mytokenizer import create_tokenizer, NOTHINKING_CHAT_TEMPLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GroupTextDataset(Dataset):
    """
    DDP-safe dataset that groups texts by token length for efficient packing.

    - __init__ only loads the precomputed group_idx cache.
    - preprocess() builds and saves the group_idx cache.

    Each sample returns {"textlist": [str, ...]} — a list of texts in one group.
    """

    def __init__(
        self,
        texts,
        tokenizer,
        conversation_max_len: int,
        cache_dir: str,
        cache_name: str,
        map_num_proc: int = 16,
        map_batch_size: int = 2048,
        num_cache: int = 100,
        preprocess_mode: bool = False,
        overwrite: bool = False,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.conversation_max_len = conversation_max_len
        self.cache_dir = cache_dir
        self.cache_name = cache_name

        # Preprocess settings stored on instance
        self.map_num_proc = map_num_proc
        self.map_batch_size = map_batch_size
        self.num_cache = num_cache

        self.cache_path = os.path.join(
            cache_dir, f"{cache_name}_group_idx_{conversation_max_len}.json"
        )

        if preprocess_mode:
            self.preprocess(overwrite=overwrite)

        # ---- init ONLY loads cache ----
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"Cache not found: {self.cache_path}\n"
                f"Create it first by calling dataset.preprocess() "
                f"(single process / rank0 before DDP)."
            )

        with open(self.cache_path, "r") as f:
            self.group_idx = json.load(f)

        print(
            f"[GroupTextDataset] Loaded {len(self.group_idx)} groups "
            f"for {len(self.texts)} texts. max_len={conversation_max_len}"
        )

    # ------------------------------------------------------------------ #
    # Instance preprocess: compute group_idx + save to cache
    # ------------------------------------------------------------------ #
    def preprocess(self, overwrite: bool = False) -> List[List[int]]:
        """
        Build group_idx from self.texts and save to self.cache_path.

        Call ONCE before DDP training (single process / rank0).
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(self.cache_path) and not overwrite:
            with open(self.cache_path, "r") as f:
                self.group_idx = json.load(f)
            print(f"[preprocess] Cache exists, loaded: {self.cache_path}")
            return self.group_idx

        print("[preprocess] Creating group_idx...")

        self.base_len = 0
        self.chat_len = 11

        # ----------------- Compute token lengths ----------------- #
        print("[preprocess] Computing token lengths with HF Dataset.map...")
        token_lens = self._compute_token_lengths_with_hf_dataset()

        max_body_len = self.conversation_max_len - self.base_len
        self.group_idx: List[List[int]] = []
        cache_group_idx = [[] for _ in range(self.num_cache)]
        cache_left_len = [max_body_len for _ in range(self.num_cache)]

        for i, tok_len in enumerate(token_lens):
            if i % 10000 == 0:
                print(f"[preprocess] processing {i}/{len(token_lens)}")

            l = int(tok_len) + self.chat_len

            if l > max_body_len:
                self.group_idx.append([i])
                continue

            success = False
            for j, leftl in enumerate(cache_left_len):
                if l <= leftl:
                    cache_group_idx[j].append(i)
                    cache_left_len[j] -= l
                    success = True
                    break

            if not success:
                t = int(np.argmin(cache_left_len))
                if cache_group_idx[t]:
                    self.group_idx.append(cache_group_idx[t])
                cache_group_idx[t] = [i]
                cache_left_len[t] = max_body_len - l

        for j in range(self.num_cache):
            if cache_group_idx[j]:
                self.group_idx.append(cache_group_idx[j])

        with open(self.cache_path, "w") as f:
            json.dump(self.group_idx, f)

        print(f"[preprocess] Saved group_idx to {self.cache_path}")
        print(
            f"[preprocess] Total {len(self.group_idx)} groups including "
            f"{len(self.texts)} texts created for max_len={self.conversation_max_len}."
        )
        return self.group_idx

    # ------------------------------------------------------------------ #
    # Compute token lengths using HF Dataset.map
    # ------------------------------------------------------------------ #
    def _compute_token_lengths_with_hf_dataset(self) -> np.ndarray:
        hf_dataset = HFDataset.from_dict(
            {"text": [str(t) for t in self.texts]}
        )

        def compute_len(batch):
            enc = self.tokenizer(
                batch["text"],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return {"tok_len": [len(ids) for ids in enc["input_ids"]]}

        hf_dataset = hf_dataset.map(
            compute_len,
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.map_num_proc,
            desc="Computing token lengths",
            writer_batch_size=10,
        )

        return np.array(hf_dataset["tok_len"], dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.group_idx)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"textlist": [str(self.texts[i]) for i in self.group_idx[idx]]}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class GroupPretrainNoChangeCollator(BaseCollator):
    """
    Collator for GroupTextDataset that applies chat-template formatting with
    reconstruction/completion tasks.

    For each batch of grouped texts:
      - Randomly assigns each text as <RECON> or <COMP> task
      - Builds context_ids from evidence (with mem_token placeholders at end)
      - Builds conversation_ids via chat template with user/assistant turns
      - Masks labels to only compute loss on assistant responses

    Output batch:
        - context_ids:      (B, context_max_length + num_mem_token)
        - conversation_ids: (B, conversation_max_length)
        - labels:           (B, conversation_max_length)
        - context_lengths:  (B,)
    """

    def __init__(
        self,
        model_path: str,
        tokenizer,
        context_max_length: int = 1024,
        conversation_max_length: int = 1024,
        pad_token_id: int = 0,
        num_mem_token: int = 0,
        completion_freq: float = 0.5,
        max_completion_ratio: float = 0.3,
        min_completion_ratio: float = 0.1,
    ):
        super().__init__(model_path)
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.conversation_max_length = conversation_max_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token
        self.completion_freq = completion_freq
        self.max_completion_ratio = max_completion_ratio
        self.min_completion_ratio = min_completion_ratio

        # Resolve special token ids
        self.eot = '<|endoftext|>'
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.eot)
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.imstart_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def split_text(self, text):
        """Split text into prefix/full for completion task."""
        t = text.split()
        if len(t) < 2:
            return text, "Nothing to complete."

        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(t) * ratio)

        left = t[:split_index]
        right = t[split_index:]

        if not right:  # ensure right is not empty
            left, right = t[:-1], t[-1:]
        elif not left:
            left, right = t[:1], t[1:]

        return ' '.join(left), ' '.join(right)

    def mask_label(self, labels):
        """Mask labels to only keep assistant response tokens."""
        masks = torch.zeros_like(labels)
        for i, id in enumerate(labels):
            last_imend = self.conversation_max_length
            for j in range(len(id) - 1, 0, -1):
                if id[j].item() == self.imend_token_id:
                    last_imend = j
                elif id[j].item() == self.assistant_token_id and id[j - 1] == self.imstart_token_id:
                    masks[i, j+2: last_imend+2] = 1
        labels = labels.masked_fill(masks == 0, -100)
        return labels

    def __call__(self, batch: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        textlists = [ex["textlist"] for ex in batch]
        batch_size = len(textlists)

        user_texts_list = []
        evidence_texts_list = []
        answer_texts_list = []
        for texts in textlists:
            tlist = [random.random() for _ in range(len(texts))]
            evidence_texts = []
            answer_texts = []
            user_texts = []
            for i, t in enumerate(tlist):
                if t < self.completion_freq:
                    split = self.split_text(texts[i])
                    evidence_texts.append(split[0])
                    answer_texts.append(texts[i])
                    user_texts.append("<COMP>")
                else:
                    evidence_texts.append(texts[i])
                    answer_texts.append(texts[i])
                    user_texts.append("<RECON>")
            evidence_texts_list.append(evidence_texts)
            answer_texts_list.append(answer_texts)
            user_texts_list.append(user_texts)

        # Encode evidence into context_ids with mem_token placeholders at end
        context_total_len = self.context_max_length + self.num_mem_token
        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        context_lengths = torch.zeros(batch_size, dtype=torch.long)

        evidence_texts_all = [
            self.eot.join(random.sample(evidence_texts, len(evidence_texts)))
            for evidence_texts in evidence_texts_list
        ]
        evidence_enc = self.tokenizer(
            evidence_texts_all,
            max_length=self.context_max_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
        )

        for i in range(batch_size):
            ids = evidence_enc["input_ids"][i]
            length = len(ids)
            context_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            context_lengths[i] = length
            # Positions [length : context_max_length] are pad
            # Positions [context_max_length : context_total_len] are mem_token placeholders (pad)

        # Build conversation messages
        messages = []
        for i in range(batch_size):
            indices = list(range(len(textlists[i])))
            random.shuffle(indices)
            msg = []
            for idx in indices:
                msg.append({"role": "user", "content": f"{user_texts_list[i][idx]}"})
                msg.append({"role": "assistant", "content": f"{answer_texts_list[i][idx]}"})
            messages.append(msg)

        # Encode conversation with chat template
        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        conversation_ids = input_enc["input_ids"]

        # Compute labels (mask non-assistant tokens)
        labels = conversation_ids.clone()
        labels = self.mask_label(labels)

        # --- Distillation: use a rotated version of the batch ---
        # Each sample i gets distillation from sample (i + shift) % batch_size
        # This ensures distillation data comes from a different data point.
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
# Data loading helper
# ---------------------------------------------------------------------------

def _get_train_dataset(data_cfg, dataset_seed: int = 42):
    """
    Return the HuggingFace Dataset object (train split only, after val split).

    Args:
        data_cfg: Data configuration dict.
        dataset_seed: Seed for reproducible train/val split.

    Returns:
        (train_texts, val_texts, num_train) — train/val text lists and count.
    """
    data_path = data_cfg.get("data_path", os.path.join("data", "transmla_pretrain_6B_tokens"))
    data_format = data_cfg.get("data_format", "hf_dataset")

    abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)

    if data_format == "hf_dataset":
        dataset = load_dataset(abs_data_path, split="train")
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=dataset_seed)
        train_texts = split_dataset["train"]["text"]
        val_texts = split_dataset["test"]["text"]
    elif data_format == "hf_disk":
        dataset = load_from_disk(abs_data_path)
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=dataset_seed)
        train_texts = split_dataset["train"]["text"]
        val_texts = split_dataset["test"]["text"]
    elif data_format == "jsonl":
        with open(abs_data_path, "r", encoding="utf-8") as f:
            item_list = [json.loads(line) for line in f]
        n = len(item_list)
        val_size = max(1, int(n * 0.0005))
        rng = random.Random(dataset_seed)
        val_indices = set(rng.sample(range(n), val_size))
        train_texts = [item_list[i]["text"] for i in range(n) if i not in val_indices]
        val_texts = [item_list[i]["text"] for i in val_indices]
    else:
        raise ValueError(f"Unknown data_format: {data_format}")

    return train_texts, val_texts, len(train_texts)


# ---------------------------------------------------------------------------
# Factory function — unified interface
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a GroupTextDataset and GroupPretrainNoChangeCollator from the data configuration.

    Args:
        cfg: Full Hydra DictConfig (has cfg.data, cfg.pretrain, etc.)
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders (unused here but kept for interface).

    Returns:
        tuple: (GroupTextDataset, GroupPretrainNoChangeCollator)
    """
    data_cfg = cfg.data

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
        f"[oldpretrainnochange] Loading data and creating dataset: "
        f"conv_seq_len={conv_seq_len}, cache_dir='{cache_dir}'"
    )

    # ---- Load data ----
    dataset_seed = cfg.seed.dataset
    train_texts, val_texts, num_train = _get_train_dataset(data_cfg, dataset_seed=dataset_seed)

    logger.info(f"[oldpretrainnochange] Train texts: {num_train}, Val texts: {len(val_texts)}")

    # ---- Build dataset ----
    train_ds = GroupTextDataset(
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

    return train_ds, collator


# ---------------------------------------------------------------------------
# Preprocess — build group_idx cache
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Build the group_idx cache for the dataset.

    This must be run once (single process / rank0) before DDP training.
    """
    import io

    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "oldpretrainnochange")

    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length

    # Cache settings
    _raw_cache = data_cfg.get("cache_dir", "old_data/transmla_pretrain_6B_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)
    cache_name = data_cfg.get("cache_name", "train")

    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer, chat_template=NOTHINKING_CHAT_TEMPLATE)

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
    _print(f"  Config file       : configs/data/pretrain/{dataset_name}.yaml")
    _print(f"  Model path        : {model_path}")
    _print(f"  Cache dir         : {cache_dir}")
    _print(f"  Conv seq length   : {conv_seq_len}")
    _print(sep)

    # ---- Load data ----
    _print("\n[oldpretrainnochange] Loading data...")
    dataset_seed = cfg.seed.dataset
    train_texts, val_texts, num_train = _get_train_dataset(data_cfg, dataset_seed=dataset_seed)
    _print(f"[oldpretrainnochange] Train texts: {num_train}, Val texts: {len(val_texts)}")

    # ---- Build dataset with preprocess_mode=True ----
    _print("\n[oldpretrainnochange] Building group_idx for train set...")
    train_ds = GroupTextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        conversation_max_len=conv_seq_len,
        cache_dir=cache_dir,
        cache_name=cache_name,
        map_num_proc=data_cfg.get("map_num_proc", 16),
        map_batch_size=data_cfg.get("map_batch_size", 2048),
        num_cache=data_cfg.get("num_cache", 100),
        preprocess_mode=True,
        overwrite=data_cfg.get("overwrite", False),
    )

    _print(f"\n[oldpretrainnochange] Train dataset: {len(train_ds)} groups")

    # Also preprocess val set
    _print("\n[oldpretrainnochange] Building group_idx for val set...")
    val_ds = GroupTextDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        conversation_max_len=conv_seq_len,
        cache_dir=cache_dir,
        cache_name="val",
        map_num_proc=data_cfg.get("map_num_proc", 16),
        map_batch_size=data_cfg.get("map_batch_size", 2048),
        num_cache=data_cfg.get("num_cache", 100),
        preprocess_mode=True,
        overwrite=data_cfg.get("overwrite", False),
    )

    _print(f"[oldpretrainnochange] Val dataset: {len(val_ds)} groups")

    # ---- Final summary ----
    _print(f"\n{sep}")
    _print(f"  FINAL SUMMARY")
    _print(sep)
    _print(f"  Total train texts : {num_train}")
    _print(f"  Train groups      : {len(train_ds)}")
    _print(f"  Val groups        : {len(val_ds)}")
    _print(f"  Status            : COMPLETE ✓")
    _print(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[oldpretrainnochange] Preprocess log written to: {output_path}")


# ---------------------------------------------------------------------------
# Debug — inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Create the dataset + collator, then call the generic ``debug_dataset``
    utility to print aligned per-token tables.

    Args:
        cfg: Hydra config.
        model_path: Path to the model / tokenizer directory.
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
        "cache_dir": data_cfg.get("cache_dir", "N/A"),
        "num_groups": len(dataset),
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("name", "oldpretrainnochange"),
        metadata=metadata,
        num_samples=100,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/oldpretrainnochange.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="oldpretrainnochange dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Build group_idx cache")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/oldpretrainnochange.yaml",
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