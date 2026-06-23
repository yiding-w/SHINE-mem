#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MS MARCO Multi-QA SFT Dataset Module (Preserve Thinking) for SHINE_V2

This module provides a supervised fine-tuning (SFT) dataset based on the
MS MARCO Multi-QA dataset, using the preserve_thinking mode (default chat
template with preserve_thinking=True).

Data format (JSONL):
    Each line is a JSON object with:
        - query_id: int
        - context: str (the evidence passage)
        - conversations: list of {question: str, answer: str}

Preprocessing:
    Tokenizes all samples using the tokenizer's default chat template
(preserve_thinking mode) with preserve_thinking=True.
    
    Context is tokenized as a chat template:
        [{"role": "user", "content": "<CTX>"}, {"role": "assistant", "content": context}]
    with preserve_thinking=True.
    
    Conversation includes thinking tokens. Labels mask everything EXCEPT
    tokens from after <think>\\n\\n to <|im_end|> (inclusive).

Dataset:
    On construction, loads pre-tokenized shards and builds an index for
    random access. Each sample returns pre-tokenized context_ids and
    conversation token ids with labels.

Collator:
    Pads/truncates to fixed lengths and produces the unified batch format:
        - context_ids:      (B, context_total_len)  where context_total_len = context_seq_length + num_mem_token
        - conversation_ids: (B, conv_seq_length)
        - labels:           (B, conv_seq_length)  with -100 for masked tokens
        - context_lengths:  (B,)  actual context lengths (before padding)
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
from torch.utils.data import Dataset

# Ensure project root is on sys.path so that local package imports work
import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.base import BaseDataset, BaseCollator
from utils.mytokenizer import create_tokenizer
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARD_PREFIX = "shard_"
MANIFEST_FILE = "manifest.json"
VERIFIED_FILE = "VERIFIED"
VERIFY_SAMPLE_SIZE = 200
TEXTS_PER_SHARD = 1000

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_train_data(data_cfg) -> List[Dict]:
    """
    Load and concatenate training JSONL files.

    Args:
        data_cfg: Data configuration with data_path and train_files.

    Returns:
        List of sample dicts.
    """
    data_path = data_cfg.get("data_path", "old_data/ms_marco_mqa")
    abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)

    train_files = data_cfg.get("train_files", ["train.jsonl", "train_v2.jsonl"])
    all_items = []
    for fname in train_files:
        fpath = os.path.join(abs_data_path, fname)
        items = _load_jsonl(fpath)
        logger.info(f"[msmarco_mqa_preserve_thinking] Loaded {len(items)} samples from {fname}")
        all_items.extend(items)

    logger.info(f"[msmarco_mqa_preserve_thinking] Total training samples: {len(all_items)}")
    return all_items


def _load_val_data(data_cfg) -> List[Dict]:
    """
    Load validation JSONL file.

    Args:
        data_cfg: Data configuration with data_path and val_file.

    Returns:
        List of sample dicts.
    """
    data_path = data_cfg.get("data_path", "old_data/ms_marco_mqa")
    abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)

    val_file = data_cfg.get("val_file", "val.jsonl")
    fpath = os.path.join(abs_data_path, val_file)
    items = _load_jsonl(fpath)
    logger.info(f"[msmarco_mqa_preserve_thinking] Loaded {len(items)} validation samples from {val_file}")
    return items


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------

def _format_messages(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert a multi-turn QA conversation into chat-template messages.

    Args:
        conversations: List of {question: str, answer: str} dicts.

    Returns:
        List of {"role": "user"/"assistant", "content": ...} message dicts.
    """
    messages = []
    for turn in conversations:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    return messages


def _build_label_mask_preserve_thinking(
    tokenizer,
    conv_token_ids: List[int],
) -> List[int]:
    """
    Build label mask for preserve_thinking mode conversation:
    mask everything EXCEPT tokens from after <think>\\n\\n to <|im_end|> (inclusive).

    In preserve_thinking mode, assistant responses have the format:
        <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{content}<|im_end|>
    
    We unmask from the first token after </think>\\n\\n up to and including <|im_end|>.
    This means the model learns to predict the actual response content and the
    end-of-turn token, but not the thinking scaffolding.

    Args:
        tokenizer: The tokenizer instance.
        conv_token_ids: Full tokenized conversation ids (from apply_chat_template).

    Returns:
        List of label ids (same length as conv_token_ids, with -100 for masked positions).
    """
    imend_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # Encode the pattern "</think>\n\n" to find where content starts
    # In preserve_thinking mode with empty thinking, the pattern is:
    # <think>\n\n</think>\n\n  followed by actual content
    think_end_pattern = tokenizer.encode("</think>\n\n", add_special_tokens=False)

    labels = [-100] * len(conv_token_ids)
    ids = list(conv_token_ids)
    n = len(ids)
    i = 0

    while i < n:
        # Look for </think>\n\n pattern
        if i + len(think_end_pattern) <= n:
            match = True
            for j, pat_id in enumerate(think_end_pattern):
                if ids[i + j] != pat_id:
                    match = False
                    break
            if match:
                # Content starts after the pattern
                content_start = i + len(think_end_pattern)
                
                # Find the matching <|im_end|>
                content_end = content_start
                while content_end < n and ids[content_end] != imend_id:
                    content_end += 1

                # Unmask content tokens AND the trailing <|im_end|>
                unmask_end = content_end + 1 if content_end < n else content_end
                for j in range(content_start, unmask_end):
                    labels[j] = ids[j]

                # Skip past <|im_end|>
                i = content_end + 1
                continue
        i += 1

    return labels


# ---------------------------------------------------------------------------
# Batch tokenization worker
# ---------------------------------------------------------------------------

def _init_batch_worker(tokenizer_path: str, tokenizer_cfg_dict: dict):
    """Pool initializer: load tokenizer once per worker (preserve thinking mode)."""
    global _worker_tokenizer
    # No chat_template override — use default (preserve thinking)
    _worker_tokenizer = create_tokenizer(
        tokenizer_path, tokenizer_cfg=OmegaConf.create(tokenizer_cfg_dict)
    )


def _tokenize_shard(args: Tuple) -> Optional[str]:
    """
    Tokenize a shard of samples and save as .npz file.

    Each shard stores:
        - context_tokens: concatenated context token ids (int32)
        - context_lengths: length of each context (int32)
        - conv_tokens: concatenated conversation token ids (int32)
        - conv_lengths: length of each conversation (int32)
        - label_tokens: concatenated label ids (int32, with -100 for masked)

    Args:
        args: (shard_idx, samples_list, cache_dir, conv_seq_length)

    Returns:
        None on success, error message string on failure.
    """
    shard_idx, samples, cache_dir, conv_seq_length = args
    out_path = os.path.join(cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz")

    # Resume support
    if os.path.exists(out_path):
        return None

    try:
        global _worker_tokenizer

        all_ctx_ids = []
        all_ctx_lengths = []
        all_conv_ids = []
        all_conv_lengths = []
        all_label_ids = []

        for sample in samples:
            context = sample["context"]
            conversations = sample["conversations"]

            # Tokenize context as chat template:
            # [{"role": "user", "content": "<CTX>"}, {"role": "assistant", "content": context}]
            # with preserve_thinking=True (preserve thinking mode)
            ctx_messages = [
                {"role": "user", "content": "<CTX>"},
                {"role": "assistant", "content": context},
            ]
            ctx_ids = _worker_tokenizer.apply_chat_template(
                ctx_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
                preserve_thinking=True,
            )
            all_ctx_ids.extend(ctx_ids)
            all_ctx_lengths.append(len(ctx_ids))

            # Format as chat messages and encode with chat template (preserve thinking)
            messages = _format_messages(conversations)
            conv_ids = _worker_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
                preserve_thinking=True,
            )
            all_conv_ids.extend(conv_ids)
            all_conv_lengths.append(len(conv_ids))

            # Build labels: mask everything except content after </think>\n\n to <|im_end|>
            labels = _build_label_mask_preserve_thinking(_worker_tokenizer, conv_ids)
            all_label_ids.extend(labels)

        # Save as npz
        ctx_tokens = np.array(all_ctx_ids, dtype=np.int32)
        ctx_lengths = np.array(all_ctx_lengths, dtype=np.int32)
        conv_tokens = np.array(all_conv_ids, dtype=np.int32)
        conv_lengths = np.array(all_conv_lengths, dtype=np.int32)
        label_tokens = np.array(all_label_ids, dtype=np.int32)

        tmp_path = out_path + ".tmp.npz"
        np.savez(
            tmp_path,
            context_tokens=ctx_tokens,
            context_lengths=ctx_lengths,
            conv_tokens=conv_tokens,
            conv_lengths=conv_lengths,
            label_tokens=label_tokens,
        )
        os.rename(tmp_path, out_path)
        return None
    except Exception as e:
        return f"[shard={shard_idx}] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_variant(data_cfg, model_path: str, train_data: List[Dict],
                        variant_cache_dir: str,
                        num_workers: int, texts_per_shard: int,
                        conv_seq_length: int, _log):
    """
    Tokenize all training samples and cache as shard .npz files.

    Uses preserve_thinking mode (default chat template with preserve_thinking=True).

    Args:
        data_cfg: Data configuration dict.
        model_path: Absolute path to the model / tokenizer directory.
        train_data: List of raw training samples.
        variant_cache_dir: Cache directory for this variant.
        num_workers: Number of parallel workers.
        texts_per_shard: Number of samples per shard.
        conv_seq_length: Conversation sequence length.
        _log: Logging function.
    """
    os.makedirs(variant_cache_dir, exist_ok=True)

    # Remove stale markers
    verified_path = os.path.join(variant_cache_dir, VERIFIED_FILE)
    if os.path.exists(verified_path):
        os.remove(verified_path)
        _log(f"[msmarco_mqa_preserve_thinking] Removed stale VERIFIED marker")

    num_samples = len(train_data)
    num_shards = (num_samples + texts_per_shard - 1) // texts_per_shard
    _log(f"[msmarco_mqa_preserve_thinking] {num_samples} samples → {num_shards} shards")

    # ---- Write / verify manifest ----
    manifest_path = os.path.join(variant_cache_dir, MANIFEST_FILE)
    manifest = {
        "num_samples": num_samples,
        "num_shards": num_shards,
        "texts_per_shard": texts_per_shard,
        "model_path": str(model_path),
        "preserve_thinking_template": True,
    }

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing = json.load(f)
        if existing.get("num_samples") != num_samples:
            raise ValueError(
                f"Manifest mismatch: existing cache has {existing.get('num_samples')} samples "
                f"but current data has {num_samples}. Delete cache_dir '{variant_cache_dir}' and re-run."
            )
        _log(f"[msmarco_mqa_preserve_thinking] Manifest verified: {num_samples} samples, {num_shards} shards")
    else:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        _log(f"[msmarco_mqa_preserve_thinking] Manifest written: {manifest_path}")

    # ---- Count already-cached shards ----
    already_done = sum(
        1 for si in range(num_shards)
        if os.path.exists(os.path.join(variant_cache_dir, f"{SHARD_PREFIX}{si:06d}.npz"))
    )
    remaining = num_shards - already_done
    _log(f"[msmarco_mqa_preserve_thinking] Cache status: {already_done}/{num_shards} shards done, {remaining} remaining")

    if remaining > 0:
        _log(f"[msmarco_mqa_preserve_thinking] Tokenizing and caching ...")
        t0 = time.time()
        errors: List[str] = []

        # Build work items
        work_items = []
        for si in range(num_shards):
            if os.path.exists(os.path.join(variant_cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")):
                continue
            start = si * texts_per_shard
            end = min(start + texts_per_shard, num_samples)
            batch = train_data[start:end]
            work_items.append((si, batch, variant_cache_dir, conv_seq_length))

        if num_workers <= 1:
            _init_batch_worker(model_path)
            for idx, item in enumerate(work_items):
                err = _tokenize_shard(item)
                if err is not None:
                    errors.append(err)
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                _log(f"[msmarco_mqa_preserve_thinking]   {idx + 1}/{remaining} [{rate:.1f} shards/s, {elapsed:.1f}s elapsed]")
        else:
            with mp.Pool(
                processes=num_workers,
                initializer=_init_batch_worker,
                initargs=(model_path, OmegaConf.to_container(cfg.tokenizer, resolve=True)),
            ) as pool:
                results = pool.map(_tokenize_shard, work_items)
                for r in results:
                    if r is not None:
                        errors.append(r)

        elapsed = time.time() - t0
        _log(f"[msmarco_mqa_preserve_thinking] Tokenization complete: {len(work_items)} shards in {elapsed:.1f}s")

        if errors:
            for e in errors[:10]:
                _log(f"[ERROR] {e}")
            raise RuntimeError(f"Tokenization failed for {len(errors)} shards.")
    else:
        _log(f"[msmarco_mqa_preserve_thinking] All shards already cached, skipping tokenization.")

    # ---- Verification ----
    _log(f"[msmarco_mqa_preserve_thinking] Starting verification ...")
    _verify_cache(variant_cache_dir, num_samples, num_shards)
    _log(f"[msmarco_mqa_preserve_thinking] Preprocessing complete. Cache is verified and ready.")


def preprocess(cfg, model_path: str):
    """
    Tokenize all training samples and cache as shard .npz files.

    Uses preserve_thinking mode (default chat template).

    Args:
        cfg: Hydra config (must have cfg.data).
        model_path: Absolute path to the model / tokenizer directory.
    """
    data_cfg = cfg.data
    _raw_cache = data_cfg.get("cache_dir", "cache/msmarco_mqa_preserve_thinking_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)
    num_workers = data_cfg.get("preprocess_workers", max(1, mp.cpu_count() // 2))
    texts_per_shard = data_cfg.get("texts_per_shard", TEXTS_PER_SHARD)
    conv_seq_length = data_cfg.get("conv_seq_length", 1120)

    os.makedirs(cache_dir, exist_ok=True)

    # ---- Open .txt log file ----
    dataset_name = data_cfg.get("name", "msmarco_mqa_preserve_thinking")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{dataset_name}_preprocess.txt")
    _log_file = open(output_path, "w", encoding="utf-8")

    def _log(msg: str):
        logger.info(msg)
        _log_file.write(msg + "\n")
        _log_file.flush()

    sep = "=" * 80
    _log_file.write(f"{sep}\n")
    _log_file.write(f"  PREPROCESS — Dataset: {dataset_name}\n")
    _log_file.write(f"  Mode: preserve_thinking (default template, preserve_thinking=True)\n")
    _log_file.write(f"{sep}\n")
    _log_file.write(f"  Config file       : configs/data/sft/{dataset_name}.yaml\n")
    _log_file.write(f"  Model path        : {model_path}\n")
    _log_file.write(f"  Cache dir         : {cache_dir}\n")
    _log_file.write(f"  Workers           : {num_workers}\n")
    _log_file.write(f"  Texts per shard   : {texts_per_shard}\n")
    _log_file.write(f"{sep}\n\n")
    _log_file.flush()

    # ---- Load training data ----
    _log("[msmarco_mqa_preserve_thinking] Loading training data ...")
    train_data = _load_train_data(data_cfg)
    num_samples = len(train_data)
    num_shards = (num_samples + texts_per_shard - 1) // texts_per_shard
    _log(f"[msmarco_mqa_preserve_thinking] Data loaded: {num_samples} samples → {num_shards} shards")

    # ---- Process (preserve_thinking template) ----
    try:
        _preprocess_variant(
            data_cfg=data_cfg,
            model_path=model_path,
            train_data=train_data,
            variant_cache_dir=cache_dir,
            num_workers=num_workers,
            texts_per_shard=texts_per_shard,
            conv_seq_length=conv_seq_length,
            _log=_log,
        )
    except Exception as e:
        _log(f"[ERROR] {e}")
        _log_file.close()
        raise

    del train_data

    _log_file.write(f"\n{sep}\n")
    _log_file.write(f"  FINAL SUMMARY\n")
    _log_file.write(f"{sep}\n")
    _log_file.write(f"  Total samples     : {num_samples}\n")
    _log_file.write(f"  Shards            : {num_shards}\n")
    _log_file.write(f"  Template          : preserve_thinking ✓\n")
    _log_file.write(f"  Status            : VERIFIED ✓\n")
    _log_file.write(f"{sep}\n")
    _log_file.flush()
    _log_file.close()

    logger.info(f"[msmarco_mqa_preserve_thinking] Preprocess log written to: {output_path}")


def _verify_cache(cache_dir: str, num_samples: int, num_shards: int):
    """
    Verify all shard files exist and have consistent internal structure.
    Write VERIFIED marker on success.
    """
    t0 = time.time()

    # Check all shards exist
    missing = []
    for si in range(num_shards):
        p = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
        if not os.path.exists(p):
            missing.append(si)
    if missing:
        raise RuntimeError(f"Verification failed: {len(missing)} shard files missing.")

    # Remove stale tmp files
    tmp_files = [f for f in os.listdir(cache_dir) if ".tmp" in f]
    for tf in tmp_files:
        os.remove(os.path.join(cache_dir, tf))

    # Check internal consistency
    total_sample_count = 0
    total_ctx_tokens = 0
    total_conv_tokens = 0

    for si in range(num_shards):
        p = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
        data = np.load(p)
        ctx_lengths = data["context_lengths"]
        conv_lengths = data["conv_lengths"]
        ctx_tokens = data["context_tokens"]
        conv_tokens = data["conv_tokens"]
        label_tokens = data["label_tokens"]

        n = len(ctx_lengths)
        assert len(conv_lengths) == n, f"Shard {si}: ctx/conv length count mismatch"
        assert int(ctx_lengths.astype(np.int64).sum()) == len(ctx_tokens), \
            f"Shard {si}: context token count mismatch"
        assert int(conv_lengths.astype(np.int64).sum()) == len(conv_tokens), \
            f"Shard {si}: conv token count mismatch"
        assert len(label_tokens) == len(conv_tokens), \
            f"Shard {si}: label/conv token count mismatch"

        total_sample_count += n
        total_ctx_tokens += len(ctx_tokens)
        total_conv_tokens += len(conv_tokens)

    assert total_sample_count == num_samples, \
        f"Total sample count {total_sample_count} != expected {num_samples}"

    # Write VERIFIED marker
    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    verified_info = {
        "num_samples": num_samples,
        "num_shards": num_shards,
        "total_context_tokens": total_ctx_tokens,
        "total_conv_tokens": total_conv_tokens,
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(verified_path, "w") as f:
        json.dump(verified_info, f, indent=2)

    elapsed = time.time() - t0
    logger.info(
        f"[msmarco_mqa_preserve_thinking] Verification complete in {elapsed:.1f}s: "
        f"{num_samples} samples, {total_ctx_tokens:,} ctx tokens, "
        f"{total_conv_tokens:,} conv tokens"
    )


# ---------------------------------------------------------------------------
# MSMarcoMQAPreserveThinkingDataset
# ---------------------------------------------------------------------------

class MSMarcoMQAPreserveThinkingDataset(BaseDataset):
    """
    SFT dataset for MS MARCO Multi-QA (preserve thinking mode).

    Loads pre-tokenized shard files and provides random access to individual
    samples. Each sample contains pre-tokenized context, conversation, and
    label token ids.
    """

    def __init__(self, model_path: str, cache_dir: str):
        """
        Args:
            model_path: Path to the pretrained model / tokenizer directory.
            cache_dir: Directory containing tokenized shard cache files.
        """
        super().__init__(model_path)

        self.cache_dir = cache_dir

        # Check VERIFIED marker
        verified_path = os.path.join(cache_dir, VERIFIED_FILE)
        if not os.path.exists(verified_path):
            raise RuntimeError(
                f"VERIFIED marker not found in '{cache_dir}'. "
                f"Run preprocess first."
            )

        # Load manifest
        manifest_path = os.path.join(cache_dir, MANIFEST_FILE)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.num_samples = manifest["num_samples"]
        self.num_shards = manifest["num_shards"]
        self.texts_per_shard = manifest.get("texts_per_shard", TEXTS_PER_SHARD)

        # Build index: for each sample, store (shard_idx, local_idx)
        # We also preload all shard data for fast access
        self._ctx_tokens_list = []  # list of np arrays per shard
        self._ctx_lengths_list = []
        self._conv_tokens_list = []
        self._conv_lengths_list = []
        self._label_tokens_list = []

        logger.info(f"[msmarco_mqa_preserve_thinking] Loading {self.num_shards} shards from {cache_dir} ...")
        t0 = time.time()
        for si in range(self.num_shards):
            p = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
            data = np.load(p)
            self._ctx_tokens_list.append(data["context_tokens"])
            self._ctx_lengths_list.append(data["context_lengths"])
            self._conv_tokens_list.append(data["conv_tokens"])
            self._conv_lengths_list.append(data["conv_lengths"])
            self._label_tokens_list.append(data["label_tokens"])

        elapsed = time.time() - t0
        logger.info(f"[msmarco_mqa_preserve_thinking] Loaded {self.num_samples} samples in {elapsed:.1f}s")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample by index.

        Returns:
            Dict with:
                - context_ids: 1-D LongTensor of context token ids
                - conv_ids: 1-D LongTensor of conversation token ids
                - label_ids: 1-D LongTensor of label ids (-100 for masked)
        """
        shard_idx = idx // self.texts_per_shard
        local_idx = idx % self.texts_per_shard

        # Get context tokens
        ctx_lengths = self._ctx_lengths_list[shard_idx]
        ctx_offset = int(ctx_lengths[:local_idx].astype(np.int64).sum())
        ctx_len = int(ctx_lengths[local_idx])
        ctx_ids = self._ctx_tokens_list[shard_idx][ctx_offset:ctx_offset + ctx_len]

        # Get conversation tokens
        conv_lengths = self._conv_lengths_list[shard_idx]
        conv_offset = int(conv_lengths[:local_idx].astype(np.int64).sum())
        conv_len = int(conv_lengths[local_idx])
        conv_ids = self._conv_tokens_list[shard_idx][conv_offset:conv_offset + conv_len]

        # Get label tokens
        label_ids = self._label_tokens_list[shard_idx][conv_offset:conv_offset + conv_len]

        return {
            "context_ids": torch.from_numpy(np.array(ctx_ids, dtype=np.int64)),
            "conv_ids": torch.from_numpy(np.array(conv_ids, dtype=np.int64)),
            "label_ids": torch.from_numpy(np.array(label_ids, dtype=np.int64)),
        }


# ---------------------------------------------------------------------------
# MSMarcoMQAPreserveThinkingCollator
# ---------------------------------------------------------------------------

class MSMarcoMQAPreserveThinkingCollator(BaseCollator):
    """
    Collator for MSMarcoMQAPreserveThinkingDataset.

    Pads/truncates context and conversation to fixed lengths.

    Output batch (unified format):
        - context_ids:      (B, context_seq_length + num_mem_token)
        - conversation_ids: (B, conv_seq_length)
        - labels:           (B, conv_seq_length)  with -100 for padding and masked tokens
        - context_lengths:  (B,)  actual context lengths (capped at context_seq_length)
    """

    def __init__(
        self,
        model_path: str,
        context_max_length: int,
        conversation_max_length: int,
        pad_token_id: int,
        num_mem_token: int = 0,
    ):
        super().__init__(model_path)

        self.context_max_length = context_max_length
        self.conversation_max_length = conversation_max_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        batch_size = len(samples)
        ctx_len = self.context_max_length
        conv_len = self.conversation_max_length
        num_mem = self.num_mem_token
        context_total_len = ctx_len + num_mem

        # Pre-allocate tensors
        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.full((batch_size, conv_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, conv_len), -100, dtype=torch.long)
        context_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, s in enumerate(samples):
            # Context: truncate or pad
            ctx = s["context_ids"]
            actual_ctx_len = min(len(ctx), ctx_len)
            context_ids[i, :actual_ctx_len] = ctx[:actual_ctx_len]
            context_lengths[i] = actual_ctx_len

            # Conversation: truncate or pad
            conv = s["conv_ids"]
            lbl = s["label_ids"]
            actual_conv_len = min(len(conv), conv_len)
            conversation_ids[i, :actual_conv_len] = conv[:actual_conv_len]
            labels[i, :actual_conv_len] = lbl[:actual_conv_len]

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

def _resolve_cache_dir(data_cfg) -> str:
    """
    Resolve the cache directory for tokenized data.

    Returns:
        Absolute path to the cache directory.
    """
    _raw_cache = data_cfg.get("cache_dir", "cache/msmarco_mqa_preserve_thinking_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)
    return cache_dir


def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create MSMarcoMQAPreserveThinkingDataset and Collator from the data configuration.

    Uses preserve_thinking mode (default chat template).

    Args:
        cfg: Full Hydra DictConfig.
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (MSMarcoMQAPreserveThinkingDataset, MSMarcoMQAPreserveThinkingCollator)
    """
    data_cfg = cfg.data

    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length

    cache_dir = _resolve_cache_dir(data_cfg)

    logger.info(
        f"[msmarco_mqa_preserve_thinking] Creating dataset: context_seq_len={context_seq_len}, "
        f"conv_seq_len={conv_seq_len}, "
        f"cache_dir='{cache_dir}', num_mem_token={num_mem_token}"
    )

    # Build dataset
    train_ds = MSMarcoMQAPreserveThinkingDataset(
        model_path=model_path,
        cache_dir=cache_dir,
    )

    # Build collator
    collator = MSMarcoMQAPreserveThinkingCollator(
        model_path=model_path,
        context_max_length=context_seq_len,
        conversation_max_length=conv_seq_len,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
    )

    return train_ds, collator


# ---------------------------------------------------------------------------
# Validation dataset (loaded directly from val.jsonl, no caching needed)
# ---------------------------------------------------------------------------

class MSMarcoMQAPreserveThinkingValDataset(BaseDataset):
    """
    Validation dataset for MS MARCO Multi-QA (preserve thinking mode).

    Loads val.jsonl directly and tokenizes on-the-fly (since val set is small).
    Returns the same format as MSMarcoMQAPreserveThinkingDataset for compatibility.
    """

    def __init__(self, model_path: str, data_cfg, tokenizer_cfg=None):
        super().__init__(model_path, tokenizer_cfg=tokenizer_cfg)

        # No chat_template override — use default (preserve thinking)
        self.tokenizer = create_tokenizer(model_path, tokenizer_cfg=tokenizer_cfg)
        self.items = _load_val_data(data_cfg)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        context = item["context"]
        conversations = item["conversations"]

        # Tokenize context as chat template (preserve thinking mode)
        ctx_messages = [
            {"role": "user", "content": "<CTX>"},
            {"role": "assistant", "content": context},
        ]
        ctx_ids = self.tokenizer.apply_chat_template(
            ctx_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            preserve_thinking=True,
        )

        # Format as chat messages and encode with chat template (preserve thinking)
        messages = _format_messages(conversations)
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            preserve_thinking=True,
        )

        # Build labels: mask everything except content after </think>\n\n to <|im_end|>
        labels = _build_label_mask_preserve_thinking(self.tokenizer, conv_ids)

        return {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "conv_ids": torch.tensor(conv_ids, dtype=torch.long),
            "label_ids": torch.tensor(labels, dtype=torch.long),
        }


def create_val_dataset(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a validation dataset from val.jsonl.

    Args:
        cfg: Full Hydra DictConfig.
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        MSMarcoMQAPreserveThinkingValDataset (or a Subset of it), or None if
        val_file is not configured / not found.
    """
    from torch.utils.data import Subset

    data_cfg = cfg.data
    val_file = data_cfg.get("val_file", None)
    if not val_file:
        return None

    data_path = data_cfg.get("data_path", "old_data/ms_marco_mqa")
    abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)
    val_path = os.path.join(abs_data_path, val_file)

    if not os.path.exists(val_path):
        logger.warning(f"[msmarco_mqa_preserve_thinking] Validation file not found: {val_path}")
        return None

    logger.info(f"[msmarco_mqa_preserve_thinking] Creating validation dataset from {val_file}")
    full_val_ds = MSMarcoMQAPreserveThinkingValDataset(model_path=model_path, data_cfg=data_cfg, tokenizer_cfg=cfg.tokenizer)

    # --- Apply validation_subset_num if configured ---
    validation_subset_num = data_cfg.get("validation_subset_num", -1)
    total_val = len(full_val_ds)

    # Parse the subset size
    subset_size = _parse_val_subset_num(validation_subset_num, total_val)

    if subset_size <= 0 or subset_size >= total_val:
        logger.info(
            f"[msmarco_mqa_preserve_thinking] Using full validation set: {total_val} samples "
            f"(validation_subset_num={validation_subset_num})"
        )
        return full_val_ds

    # Select a reproducible subset using the global dataset seed
    try:
        dataset_seed = cfg.seed.dataset
    except Exception:
        from omegaconf import OmegaConf as _OC
        _base_path = os.path.join(_project_root, "configs", "base.yaml")
        _base = _OC.load(_base_path) if os.path.exists(_base_path) else _OC.create({})
        dataset_seed = _base.get("seed", {}).get("dataset", 42)
    rng = random.Random(dataset_seed)
    all_indices = list(range(total_val))
    rng.shuffle(all_indices)
    subset_indices = sorted(all_indices[:subset_size])

    logger.info(
        f"[msmarco_mqa_preserve_thinking] Validation subset: {subset_size}/{total_val} samples "
        f"(seed={dataset_seed}, validation_subset_num={validation_subset_num})"
    )
    return Subset(full_val_ds, subset_indices)


def _parse_val_subset_num(validation_subset_num, total_samples: int) -> int:
    """
    Parse validation_subset_num into an actual sample count.

    Args:
        validation_subset_num: -1 (use all), int >= 1 (exact count),
                               or float in (0, 1) (fraction).
        total_samples: Total number of validation samples.

    Returns:
        Number of samples to use (0 means use all).
    """
    if isinstance(validation_subset_num, int):
        if validation_subset_num == -1:
            return 0  # means "use all"
        if validation_subset_num >= 1:
            return min(validation_subset_num, total_samples)
        return 0
    elif isinstance(validation_subset_num, float):
        if 0.0 < validation_subset_num < 1.0:
            return max(1, int(total_samples * validation_subset_num))
        return 0
    else:
        return 0


# ---------------------------------------------------------------------------
# Debug — inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Debug the dataset (preserve_thinking template).

    Creates the dataset + collator, then calls the generic
    debug_dataset utility to print aligned per-token tables.

    Args:
        cfg: Hydra config.
        model_path: Path to the model / tokenizer directory.
    """
    from utils.mydata import resolve_pad_token_id, debug_dataset
    from omegaconf import OmegaConf

    if "model" not in cfg:
        cfg = OmegaConf.merge(cfg, {"model": {"path": model_path}})

    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    num_mem_token = 10
    data_cfg = cfg.data

    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  DEBUG — preserve_thinking template")
    print(f"{sep}\n")

    # Resolve cache dir
    cache_dir = _resolve_cache_dir(data_cfg)

    # Check if cache exists
    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    if not os.path.exists(verified_path):
        print(f"  [SKIP] Not preprocessed yet.")
        print(f"         Expected cache: {cache_dir}")
        print(f"         Run --preprocess first.")
        return

    dataset, collator = create_dataset_and_collator(
        cfg, model_path, pad_token_id, num_mem_token,
    )

    # Use default tokenizer (preserve thinking)
    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer)

    metadata = {
        "context_seq_len": data_cfg.context_seq_length,
        "conv_seq_len": data_cfg.conv_seq_length,
        "cache_dir": cache_dir,
        "num_samples": len(dataset),
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=f"{data_cfg.get('name', 'msmarco_mqa_preserve_thinking')}_preserve_thinking",
        metadata=metadata,
        num_samples=5,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
        strict_shape_check=False,
    )

    del dataset, collator, tokenizer

    print(f"\n{sep}")
    print(f"  DEBUG COMPLETE.")
    print(f"{sep}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="msmarco_mqa_preserve_thinking SFT dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Parallel tokenize and cache")
    parser.add_argument("--config", type=str, default="configs/data/sft/msmarco_mqa_preserve_thinking.yaml",
                        help="Path to data config YAML")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model directory (for tokenizer)")
    args = parser.parse_args()

    # Load config
    data_cfg = OmegaConf.load(args.config)
    # Load base.yaml to get the canonical seed.dataset value
    _base_yaml = os.path.join("configs", "base.yaml")
    _base_cfg = OmegaConf.load(_base_yaml) if os.path.exists(_base_yaml) else OmegaConf.create({})
    _dataset_seed = _base_cfg.get("seed", {}).get("dataset", 42)
    cfg = OmegaConf.create({"data": data_cfg, "seed": {"dataset": _dataset_seed}})

    # Resolve model_path
    model_path = args.model_path
    if model_path is None:
        # Try main_pretrain.yaml first, then main_sft.yaml
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.preprocess:
        if model_path is None:
            print("ERROR: --model_path is required for --preprocess")
            sys.exit(1)
        preprocess(cfg, model_path)
    elif args.debug:
        if model_path is None:
            print("ERROR: --model_path is required for --debug")
            sys.exit(1)
        debug(cfg, model_path)
