#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Old Pretrain Dataset Module for SHINE_V2

This module provides a text dataset for pretraining with efficient batch
tokenization, caching, and chunking.

Preprocessing:
    Tokenizes raw texts in batches using a streaming approach — texts are
    read from the data source in batches (texts_per_shard at a time) and
    dispatched to tokenization workers without loading the entire dataset
    into memory at once.  Each batch of tokenized results is saved as a
    ``.npz`` shard file.  Supports resume — already-cached shards are
    skipped.  After tokenization, a thorough verification pass checks every
    shard for correctness (count, content via random re-tokenization) and
    writes a ``VERIFIED`` marker.

Dataset:
    On construction, checks for the ``VERIFIED`` marker (fails fast if
    missing).  Builds a memory-mapped ``.npy`` file from cached shard files
    (concatenated with ``<|endoftext|>`` separators), then chunks the
    resulting stream into fixed-length segments.  The memory-mapped approach
    means the OS pages data in/out on demand, keeping RSS low even for
    datasets with billions of tokens.

Collator:
    Enforces ``context_max_length == conversation_max_length``.
    For each chunk:
        context_ids      = chunk  (+ mem_token placeholders)
        conversation_ids = chunk
        labels           = chunk  (all positions contribute to loss)

Unified batch format (output of collator):
    - context_ids:      (B, context_total_len)  where context_total_len = chunk_length + num_mem_token
    - conversation_ids: (B, chunk_length)
    - labels:           (B, chunk_length)
    - context_lengths:  (B,)  always == chunk_length (every chunk is full)
"""

from __future__ import annotations

import os
import json
import random
import logging
import time
import fcntl
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is on sys.path so that local package imports
# (e.g. ``mydatasets.base``, ``utils.mydata``) work when this file is
# executed directly (python mydatasets/oldpretrain.py ...).
import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from datasets import load_dataset, load_from_disk  # HuggingFace datasets
from mydatasets.base import BaseDataset, BaseCollator
from utils.mytokenizer import create_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARD_PREFIX = "shard_"          # shard files: shard_000000.npz, shard_000001.npz, ...
MANIFEST_FILE = "manifest.json"
VERIFIED_FILE = "VERIFIED"       # marker written after successful verification
VERIFY_SAMPLE_SIZE = 200         # number of random texts to re-tokenize during verification
TEXTS_PER_SHARD = 1000           # number of texts tokenized per shard file


# ---------------------------------------------------------------------------
# Batch tokenization worker
# ---------------------------------------------------------------------------

def _init_batch_worker(tokenizer_path: str, tokenizer_cfg_dict: dict):
    """Pool initializer: load tokenizer once per worker."""
    global _worker_tokenizer
    from omegaconf import OmegaConf
    _worker_tokenizer = create_tokenizer(
        tokenizer_path, tokenizer_cfg=OmegaConf.create(tokenizer_cfg_dict)
    )


def _tokenize_shard(args: Tuple) -> Optional[str]:
    """
    Tokenize a shard (batch of texts) and save as a single ``.npz`` file.

    Each shard file stores a 1-D int32 numpy array that is the concatenation
    of all token-id sequences in the shard, plus a small header array that
    records the length of each individual sequence so we can reconstruct
    per-text boundaries later for verification.

    Shard file layout (saved via ``np.savez``):
        - ``tokens``:  int32 1-D array — concatenated token ids
        - ``lengths``: int32 1-D array — length of each text's token sequence

    Args:
        args: (shard_idx, texts_list, cache_dir)

    Returns:
        None on success, error message string on failure.
    """
    shard_idx, texts, cache_dir = args
    out_path = os.path.join(cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz")

    # Resume support: skip if already exists
    if os.path.exists(out_path):
        return None

    try:
        global _worker_tokenizer
        # Batch tokenize — leverages Rust parallelism inside the tokenizer
        encoded = _worker_tokenizer(
            texts, add_special_tokens=False, return_attention_mask=False,
            return_token_type_ids=False,
        )
        all_ids = encoded["input_ids"]  # list of list[int]

        lengths = np.array([len(ids) for ids in all_ids], dtype=np.int32)
        if int(lengths.astype(np.int64).sum()) > 0:
            tokens = np.concatenate(
                [np.array(ids, dtype=np.int32) for ids in all_ids]
            )
        else:
            tokens = np.array([], dtype=np.int32)

        # Atomic write: use a .tmp.npz extension so np.savez doesn't append
        # another .npz suffix, then rename to the final path.
        tmp_path = out_path + ".tmp.npz"
        np.savez(tmp_path, tokens=tokens, lengths=lengths)
        os.rename(tmp_path, out_path)
        return None
    except Exception as e:
        return f"[shard={shard_idx}] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Raw text loading — streaming-friendly
# ---------------------------------------------------------------------------

def _get_train_dataset(data_cfg, dataset_seed: int = 42):
    """
    Return the HuggingFace Dataset object (train split only, after val split).

    This does NOT load all texts into memory — it returns a lazy Dataset
    that can be iterated or indexed on demand.

    Args:
        data_cfg: Data configuration dict.
        dataset_seed: Seed for reproducible train/val split.

    Returns:
        (hf_dataset, num_texts) — the HF Dataset and its length.
    """
    data_path = data_cfg.get("data_path", os.path.join("data", "transmla_pretrain_6B_tokens"))
    data_format = data_cfg.get("data_format", "hf_dataset")
    abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)

    if data_format == "hf_dataset":
        dataset = load_dataset(abs_data_path, split="train")
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=dataset_seed)
        train_ds = split_dataset["train"]
    elif data_format == "hf_disk":
        dataset = load_from_disk(abs_data_path)
        split_dataset = dataset.train_test_split(test_size=0.0001, seed=dataset_seed)
        train_ds = split_dataset["train"]
    elif data_format == "jsonl":
        # For jsonl we must determine val indices first
        # Count lines without loading all content
        with open(abs_data_path, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f)
        val_size = max(1, int(n * 0.0005))
        rng = random.Random(dataset_seed)
        val_indices = set(rng.sample(range(n), val_size))
        # Build a HF dataset from the jsonl for consistent interface
        # But since jsonl can be huge, we use a generator approach
        train_ds = _JsonlTrainDataset(abs_data_path, val_indices, n - val_size)
        return train_ds, len(train_ds)
    else:
        raise ValueError(f"Unknown data_format: {data_format}")

    return train_ds, len(train_ds)


class _JsonlTrainDataset:
    """
    Lightweight wrapper for streaming a JSONL file while skipping val indices.
    Supports len() and iteration in batches without loading all into memory.
    """

    def __init__(self, path: str, val_indices: set, num_train: int):
        self.path = path
        self.val_indices = val_indices
        self._num_train = num_train

    def __len__(self) -> int:
        return self._num_train

    def iter_texts(self, batch_size: int):
        """
        Yield batches of text strings, skipping val indices.
        Each yield is a list of up to ``batch_size`` strings.
        """
        batch: List[str] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx in self.val_indices:
                    continue
                item = json.loads(line)
                batch.append(item["text"])
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def get_text(self, global_idx: int) -> str:
        """
        Get a single text by its train-set index (skipping val indices).
        This is O(n) — only used for verification spot checks.
        """
        train_count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx in self.val_indices:
                    continue
                if train_count == global_idx:
                    return json.loads(line)["text"]
                train_count += 1
        raise IndexError(f"global_idx {global_idx} out of range")


def _iter_texts_batched(train_ds, batch_size: int):
    """
    Yield (shard_idx, batch_of_texts) from the dataset in a streaming fashion.

    For HuggingFace Dataset objects, iterates using select() in slices to
    avoid loading the entire text column into memory at once.
    For _JsonlTrainDataset, uses its native iter_texts() method.

    Yields:
        (shard_idx: int, texts: List[str])
    """
    if isinstance(train_ds, _JsonlTrainDataset):
        for shard_idx, batch in enumerate(train_ds.iter_texts(batch_size)):
            yield shard_idx, batch
    else:
        # HuggingFace Dataset — iterate in slices
        num_texts = len(train_ds)
        shard_idx = 0
        for start in range(0, num_texts, batch_size):
            end = min(start + batch_size, num_texts)
            # select() returns a view, doesn't copy the whole dataset
            batch_texts = train_ds[start:end]["text"]
            yield shard_idx, batch_texts
            shard_idx += 1


def _get_text_by_index(train_ds, global_idx: int) -> str:
    """
    Retrieve a single text by index from the train dataset.
    Used for verification spot checks.
    """
    if isinstance(train_ds, _JsonlTrainDataset):
        return train_ds.get_text(global_idx)
    else:
        return train_ds[int(global_idx)]["text"]


# ---------------------------------------------------------------------------
# Preprocessing: batch tokenize + cache + verify
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Batch-tokenize every raw text and cache as shard ``.npz`` files.

    Uses streaming: texts are read from the data source in batches and
    dispatched to tokenization workers without loading the entire dataset
    into memory at once.

    Cache layout::

        {cache_dir}/
            manifest.json              # metadata
            shard_000000.npz           # tokens + lengths for texts [0, TEXTS_PER_SHARD)
            shard_000001.npz           # ...
            ...
            VERIFIED                   # written after successful verification

    Supports resume: already-existing shard files are skipped.

    Args:
        cfg: Hydra config (must have cfg.data).
        model_path: Absolute path to the model / tokenizer directory.
    """
    data_cfg = cfg.data
    _raw_cache = data_cfg.get("cache_dir", "cache/oldpretrain_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)
    num_workers = data_cfg.get("preprocess_workers", max(1, mp.cpu_count() // 2))
    texts_per_shard = data_cfg.get("texts_per_shard", TEXTS_PER_SHARD)

    os.makedirs(cache_dir, exist_ok=True)

    # ---- Open .txt log file for real-time output ----
    dataset_name = data_cfg.get("name", "oldpretrain")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{dataset_name}_preprocess.txt")
    _log_file = open(output_path, "w", encoding="utf-8")

    def _log(msg: str):
        """Write message to both logger and the .txt file (flushed immediately)."""
        logger.info(msg)
        _log_file.write(msg + "\n")
        _log_file.flush()

    sep = "=" * 80
    _log_file.write(f"{sep}\n")
    _log_file.write(f"  PREPROCESS — Dataset: {dataset_name}\n")
    _log_file.write(f"{sep}\n")
    _log_file.write(f"  Config file       : configs/data/pretrain/{dataset_name}.yaml\n")
    _log_file.write(f"  Model path        : {model_path}\n")
    _log_file.write(f"  Cache dir         : {cache_dir}\n")
    _log_file.write(f"  Workers           : {num_workers}\n")
    _log_file.write(f"  Texts per shard   : {texts_per_shard}\n")
    _log_file.write(f"{sep}\n\n")
    _log_file.flush()

    # Remove stale VERIFIED marker (we will re-verify after tokenization)
    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    if os.path.exists(verified_path):
        os.remove(verified_path)
        _log("[oldpretrain] Removed stale VERIFIED marker")

    # Remove stale token_stream.npy (will be rebuilt on next dataset load)
    mmap_path = os.path.join(cache_dir, "token_stream.npy")
    if os.path.exists(mmap_path):
        os.remove(mmap_path)
        _log("[oldpretrain] Removed stale token_stream.npy (will be rebuilt)")

    # ---- Get train dataset (lazy, not loaded into memory) ----
    _log("[oldpretrain] Opening data source (streaming mode) ...")
    dataset_seed = cfg.seed.dataset
    train_ds, num_texts = _get_train_dataset(data_cfg, dataset_seed=dataset_seed)
    num_shards = (num_texts + texts_per_shard - 1) // texts_per_shard
    _log(
        f"[oldpretrain] Data source ready: {num_texts} texts → "
        f"{num_shards} shards (texts_per_shard={texts_per_shard})"
    )

    # ---- Write / verify manifest ----
    manifest_path = os.path.join(cache_dir, MANIFEST_FILE)
    manifest = {
        "num_texts": num_texts,
        "num_shards": num_shards,
        "texts_per_shard": texts_per_shard,
        "model_path": str(model_path),
    }

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing = json.load(f)
        if existing.get("num_texts") != num_texts:
            _log_file.close()
            raise ValueError(
                f"Manifest mismatch: existing cache has {existing.get('num_texts')} texts "
                f"but current data has {num_texts}. Delete cache_dir '{cache_dir}' and re-run."
            )
        if existing.get("texts_per_shard") != texts_per_shard:
            _log_file.close()
            raise ValueError(
                f"Manifest mismatch: existing cache has texts_per_shard="
                f"{existing.get('texts_per_shard')} but config says {texts_per_shard}. "
                f"Delete cache_dir '{cache_dir}' and re-run."
            )
        _log(f"[oldpretrain] Manifest verified: {num_texts} texts, {num_shards} shards")
    else:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        _log(f"[oldpretrain] Manifest written: {manifest_path}")

    # ---- Count already-cached shards ----
    already_done = sum(
        1 for si in range(num_shards)
        if os.path.exists(os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz"))
    )
    remaining = num_shards - already_done
    _log(
        f"[oldpretrain] Cache status: {already_done}/{num_shards} shards done, "
        f"{remaining} remaining"
    )

    if remaining > 0:
        # ---- Stream texts and build work items for missing shards only ----
        # We stream through the data source in batches (texts_per_shard at a time).
        # Only batches whose shard file doesn't exist yet are dispatched.
        # This avoids loading all texts into memory simultaneously.
        _log("[oldpretrain] Streaming texts and dispatching to workers ...")

        t0 = time.time()
        errors: List[str] = []
        total_dispatched = 0

        if num_workers <= 1:
            # Single-worker mode: process sequentially in main process
            from omegaconf import OmegaConf as _OC
            _init_batch_worker(model_path, _OC.to_container(cfg.tokenizer, resolve=True))
            for shard_idx, batch in _iter_texts_batched(train_ds, texts_per_shard):
                shard_path = os.path.join(cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz")
                if os.path.exists(shard_path):
                    continue
                err = _tokenize_shard((shard_idx, batch, cache_dir))
                if err is not None:
                    errors.append(err)
                total_dispatched += 1
                elapsed = time.time() - t0
                rate = total_dispatched / elapsed if elapsed > 0 else 0
                _log(
                    f"[oldpretrain]   {total_dispatched}/{remaining} "
                    f"[{rate:.1f} shards/s, {elapsed:.1f}s elapsed]"
                )
        else:
            # Multi-worker mode: stream batches into a pool
            # We use a bounded submission approach to avoid accumulating all
            # pending work items in memory. We submit in chunks of
            # pool_buffer_size and wait for results before submitting more.
            pool_buffer_size = num_workers * 4  # keep workers busy

            with mp.Pool(
                processes=num_workers,
                initializer=_init_batch_worker,
                initargs=(model_path, OmegaConf.to_container(cfg.tokenizer, resolve=True)),
            ) as pool:
                pending_results = []
                done_count = 0

                for shard_idx, batch in _iter_texts_batched(train_ds, texts_per_shard):
                    shard_path = os.path.join(cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz")
                    if os.path.exists(shard_path):
                        continue

                    # Submit work
                    ar = pool.apply_async(_tokenize_shard, ((shard_idx, batch, cache_dir),))
                    pending_results.append(ar)
                    total_dispatched += 1

                    # Drain completed results when buffer is full
                    if len(pending_results) >= pool_buffer_size:
                        for ar in pending_results:
                            result = ar.get()
                            if result is not None:
                                errors.append(result)
                            done_count += 1
                            elapsed = time.time() - t0
                            rate = done_count / elapsed if elapsed > 0 else 0
                            _log(
                                f"[oldpretrain]   {done_count}/{remaining} "
                                f"[{rate:.1f} shards/s, {elapsed:.1f}s elapsed]"
                            )
                        pending_results = []

                # Drain remaining results
                for ar in pending_results:
                    result = ar.get()
                    if result is not None:
                        errors.append(result)
                    done_count += 1
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0
                    _log(
                        f"[oldpretrain]   {done_count}/{remaining} "
                        f"[{rate:.1f} shards/s, {elapsed:.1f}s elapsed]"
                    )

        elapsed = time.time() - t0
        _log(
            f"[oldpretrain] Tokenization complete: {total_dispatched} shards in {elapsed:.1f}s"
        )

        if errors:
            for e in errors[:10]:
                logger.error(f"[oldpretrain] Tokenization error: {e}")
                _log_file.write(f"[ERROR] {e}\n")
            if len(errors) > 10:
                logger.error(f"[oldpretrain] ... and {len(errors) - 10} more errors")
                _log_file.write(f"[ERROR] ... and {len(errors) - 10} more errors\n")
            _log_file.flush()
            _log_file.close()
            raise RuntimeError(
                f"Tokenization failed for {len(errors)} shards. See errors above."
            )

    # Free dataset reference
    del train_ds

    # ---- Verification ----
    _log("[oldpretrain] Starting verification ...")
    _verify_and_mark(cfg, cache_dir, model_path, num_texts, num_shards, texts_per_shard)
    _log("[oldpretrain] Preprocessing complete. Cache is verified and ready.")

    # ---- Final summary ----
    _log_file.write(f"\n{sep}\n")
    _log_file.write(f"  FINAL SUMMARY\n")
    _log_file.write(f"{sep}\n")
    _log_file.write(f"  Total texts       : {num_texts}\n")
    _log_file.write(f"  Shards            : {num_shards}\n")
    _log_file.write(f"  Status            : VERIFIED ✓\n")
    _log_file.write(f"{sep}\n")
    _log_file.flush()
    _log_file.close()

    logger.info(f"[oldpretrain] Preprocess log written to: {output_path}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_and_mark(
    cfg,
    cache_dir: str,
    model_path: str,
    num_texts: int,
    num_shards: int,
    texts_per_shard: int,
):
    """
    Thorough verification of the tokenization cache, then write VERIFIED marker.

    Checks performed:
        1. All shard files exist (no missing shards).
        2. No stale ``.tmp`` files.
        3. Total text count across all shards matches ``num_texts``.
        4. Each shard's ``lengths`` array sums to the length of its ``tokens`` array.
        5. Random sample of texts are re-tokenized and compared token-by-token
           against the cached version.

    On success, writes ``VERIFIED`` file containing a JSON summary.

    Raises:
        RuntimeError: On any verification failure.
    """
    logger.info("[oldpretrain] Starting verification ...")
    t0 = time.time()

    # --- Check 1: all shard files exist ---
    missing_shards = []
    for si in range(num_shards):
        p = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
        if not os.path.exists(p):
            missing_shards.append(si)
    if missing_shards:
        raise RuntimeError(
            f"Verification failed: {len(missing_shards)} shard files missing. "
            f"First missing: {missing_shards[:20]}. Re-run preprocess."
        )

    # --- Check 2: no stale .tmp files ---
    tmp_files = [f for f in os.listdir(cache_dir) if ".tmp" in f]
    if tmp_files:
        logger.warning(
            f"[oldpretrain] Found {len(tmp_files)} stale .tmp files — removing them."
        )
        for tf in tmp_files:
            os.remove(os.path.join(cache_dir, tf))

    # --- Check 3 & 4: text count and internal consistency ---
    total_text_count = 0
    total_token_count = 0
    shard_token_counts: List[int] = []

    for si in range(num_shards):
        p = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
        try:
            data = np.load(p)
            tokens = data["tokens"]
            lengths = data["lengths"]
        except Exception as e:
            raise RuntimeError(
                f"Verification failed: cannot load shard {si} at '{p}': {e}"
            )

        # Internal consistency: sum of lengths == len(tokens)
        expected_total = int(lengths.astype(np.int64).sum())
        if expected_total != len(tokens):
            raise RuntimeError(
                f"Verification failed: shard {si} has lengths.sum()={expected_total} "
                f"but tokens has {len(tokens)} elements."
            )

        total_text_count += len(lengths)
        total_token_count += len(tokens)
        shard_token_counts.append(len(tokens))

    if total_text_count != num_texts:
        raise RuntimeError(
            f"Verification failed: total text count across shards is {total_text_count} "
            f"but expected {num_texts}."
        )

    logger.info(
        f"[oldpretrain] Shard structure OK: {num_shards} shards, "
        f"{total_text_count} texts, {total_token_count:,} tokens total"
    )

    # --- Check 5: random re-tokenization spot check ---
    sample_size = min(VERIFY_SAMPLE_SIZE, num_texts)
    if sample_size > 0:
        logger.info(
            f"[oldpretrain] Re-tokenizing {sample_size} random texts for spot check ..."
        )
        tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer)

        # Get train dataset in streaming mode for spot check
        data_cfg = cfg.data
        dataset_seed = cfg.seed.dataset
        train_ds, ds_len = _get_train_dataset(data_cfg, dataset_seed=dataset_seed)
        assert ds_len == num_texts, (
            f"Raw text count changed: {ds_len} vs {num_texts}"
        )

        rng = random.Random(12345)
        sample_indices = sorted(rng.sample(range(num_texts), sample_size))

        # For each sampled text, find which shard it belongs to and compare
        mismatches = []
        for global_idx in sample_indices:
            shard_idx = global_idx // texts_per_shard
            local_idx = global_idx % texts_per_shard

            # Load shard and extract the specific text's tokens
            shard_path = os.path.join(
                cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz"
            )
            data = np.load(shard_path)
            lengths = data["lengths"]
            tokens = data["tokens"]

            # Compute offset for local_idx
            offset = int(lengths[:local_idx].astype(np.int64).sum())
            length = int(lengths[local_idx])
            cached_ids = tokens[offset : offset + length].tolist()

            # Re-tokenize — fetch text on demand (no full load)
            text = str(_get_text_by_index(train_ds, global_idx))
            fresh_ids = tokenizer.encode(text, add_special_tokens=False)

            if cached_ids != fresh_ids:
                mismatches.append(
                    f"  idx={global_idx}: cached {len(cached_ids)} tokens vs "
                    f"fresh {len(fresh_ids)} tokens"
                )

        del train_ds  # free reference

        if mismatches:
            detail = "\n".join(mismatches[:10])
            raise RuntimeError(
                f"Verification failed: {len(mismatches)}/{sample_size} spot-check "
                f"texts have mismatched tokens:\n{detail}"
            )

        logger.info(
            f"[oldpretrain] Spot check passed: {sample_size} texts re-tokenized and matched"
        )

    # --- Write VERIFIED marker ---
    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    verified_info = {
        "num_texts": num_texts,
        "num_shards": num_shards,
        "texts_per_shard": texts_per_shard,
        "total_tokens": total_token_count,
        "spot_check_size": sample_size,
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
    }
    with open(verified_path, "w") as f:
        json.dump(verified_info, f, indent=2)

    elapsed = time.time() - t0
    logger.info(
        f"[oldpretrain] Verification complete in {elapsed:.1f}s. "
        f"VERIFIED marker written to '{verified_path}'"
    )


def _check_verified_marker(cache_dir: str):
    """
    Check that the VERIFIED marker exists in ``cache_dir``.

    This is called at dataset load time (not preprocess time).
    If the marker is missing, it means preprocess was never run or
    verification failed.

    Raises:
        RuntimeError: If VERIFIED marker is missing.
    """
    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    if not os.path.exists(verified_path):
        raise RuntimeError(
            f"VERIFIED marker not found in '{cache_dir}'. "
            f"You must run preprocess (with successful verification) before "
            f"loading the dataset. Run:\n"
            f"  python mydatasets/oldpretrain.py --preprocess --model_path <path>"
        )

    with open(verified_path, "r") as f:
        info = json.load(f)

    logger.info(
        f"[oldpretrain] Cache verified: {info['num_texts']} texts, "
        f"{info['num_shards']} shards, {info['total_tokens']:,} tokens "
        f"(verified at {info['verified_at']})"
    )
    return info


# ---------------------------------------------------------------------------
# OldPretrainDataset
# ---------------------------------------------------------------------------

class OldPretrainDataset(BaseDataset):
    """
    Chunked pretrain dataset.

    Loads pre-tokenized shard ``.npz`` files from cache, concatenates all
    token arrays with ``<|endoftext|>`` separators into a memory-mapped
    ``.npy`` file, and chunks the resulting stream into fixed-length segments.

    Memory-efficient: uses a memory-mapped numpy file on disk. The OS pages
    data in/out on demand, so RSS stays low even for datasets with billions
    of tokens. The mmap file is built once (on first load after preprocessing)
    and reused on subsequent runs.

    Each sample (chunk) is a 1-D ``torch.LongTensor`` of length
    ``chunk_length``.
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        chunk_length: int,
    ):
        """
        Args:
            model_path: Path to the pretrained model / tokenizer directory.
            cache_dir: Directory containing the tokenized shard cache files
                       and ``manifest.json``.
            chunk_length: Length of each chunk (== context_seq_length == conv_seq_length).
        """
        super().__init__(model_path)

        self.chunk_length = chunk_length
        self.cache_dir = cache_dir

        # ---- Check VERIFIED marker (fast fail) ----
        verified_info = _check_verified_marker(cache_dir)

        # ---- Load manifest ----
        manifest_path = os.path.join(cache_dir, MANIFEST_FILE)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        num_texts = manifest["num_texts"]
        num_shards = manifest["num_shards"]

        # ---- Build or reuse memory-mapped token stream ----
        # The mmap file concatenates all shard tokens with <|endoftext|>
        # separators. Building it requires scanning all shards (slow), so
        # we skip the scan entirely when the mmap already exists.
        #
        # Use a file lock so that only ONE process per node performs the
        # expensive scan + build. Other processes wait for the lock, then
        # directly load the finished mmap file.
        eos_id = self.special_tokens["<|endoftext|>"]
        mmap_path = os.path.join(cache_dir, "token_stream.npy")
        lock_path = os.path.join(cache_dir, "token_stream.lock")
        verified_path = os.path.join(cache_dir, VERIFIED_FILE)

        t0 = time.time()

        def _mmap_is_valid():
            """Check if the mmap file exists and is valid."""
            if not os.path.exists(mmap_path):
                return False
            try:
                existing_mmap = np.load(mmap_path, mmap_mode="r")
                mmap_mtime = os.path.getmtime(mmap_path)
                verified_mtime = os.path.getmtime(verified_path)
                valid = (existing_mmap.dtype == np.int32
                         and len(existing_mmap.shape) == 1
                         and existing_mmap.shape[0] > 0
                         and mmap_mtime >= verified_mtime)
                del existing_mmap
                return valid
            except Exception:
                return False

        # First check without lock — fast path for when mmap already exists
        if not _mmap_is_valid():
            with open(lock_path, "w") as lock_file:
                # Acquire exclusive lock — only one process builds at a time;
                # others block here until the lock is released.
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    # Re-check after acquiring lock — another process (possibly
                    # on another node) may have already built the mmap while we
                    # were waiting for the lock.
                    if not _mmap_is_valid():
                        # ---- Scan shards to compute total token count ----
                        logger.info(
                            f"[oldpretrain] Scanning {num_shards} shards to compute total token count ..."
                        )

                        shard_infos: List[Tuple[str, int, int]] = []
                        total_stream_len = 0
                        for si in range(num_shards):
                            shard_path = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
                            data = np.load(shard_path)
                            lengths = data["lengths"]
                            n_tokens = int(lengths.astype(np.int64).sum())
                            n_nonempty = int((lengths > 0).sum())
                            stream_contribution = n_tokens + n_nonempty
                            shard_infos.append((shard_path, n_tokens, n_nonempty))
                            total_stream_len += stream_contribution
                            del data, lengths

                        elapsed_scan = time.time() - t0
                        logger.info(
                            f"[oldpretrain] Scan complete in {elapsed_scan:.1f}s: "
                            f"{total_stream_len:,} tokens (including separators)"
                        )

                        # ---- Build memory-mapped token stream ----
                        logger.info(
                            f"[oldpretrain] Building memory-mapped token stream: "
                            f"{total_stream_len:,} tokens × 4 bytes = "
                            f"{total_stream_len * 4 / (1024**3):.2f} GB → {mmap_path}"
                        )

                        # Use hostname+pid to create a unique tmp file per process,
                        # avoiding cross-node conflicts on shared filesystems.
                        import socket
                        hostname = socket.gethostname()
                        mmap_tmp_path = f"{mmap_path}.tmp.{hostname}.{os.getpid()}"
                        fp = np.lib.format.open_memmap(
                            mmap_tmp_path, mode="w+", dtype=np.int32, shape=(total_stream_len,)
                        )

                        write_pos = 0
                        for si, (shard_path, n_tokens, n_nonempty) in enumerate(shard_infos):
                            if n_tokens == 0:
                                continue

                            data = np.load(shard_path)
                            tokens = data["tokens"]
                            lengths = data["lengths"]

                            read_pos = 0
                            for length in lengths:
                                length = int(length)
                                if length == 0:
                                    continue
                                fp[write_pos : write_pos + length] = tokens[read_pos : read_pos + length]
                                write_pos += length
                                fp[write_pos] = eos_id
                                write_pos += 1
                                read_pos += length

                            del data, tokens, lengths

                            if (si + 1) % 100 == 0:
                                fp.flush()

                        assert write_pos == total_stream_len, (
                            f"Write position {write_pos} != expected {total_stream_len}"
                        )

                        fp.flush()
                        del fp

                        # Atomic rename
                        os.rename(mmap_tmp_path, mmap_path)

                        elapsed_build = time.time() - t0
                        logger.info(
                            f"[oldpretrain] Token stream built in {elapsed_build:.1f}s: "
                            f"{total_stream_len:,} tokens → {mmap_path}"
                        )
                    else:
                        logger.info(
                            f"[oldpretrain] Token stream already built by another process, skipping."
                        )

                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        else:
            logger.info(
                f"[oldpretrain] Token stream already exists, skipping build."
            )

        # ---- Memory-map the token stream (read-only) ----
        token_stream = np.load(mmap_path, mmap_mode="r")
        total_stream_len = token_stream.shape[0]

        elapsed_total = time.time() - t0
        logger.info(
            f"[oldpretrain] Loaded memory-mapped token stream: "
            f"{mmap_path} ({total_stream_len:,} tokens, {elapsed_total:.1f}s)"
        )

        # ---- Chunk into fixed-length segments ----
        num_chunks = total_stream_len // chunk_length
        if num_chunks == 0:
            raise ValueError(
                f"Total tokens ({total_stream_len}) < chunk_length ({chunk_length}). "
                f"Not enough data to form even one chunk."
            )

        # Trim the tail that doesn't fill a complete chunk
        used_tokens = num_chunks * chunk_length
        # Reshape as a view over the memory-mapped array — no copy, no extra RAM
        self.chunks = token_stream[:used_tokens].reshape(num_chunks, chunk_length)
        # Keep a reference to prevent GC of the underlying mmap
        self._token_stream_mmap = token_stream

        # Deterministic distill offset: fixed cyclic shift ensures a bijection
        # (every chunk is used as a distill target exactly once per epoch).
        self._distill_offset = min(self.DISTILL_MIN_DISTANCE, num_chunks // 2)

        logger.info(
            f"[oldpretrain] Created {num_chunks:,} chunks of length {chunk_length} "
            f"({used_tokens:,} tokens used, "
            f"{total_stream_len - used_tokens} tokens trimmed) [memory-mapped] "
            f"distill_offset={self._distill_offset}"
        )

    # Minimum distance (in chunks) between the primary chunk and the
    # distillation chunk to avoid data correlation.
    DISTILL_MIN_DISTANCE = 1000

    def __len__(self) -> int:
        return self.chunks.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Copy from mmap into a writable array (torch requires writable memory)
        chunk = torch.from_numpy(np.array(self.chunks[idx], dtype=np.int64))

        # Deterministic mapping: fixed cyclic shift ensures bijection
        # (every chunk is distilled exactly once per epoch)
        num_chunks = self.chunks.shape[0]
        distill_idx = (idx + self._distill_offset) % num_chunks
        distill_chunk = torch.from_numpy(np.array(self.chunks[distill_idx], dtype=np.int64))

        return {"chunk": chunk, "distill_chunk": distill_chunk}


# ---------------------------------------------------------------------------
# OldPretrainCollator
# ---------------------------------------------------------------------------

class OldPretrainCollator(BaseCollator):
    """
    Collator for OldPretrainDataset.

    Enforces ``context_max_length == conversation_max_length``.

    For each chunk of length L:
        context_ids      = [chunk_tokens | mem_placeholders × num_mem_token]
                           shape: (L + num_mem_token,)
        conversation_ids = chunk_tokens
                           shape: (L,)
        labels           = chunk_tokens  (all positions contribute to loss)
                           shape: (L,)
        context_lengths  = L  (every chunk is full-length)

    Output batch:
        context_ids:      (B, L + num_mem_token)
        conversation_ids: (B, L)
        labels:           (B, L)
        context_lengths:  (B,)
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

        if context_max_length != conversation_max_length:
            raise ValueError(
                f"context_max_length ({context_max_length}) must equal "
                f"conversation_max_length ({conversation_max_length}) "
                f"for oldpretrain dataset."
            )

        self.chunk_length = context_max_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        batch_size = len(samples)
        L = self.chunk_length
        num_mem = self.num_mem_token
        context_total_len = L + num_mem

        # Pre-allocate tensors
        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.empty((batch_size, L), dtype=torch.long)
        labels = torch.empty((batch_size, L), dtype=torch.long)
        context_lengths = torch.full((batch_size,), L, dtype=torch.long)

        # Distillation tensors
        distill_conversation_ids = torch.empty((batch_size, L), dtype=torch.long)
        distill_labels = torch.empty((batch_size, L), dtype=torch.long)

        for i, s in enumerate(samples):
            chunk = s["chunk"]  # (L,)
            # context_ids: [chunk | mem_placeholders]
            # mem_placeholders are left as pad_token_id (already filled)
            context_ids[i, :L] = chunk
            conversation_ids[i] = chunk
            labels[i] = chunk

            # Distillation: use a distant chunk
            distill_chunk = s["distill_chunk"]  # (L,)
            distill_conversation_ids[i] = distill_chunk
            distill_labels[i] = distill_chunk

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
    Create an OldPretrainDataset and OldPretrainCollator from the data configuration.

    Args:
        cfg: Full Hydra DictConfig (has cfg.data, cfg.training, etc.)
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (OldPretrainDataset, OldPretrainCollator)
    """
    data_cfg = cfg.data

    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length

    if context_seq_len != conv_seq_len:
        raise ValueError(
            f"context_seq_length ({context_seq_len}) must equal "
            f"conv_seq_length ({conv_seq_len}) for oldpretrain dataset."
        )

    _raw_cache = data_cfg.get("cache_dir", "cache/oldpretrain_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)

    logger.info(
        f"[oldpretrain] Creating dataset: chunk_length={context_seq_len}, "
        f"cache_dir='{cache_dir}', num_mem_token={num_mem_token}"
    )

    # ---- Build dataset ----
    train_ds = OldPretrainDataset(
        model_path=model_path,
        cache_dir=cache_dir,
        chunk_length=context_seq_len,
    )

    # ---- Build collator ----
    collator = OldPretrainCollator(
        model_path=model_path,
        context_max_length=context_seq_len,
        conversation_max_length=conv_seq_len,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
    )

    return train_ds, collator


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
    from omegaconf import OmegaConf

    if "model" not in cfg:
        cfg = OmegaConf.merge(cfg, {"model": {"path": model_path}})

    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    num_mem_token = 10

    dataset, collator = create_dataset_and_collator(
        cfg, model_path, pad_token_id, num_mem_token,
    )

    data_cfg = cfg.data
    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer)

    metadata = {
        "context_seq_len": data_cfg.context_seq_length,
        "conv_seq_len": data_cfg.conv_seq_length,
        "cache_dir": data_cfg.get("cache_dir", "N/A"),
        "num_chunks": len(dataset),
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("name", "oldpretrain"),
        metadata=metadata,
        num_samples=5,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/oldpretrain.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="oldpretrain dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Parallel tokenize and cache")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/oldpretrain.yaml",
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