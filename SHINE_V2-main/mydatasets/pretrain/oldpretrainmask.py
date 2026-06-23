#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Old Pretrain Mask Dataset Module for SHINE_V2

Preprocessing:
    1. Reads raw texts from the data source in batches.
    2. For each text, formats as chat message:
       [{"role": "user", "content": "<CTX>"}, {"role": "assistant", "content": text}]
    3. Tokenizes using apply_chat_template with preserve_thinking=True
       (preserve_thinking mode).
    4. Shuffles all tokenized sequences at text level (deterministic seed).
    5. Saves shuffled token sequences as .npz shard files.

Dataset:
    Builds a memory-mapped .npy file from cached shards (concatenated with
    <|endoftext|> separators), then chunks into fixed-length segments.
    Designed for sequential (non-shuffled) training with contiguous DP sharding.

Collator:
    - conversation_ids = chunk (unmasked, the prediction target)
    - context_ids = chunk with masking ONLY on text content tokens (replaced by <MASK>)
    - labels = only text content positions and trailing <|im_end|> are kept;
      all other positions are set to -100
    Text content is defined as tokens AFTER the pattern </think>\n\n and BEFORE
    the next <|im_end|>. Only these positions are eligible for masking and
    contribute to the training loss.
"""

from __future__ import annotations

import os
import json
import random
import logging
import time
import fcntl
import multiprocessing as mp
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from datasets import load_dataset, load_from_disk
from mydatasets.base import BaseDataset, BaseCollator
from utils.mytokenizer import create_tokenizer
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

SHARD_PREFIX = "shard_"
MANIFEST_FILE = "manifest.json"
VERIFIED_FILE = "VERIFIED"
TEXTS_PER_SHARD = 1000


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
    Tokenize a shard of texts using chat template with preserve_thinking mode.

    Each text is formatted as:
        [{"role": "user", "content": "<CTX>"}, {"role": "assistant", "content": text}]
    Then tokenized with apply_chat_template(preserve_thinking=True).
    """
    shard_idx, texts, cache_dir = args
    out_path = os.path.join(cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz")

    if os.path.exists(out_path):
        return None

    try:
        global _worker_tokenizer

        all_ids = []
        for text in texts:
            messages = [
                {"role": "user", "content": "<CTX>"},
                {"role": "assistant", "content": str(text)},
            ]
            result = _worker_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                preserve_thinking=True,
            )
            # apply_chat_template may return a list of ints, a dict,
            # or a BatchEncoding (inherits UserDict, not dict).
            if isinstance(result, list):
                token_ids = result
            elif hasattr(result, "input_ids"):
                token_ids = result.input_ids
            elif hasattr(result, "__getitem__") and "input_ids" in result:
                token_ids = result["input_ids"]
            else:
                token_ids = list(result)
            all_ids.append(token_ids)

        lengths = np.array([len(ids) for ids in all_ids], dtype=np.int32)
        if int(lengths.astype(np.int64).sum()) > 0:
            tokens = np.concatenate(
                [np.array(ids, dtype=np.int32) for ids in all_ids]
            )
        else:
            tokens = np.array([], dtype=np.int32)

        tmp_path = out_path + ".tmp.npz"
        np.savez(tmp_path, tokens=tokens, lengths=lengths)
        os.rename(tmp_path, out_path)
        return None
    except Exception as e:
        return f"[shard={shard_idx}] {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Raw text loading
# ---------------------------------------------------------------------------

def _get_train_dataset(data_cfg, dataset_seed: int = 42):
    """Return the train dataset."""
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
        with open(abs_data_path, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f)
        val_size = max(1, int(n * 0.0005))
        rng = random.Random(dataset_seed)
        val_indices = set(rng.sample(range(n), val_size))
        train_ds = _JsonlTrainDataset(abs_data_path, val_indices, n - val_size)
        return train_ds, len(train_ds)
    else:
        raise ValueError(f"Unknown data_format: {data_format}")

    return train_ds, len(train_ds)


class _JsonlTrainDataset:
    """Lightweight wrapper for streaming a JSONL file while skipping val indices."""

    def __init__(self, path: str, val_indices: set, num_train: int):
        self.path = path
        self.val_indices = val_indices
        self._num_train = num_train

    def __len__(self) -> int:
        return self._num_train

    def iter_texts(self, batch_size: int):
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


def _iter_texts_batched(train_ds, batch_size: int):
    """Yield (shard_idx, batch_of_texts) from the dataset."""
    if isinstance(train_ds, _JsonlTrainDataset):
        for shard_idx, batch in enumerate(train_ds.iter_texts(batch_size)):
            yield shard_idx, batch
    else:
        num_texts = len(train_ds)
        shard_idx = 0
        for start in range(0, num_texts, batch_size):
            end = min(start + batch_size, num_texts)
            batch_texts = train_ds[start:end]["text"]
            yield shard_idx, batch_texts
            shard_idx += 1


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Tokenize all texts with chat template, shuffle at text level, and cache.

    Steps:
        1. Tokenize into temporary unshuffled shards.
        2. Load all sequences, shuffle with deterministic seed (text-level).
        3. Re-shard shuffled sequences into final cache dir.
        4. Verify correctness.
    """
    data_cfg = cfg.data
    _raw_cache = data_cfg.get("cache_dir", "cache/oldpretrainmask_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)
    num_workers = data_cfg.get("preprocess_workers", max(1, mp.cpu_count() // 2))
    texts_per_shard = data_cfg.get("texts_per_shard", TEXTS_PER_SHARD)
    shuffle_seed = data_cfg.get("preprocess_shuffle_seed", 42)

    os.makedirs(cache_dir, exist_ok=True)

    # Log file
    dataset_name = data_cfg.get("name", "oldpretrainmask")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{dataset_name}_preprocess.txt")
    _log_file = open(output_path, "w", encoding="utf-8")

    def _log(msg: str):
        logger.info(msg)
        _log_file.write(msg + "\n")
        _log_file.flush()

    sep = "=" * 80
    _log(sep)
    _log(f"  PREPROCESS — Dataset: {dataset_name}")
    _log(sep)
    _log(f"  Model path        : {model_path}")
    _log(f"  Cache dir         : {cache_dir}")
    _log(f"  Workers           : {num_workers}")
    _log(f"  Texts per shard   : {texts_per_shard}")
    _log(f"  Shuffle seed      : {shuffle_seed}")
    _log(sep)

    # Remove stale markers
    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    if os.path.exists(verified_path):
        os.remove(verified_path)
    mmap_path = os.path.join(cache_dir, "token_stream.npy")
    if os.path.exists(mmap_path):
        os.remove(mmap_path)

    # Phase 1: Tokenize into temporary unshuffled shards
    tmp_cache_dir = os.path.join(cache_dir, "_tmp_unshuffled")
    os.makedirs(tmp_cache_dir, exist_ok=True)

    _log("[oldpretrainmask] Opening data source ...")
    dataset_seed = cfg.seed.dataset
    train_ds, num_texts = _get_train_dataset(data_cfg, dataset_seed=dataset_seed)
    num_shards_raw = (num_texts + texts_per_shard - 1) // texts_per_shard
    _log(f"[oldpretrainmask] {num_texts} texts -> {num_shards_raw} raw shards")

    already_done = sum(
        1 for si in range(num_shards_raw)
        if os.path.exists(os.path.join(tmp_cache_dir, f"{SHARD_PREFIX}{si:06d}.npz"))
    )
    remaining = num_shards_raw - already_done
    _log(f"[oldpretrainmask] Tokenization: {already_done}/{num_shards_raw} done, {remaining} remaining")

    if remaining > 0:
        t0 = time.time()
        errors: List[str] = []
        total_dispatched = 0

        if num_workers <= 1:
            _init_batch_worker(model_path)
            for shard_idx, batch in _iter_texts_batched(train_ds, texts_per_shard):
                shard_path = os.path.join(tmp_cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz")
                if os.path.exists(shard_path):
                    continue
                err = _tokenize_shard((shard_idx, batch, tmp_cache_dir))
                if err is not None:
                    errors.append(err)
                total_dispatched += 1
                if total_dispatched % 10 == 0:
                    elapsed = time.time() - t0
                    _log(f"[oldpretrainmask]   {total_dispatched}/{remaining} [{total_dispatched/max(elapsed,1e-9):.1f} shards/s]")
        else:
            pool_buffer_size = num_workers * 4
            with mp.Pool(processes=num_workers, initializer=_init_batch_worker, initargs=(model_path, OmegaConf.to_container(cfg.tokenizer, resolve=True))) as pool:
                pending_results = []
                done_count = 0
                for shard_idx, batch in _iter_texts_batched(train_ds, texts_per_shard):
                    shard_path = os.path.join(tmp_cache_dir, f"{SHARD_PREFIX}{shard_idx:06d}.npz")
                    if os.path.exists(shard_path):
                        continue
                    ar = pool.apply_async(_tokenize_shard, ((shard_idx, batch, tmp_cache_dir),))
                    pending_results.append(ar)
                    total_dispatched += 1
                    if len(pending_results) >= pool_buffer_size:
                        for ar in pending_results:
                            result = ar.get()
                            if result is not None:
                                errors.append(result)
                            done_count += 1
                        pending_results = []
                        elapsed = time.time() - t0
                        _log(f"[oldpretrainmask]   {done_count}/{remaining} [{done_count/max(elapsed,1e-9):.1f} shards/s]")
                for ar in pending_results:
                    result = ar.get()
                    if result is not None:
                        errors.append(result)
                    done_count += 1

        elapsed = time.time() - t0
        _log(f"[oldpretrainmask] Tokenization complete: {total_dispatched} shards in {elapsed:.1f}s")
        if errors:
            for e in errors[:10]:
                _log(f"[ERROR] {e}")
            _log_file.close()
            raise RuntimeError(f"Tokenization failed for {len(errors)} shards.")

    del train_ds

    # Phase 2: Load all sequences, shuffle at text level, re-shard
    _log("[oldpretrainmask] Phase 2: Loading sequences for text-level shuffling ...")
    t0 = time.time()

    all_sequences: List[np.ndarray] = []
    for si in range(num_shards_raw):
        shard_path = os.path.join(tmp_cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
        data = np.load(shard_path)
        tokens = data["tokens"]
        lengths = data["lengths"]
        offset = 0
        for length in lengths:
            length = int(length)
            if length > 0:
                all_sequences.append(tokens[offset:offset + length].copy())
            offset += length
        del data, tokens, lengths

    total_sequences = len(all_sequences)
    _log(f"[oldpretrainmask] Loaded {total_sequences} text sequences")

    # Shuffle at text level (each sequence = one tokenized text)
    _log(f"[oldpretrainmask] Shuffling {total_sequences} text sequences (seed={shuffle_seed}) ...")
    rng = random.Random(shuffle_seed)
    indices = list(range(total_sequences))
    rng.shuffle(indices)
    all_sequences = [all_sequences[i] for i in indices]

    # Re-shard
    num_shards_final = (total_sequences + texts_per_shard - 1) // texts_per_shard
    _log(f"[oldpretrainmask] Writing {num_shards_final} shuffled shards ...")

    for si in range(num_shards_final):
        start = si * texts_per_shard
        end = min(start + texts_per_shard, total_sequences)
        shard_seqs = all_sequences[start:end]
        lengths = np.array([len(seq) for seq in shard_seqs], dtype=np.int32)
        if int(lengths.astype(np.int64).sum()) > 0:
            tokens = np.concatenate(shard_seqs)
        else:
            tokens = np.array([], dtype=np.int32)
        out_path = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
        tmp_path = out_path + ".tmp.npz"
        np.savez(tmp_path, tokens=tokens, lengths=lengths)
        os.rename(tmp_path, out_path)

    elapsed = time.time() - t0
    _log(f"[oldpretrainmask] Phase 2 complete in {elapsed:.1f}s")
    del all_sequences

    # Write manifest
    manifest_path = os.path.join(cache_dir, MANIFEST_FILE)
    manifest = {
        "num_texts": total_sequences,
        "num_shards": num_shards_final,
        "texts_per_shard": texts_per_shard,
        "model_path": str(model_path),
        "shuffle_seed": shuffle_seed,
        "original_num_texts": num_texts,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Verification
    _log("[oldpretrainmask] Verifying ...")
    _verify_and_mark(cache_dir, model_path, total_sequences, num_shards_final, texts_per_shard)
    _log("[oldpretrainmask] Preprocessing complete.")

    _log(f"\n{sep}")
    _log(f"  FINAL: {num_texts} texts -> {total_sequences} sequences, {num_shards_final} shards, seed={shuffle_seed}")
    _log(sep)
    _log_file.close()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_and_mark(cache_dir, model_path, num_texts, num_shards, texts_per_shard):
    """Verify cache structure and write VERIFIED marker."""
    missing = [si for si in range(num_shards)
               if not os.path.exists(os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz"))]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} shards")

    for f in os.listdir(cache_dir):
        if ".tmp" in f:
            os.remove(os.path.join(cache_dir, f))

    total_text_count = 0
    total_token_count = 0
    for si in range(num_shards):
        p = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
        data = np.load(p)
        tokens = data["tokens"]
        lengths = data["lengths"]
        expected = int(lengths.astype(np.int64).sum())
        if expected != len(tokens):
            raise RuntimeError(f"Shard {si}: lengths.sum()={expected} != len(tokens)={len(tokens)}")
        total_text_count += len(lengths)
        total_token_count += len(tokens)

    if total_text_count != num_texts:
        raise RuntimeError(f"Total count {total_text_count} != expected {num_texts}")

    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    with open(verified_path, "w") as f:
        json.dump({
            "num_texts": num_texts,
            "num_shards": num_shards,
            "texts_per_shard": texts_per_shard,
            "total_tokens": total_token_count,
            "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
        }, f, indent=2)
    logger.info(f"[oldpretrainmask] Verified: {num_shards} shards, {total_token_count:,} tokens")


def _check_verified_marker(cache_dir: str):
    """Check VERIFIED marker exists."""
    verified_path = os.path.join(cache_dir, VERIFIED_FILE)
    if not os.path.exists(verified_path):
        raise RuntimeError(
            f"VERIFIED marker not found in '{cache_dir}'. Run preprocess first:\n"
            f"  python mydatasets/pretrain/oldpretrainmask.py --preprocess --model_path <path>"
        )
    with open(verified_path, "r") as f:
        info = json.load(f)
    logger.info(
        f"[oldpretrainmask] Cache verified: {info['num_texts']} sequences, "
        f"{info['total_tokens']:,} tokens"
    )
    return info


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OldPretrainMaskDataset(BaseDataset):
    """
    Chunked pretrain dataset with chat-template tokenization and pre-shuffled
    text sequences. Designed for SEQUENTIAL access (shuffle=false).

    Each sample returns a chunk of tokens from the concatenated stream.
    """

    def __init__(self, model_path: str, cache_dir: str, chunk_length: int,
                 big_chunk_size: int = -1):
        super().__init__(model_path)
        self.chunk_length = chunk_length
        self.cache_dir = cache_dir
        self.big_chunk_size = big_chunk_size

        # Validate big_chunk_size
        if big_chunk_size != -1:
            if big_chunk_size < chunk_length:
                raise ValueError(
                    f"big_chunk_size ({big_chunk_size}) must be >= chunk_length ({chunk_length}) or -1"
                )
            if big_chunk_size % chunk_length != 0:
                raise ValueError(
                    f"big_chunk_size ({big_chunk_size}) must be a multiple of "
                    f"chunk_length ({chunk_length}) or -1"
                )

        _check_verified_marker(cache_dir)

        manifest_path = os.path.join(cache_dir, MANIFEST_FILE)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        num_shards = manifest["num_shards"]

        eos_id = self.special_tokens["<|endoftext|>"]

        # Determine mmap filename based on big_chunk_size
        # Different big_chunk_size values produce different token streams,
        # so they need separate mmap files.
        if big_chunk_size == -1:
            mmap_path = os.path.join(cache_dir, "token_stream.npy")
        else:
            mmap_path = os.path.join(cache_dir, f"token_stream_bigchunk{big_chunk_size}.npy")
        lock_path = mmap_path + ".lock"
        verified_path = os.path.join(cache_dir, VERIFIED_FILE)

        t0 = time.time()

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
                        if big_chunk_size == -1:
                            self._build_mmap_legacy(num_shards, cache_dir, eos_id, mmap_path)
                        else:
                            self._build_mmap_bigchunk(num_shards, cache_dir, big_chunk_size, mmap_path)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        token_stream = np.load(mmap_path, mmap_mode="r")
        total_stream_len = token_stream.shape[0]
        logger.info(f"[oldpretrainmask] Loaded mmap: {total_stream_len:,} tokens ({time.time()-t0:.1f}s)")

        num_chunks = total_stream_len // chunk_length
        if num_chunks == 0:
            raise ValueError(f"Total tokens ({total_stream_len}) < chunk_length ({chunk_length})")

        used_tokens = num_chunks * chunk_length
        self.chunks = token_stream[:used_tokens].reshape(num_chunks, chunk_length)
        self._token_stream_mmap = token_stream

        # Deterministic distill offset: fixed cyclic shift ensures a bijection
        # (every chunk is used as a distill target exactly once per epoch).
        self._distill_offset = min(self.DISTILL_MIN_DISTANCE, num_chunks // 2)

        logger.info(
            f"[oldpretrainmask] {num_chunks:,} chunks of length {chunk_length}, "
            f"big_chunk_size={big_chunk_size}, distill_offset={self._distill_offset}"
        )

    def _build_mmap_legacy(self, num_shards: int, cache_dir: str, eos_id: int, mmap_path: str):
        """
        Legacy mmap building: concatenate all sequences with <|endoftext|> separators
        into one long token stream. Used when big_chunk_size == -1.
        """
        logger.info(f"[oldpretrainmask] Building legacy mmap from {num_shards} shards ...")
        shard_infos = []
        total_stream_len = 0
        for si in range(num_shards):
            sp = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
            data = np.load(sp)
            lengths = data["lengths"]
            n_tokens = int(lengths.astype(np.int64).sum())
            n_nonempty = int((lengths > 0).sum())
            shard_infos.append((sp, n_tokens, n_nonempty))
            # Each non-empty sequence contributes its tokens + 1 eos separator
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
        logger.info(f"[oldpretrainmask] Legacy mmap built: {total_stream_len:,} tokens")

    def _build_mmap_bigchunk(self, num_shards: int, cache_dir: str,
                             big_chunk_size: int, mmap_path: str):
        """
        Big-chunk mmap building: concatenate sequences WITHOUT eos separators,
        filling big_chunk_size-sized segments. Each big chunk starts with the
        beginning of a chat template sequence.

        Algorithm:
            - Track how many tokens have been written into the current big chunk.
            - For each sequence, write as many tokens as needed to fill the
              current big chunk. Once full, discard the rest of that sequence
              and start the next big chunk from the next sequence.

        This ensures every big_chunk_size boundary starts at a chat template
        beginning. Within a big chunk, later positions may be mid-sentence
        (from concatenated shorter texts), which is the desired behavior.
        """
        logger.info(
            f"[oldpretrainmask] Building big-chunk mmap (big_chunk_size={big_chunk_size}) "
            f"from {num_shards} shards ..."
        )

        # First pass: collect sequence metadata and count big chunks
        all_sequences = []  # list of (shard_path, offset_in_shard, length)
        for si in range(num_shards):
            sp = os.path.join(cache_dir, f"{SHARD_PREFIX}{si:06d}.npz")
            data = np.load(sp)
            lengths = data["lengths"]
            offset = 0
            for length in lengths:
                length = int(length)
                if length > 0:
                    all_sequences.append((sp, offset, length))
                offset += length
            del data, lengths

        # Simulate to count big chunks
        num_big_chunks = 0
        filled = 0  # tokens filled in current big chunk
        for _, _, seq_len in all_sequences:
            remaining_in_chunk = big_chunk_size - filled
            if seq_len >= remaining_in_chunk:
                # This sequence fills (or overfills) the current big chunk
                num_big_chunks += 1
                filled = 0  # next big chunk starts fresh from next sequence
            else:
                filled += seq_len

        total_stream_len = num_big_chunks * big_chunk_size
        logger.info(
            f"[oldpretrainmask] Big-chunk layout: {len(all_sequences)} sequences -> "
            f"{num_big_chunks} big chunks, {total_stream_len:,} tokens"
        )

        if total_stream_len == 0:
            raise ValueError(
                f"No big chunks produced. Sequences may be too short for "
                f"big_chunk_size={big_chunk_size}"
            )

        # Second pass: build the mmap by writing directly
        import socket
        mmap_tmp = f"{mmap_path}.tmp.{socket.gethostname()}.{os.getpid()}"
        fp = np.lib.format.open_memmap(mmap_tmp, mode="w+", dtype=np.int32, shape=(total_stream_len,))

        shard_cache: Dict[str, np.ndarray] = {}
        write_pos = 0  # absolute position in mmap
        filled = 0     # tokens filled in current big chunk

        for sp, offset, seq_len in all_sequences:
            if write_pos >= total_stream_len:
                break
            if sp not in shard_cache:
                shard_cache[sp] = np.load(sp)["tokens"]
            tokens = shard_cache[sp]

            remaining_in_chunk = big_chunk_size - filled
            if seq_len >= remaining_in_chunk:
                # Write only what's needed to fill the current big chunk
                fp[write_pos:write_pos + remaining_in_chunk] = tokens[offset:offset + remaining_in_chunk]
                write_pos += remaining_in_chunk
                # Discard the rest of this sequence; next big chunk starts fresh
                filled = 0
            else:
                # Sequence fits entirely in current big chunk
                fp[write_pos:write_pos + seq_len] = tokens[offset:offset + seq_len]
                write_pos += seq_len
                filled += seq_len

        assert write_pos == total_stream_len
        fp.flush()
        del fp, shard_cache
        os.rename(mmap_tmp, mmap_path)
        logger.info(f"[oldpretrainmask] Big-chunk mmap built: {total_stream_len:,} tokens")

    # Minimum distance (in chunks) between the primary chunk and the
    # distillation chunk to avoid data correlation.
    DISTILL_MIN_DISTANCE = 1000

    def __len__(self) -> int:
        return self.chunks.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = torch.from_numpy(np.array(self.chunks[idx], dtype=np.int64))

        # Deterministic mapping: fixed cyclic shift ensures bijection
        # (every chunk is distilled exactly once per epoch)
        num_chunks = self.chunks.shape[0]
        distill_idx = (idx + self._distill_offset) % num_chunks
        distill_chunk = torch.from_numpy(np.array(self.chunks[distill_idx], dtype=np.int64))

        return {"chunk": chunk, "distill_chunk": distill_chunk}


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------


def _find_text_content_ranges(
    tokens: torch.Tensor,
    end_think_id: int,
    newline_newline_id: int,
    im_end_id: int,
    im_start_id: int,
    eos_id: int,
) -> List[Tuple[int, int]]:
    """
    Find all text content ranges in a chunk.

    Text content is defined as the tokens AFTER the pattern </think>\n\n
    and BEFORE the next <|im_end|>.

    The pattern we look for is:
        </think>(248069) + \n\n(271) + [text content...] + <|im_end|>(248046)

    Additionally handles boundary cases:
    - If the chunk starts in the middle of a text (no preceding </think>\n\n),
      the content from position 0 up to the first <|im_end|> is treated as
      text content.
    - If the chunk ends in the middle of a text (no trailing <|im_end|>),
      the content from the last </think>\n\n to the end of the chunk is
      treated as text content.

    Returns:
        List of (start, end) tuples where start is inclusive and end is exclusive.
        These ranges represent the text content positions (not including </think>\n\n
        or <|im_end|> itself).
    """
    L = tokens.shape[0]
    ranges = []

    # --- Handle chunk starting in the middle of a text ---
    # If the chunk does NOT start with <|im_start|> or <|endoftext|>,
    # it means we're in the middle of a text. The content from position 0
    # up to the first <|im_end|> (or first <|endoftext|>) is text content.
    first_token = tokens[0].item()
    if first_token != im_start_id and first_token != eos_id:
        # Find the first <|im_end|> or <|endoftext|>
        content_end = 0
        while content_end < L:
            tid = tokens[content_end].item()
            if tid == im_end_id or tid == eos_id:
                break
            content_end += 1
        # content_end now points to <|im_end|>, <|endoftext|>, or end of chunk
        if content_end > 0:
            ranges.append((0, content_end))

    # --- Standard pattern matching: </think>\n\n ... <|im_end|> ---
    i = 0
    while i < L - 1:
        # Look for </think> followed by \n\n
        if tokens[i].item() == end_think_id and i + 1 < L and tokens[i + 1].item() == newline_newline_id:
            # Text content starts after </think>\n\n
            content_start = i + 2
            # Find the next <|im_end|>
            content_end = content_start
            while content_end < L:
                if tokens[content_end].item() == im_end_id:
                    break
                content_end += 1
            # content_end now points to <|im_end|> or end of chunk
            if content_start < content_end:
                ranges.append((content_start, content_end))
            i = content_end + 1
        else:
            i += 1
    return ranges


def _random_mask(
    tokens: torch.Tensor,
    content_ranges: List[Tuple[int, int]],
    mask_token_id: int,
    mask_ratio: float,
) -> torch.Tensor:
    """
    Apply random masking: independently mask each text content token with probability mask_ratio.

    Only tokens within content_ranges are eligible for masking.

    Args:
        tokens: (L,) token ids
        content_ranges: list of (start, end) tuples defining maskable regions
        mask_token_id: the <MASK> token ID to replace with
        mask_ratio: probability of masking each content token

    Returns:
        masked tokens (L,)
    """
    masked = tokens.clone()

    # Collect all content positions from ranges
    content_indices = []
    for start, end in content_ranges:
        content_indices.extend(range(start, end))

    num_content = len(content_indices)
    if num_content == 0:
        return masked

    num_to_mask = max(1, int(num_content * mask_ratio))
    content_indices_t = torch.tensor(content_indices, dtype=torch.long)
    perm = torch.randperm(num_content)[:num_to_mask]
    mask_positions = content_indices_t[perm]
    masked[mask_positions] = mask_token_id

    return masked


def _span_mask(
    tokens: torch.Tensor,
    content_ranges: List[Tuple[int, int]],
    mask_token_id: int,
    mask_ratio: float,
    span_mean_length: int = 3,
    span_max_length: int = 10,
) -> torch.Tensor:
    """
    Apply span masking: mask contiguous spans of text content tokens.

    Spans are sampled with lengths drawn from a geometric distribution
    (clamped to [1, span_max_length]). Spans only cover positions within
    content_ranges.

    Args:
        tokens: (L,) token ids
        content_ranges: list of (start, end) tuples defining maskable regions
        mask_token_id: the <MASK> token ID to replace with
        mask_ratio: target fraction of content tokens to mask
        span_mean_length: mean of geometric distribution for span lengths
        span_max_length: maximum span length

    Returns:
        masked tokens (L,)
    """
    masked = tokens.clone()

    # Collect all content positions from ranges (ordered)
    content_positions = []
    for start, end in content_ranges:
        content_positions.extend(range(start, end))

    num_content = len(content_positions)
    if num_content == 0:
        return masked

    num_to_mask = max(1, int(num_content * mask_ratio))

    # Generate spans over content positions
    masked_count = 0
    content_pos_set = set()  # indices in content_positions that are masked

    # Geometric distribution parameter: P(length=k) = (1-p)^(k-1) * p
    # Mean = 1/p => p = 1/span_mean_length
    p_geom = 1.0 / max(1, span_mean_length)

    while masked_count < num_to_mask:
        # Pick a random starting position in content space
        start_in_content = random.randint(0, num_content - 1)

        # Sample span length from geometric distribution
        span_len = 1
        while random.random() > p_geom and span_len < span_max_length:
            span_len += 1
        span_len = min(span_len, span_max_length)

        # Mask the span (in content position space)
        for offset in range(span_len):
            pos_in_content = start_in_content + offset
            if pos_in_content >= num_content:
                break
            if pos_in_content not in content_pos_set:
                content_pos_set.add(pos_in_content)
                masked_count += 1
                if masked_count >= num_to_mask:
                    break

    # Apply mask
    for pos_in_content in content_pos_set:
        actual_pos = content_positions[pos_in_content]
        masked[actual_pos] = mask_token_id

    return masked


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class OldPretrainMaskCollator(BaseCollator):
    """
    Collator for OldPretrainMaskDataset.

    - conversation_ids = chunk (unmasked, prediction target)
    - context_ids = chunk with masking ONLY on text content tokens
      (after </think>\n\n and before <|im_end|>), replaced by <MASK>
    - labels = only text content positions and trailing <|im_end|> have
      valid labels; all other positions are set to -100

    Masking is applied dynamically at collation time (different mask each epoch).
    """

    def __init__(
        self,
        model_path: str,
        context_max_length: int,
        conversation_max_length: int,
        pad_token_id: int,
        num_mem_token: int = 0,
        mask_ratio_start: float = 0.15,
        mask_ratio_end: float = 0.15,
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

        # Training progress state (updated externally via set_training_progress)
        self._current_step = 0
        self._max_steps = 1
        self._is_eval = False

        # Get <MASK> token ID
        if "<MASK>" not in self.special_tokens:
            raise ValueError(
                "<MASK> token not found in special_tokens. "
                "Ensure it is defined in configs/tokenizer/origin.yaml"
            )
        self._mask_token_id = self.special_tokens["<MASK>"]

        # Get token IDs needed for finding text content boundaries
        self._end_think_id = self.special_tokens["</think>"]
        self._im_end_id = self.special_tokens["<|im_end|>"]
        self._im_start_id = self.special_tokens["<|im_start|>"]
        self._eos_id = self.special_tokens["<|endoftext|>"]
        # \n\n token ID (271 for Qwen tokenizers)
        # We resolve it dynamically from the tokenizer
        _tokenizer = create_tokenizer(model_path, tokenizer_cfg=tokenizer_cfg)
        _nn_ids = _tokenizer.encode("\n\n", add_special_tokens=False)
        if len(_nn_ids) == 1:
            self._newline_newline_id = _nn_ids[0]
        else:
            # Fallback: use 271 which is the standard \n\n token for Qwen
            self._newline_newline_id = 271
            logger.warning(
                f"[OldPretrainMaskCollator] '\\n\\n' encodes to {_nn_ids}, "
                f"using fallback ID 271"
            )
        del _tokenizer

        logger.info(
            f"[OldPretrainMaskCollator] mask_ratio_start={mask_ratio_start}, "
            f"mask_ratio_end={mask_ratio_end}, mask_ratio_std={mask_ratio_std}, "
            f"strategy={mask_strategy}, mask_token_id={self._mask_token_id}, "
            f"end_think_id={self._end_think_id}, im_end_id={self._im_end_id}, "
            f"im_start_id={self._im_start_id}, eos_id={self._eos_id}, "
            f"newline_newline_id={self._newline_newline_id}"
        )

    def set_training_progress(self, current_step: int, max_steps: int):
        """Update training progress for dynamic mask ratio scheduling.

        Should be called before each training step by the training loop.

        Args:
            current_step: Current global training step (0-indexed).
            max_steps: Total number of training steps.
        """
        self._current_step = current_step
        self._max_steps = max(1, max_steps)

    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode. When True, no masking is applied to context_ids.

        Args:
            is_eval: Whether the collator is being used for validation.
        """
        self._is_eval = is_eval

    def _get_current_mask_ratio(self) -> float:
        """Compute the mask ratio for the current step.

        Linearly interpolates between mask_ratio_start and mask_ratio_end
        based on training progress, then samples from N(mean, std^2).
        Result is clamped to [0, 1].

        Returns:
            Sampled mask ratio for this step.
        """
        # Linear interpolation of the mean
        progress = self._current_step / self._max_steps  # 0.0 -> 1.0
        progress = min(1.0, max(0.0, progress))
        mean_ratio = self.mask_ratio_start + (self.mask_ratio_end - self.mask_ratio_start) * progress

        # Sample around the mean with Gaussian noise
        if self.mask_ratio_std > 0:
            sampled_ratio = random.gauss(mean_ratio, self.mask_ratio_std)
        else:
            sampled_ratio = mean_ratio

        # Clamp to [0, 1]
        return max(0.0, min(1.0, sampled_ratio))

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        batch_size = len(samples)
        L = self.chunk_length
        num_mem = self.num_mem_token
        context_total_len = L + num_mem

        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.empty((batch_size, L), dtype=torch.long)
        labels = torch.full((batch_size, L), -100, dtype=torch.long)
        context_lengths = torch.full((batch_size,), L, dtype=torch.long)

        # Distillation tensors
        distill_conversation_ids = torch.empty((batch_size, L), dtype=torch.long)
        distill_labels = torch.full((batch_size, L), -100, dtype=torch.long)

        # Determine mask ratio: no masking in eval mode, dynamic in training
        if self._is_eval:
            effective_mask_ratio = 0.0
        else:
            effective_mask_ratio = self._get_current_mask_ratio()

        for i, s in enumerate(samples):
            chunk = s["chunk"]  # (L,) unmasked tokens

            # conversation_ids = unmasked chunk (prediction target)
            conversation_ids[i] = chunk

            # Find text content ranges: positions after </think>\n\n and before <|im_end|>
            # Also handles boundary cases (chunk starts/ends in middle of text)
            content_ranges = _find_text_content_ranges(
                chunk, self._end_think_id, self._newline_newline_id, self._im_end_id,
                self._im_start_id, self._eos_id,
            )

            # labels: only keep text content positions and trailing <|im_end|>
            # All other positions remain -100
            for start, end in content_ranges:
                labels[i, start:end] = chunk[start:end]
                # Also include the <|im_end|> right after the content
                if end < L and chunk[end].item() == self._im_end_id:
                    labels[i, end] = chunk[end]

            # context_ids = masked chunk (only mask within text content ranges)
            if effective_mask_ratio > 0 and content_ranges:
                if self.mask_strategy == "random":
                    masked_chunk = _random_mask(
                        chunk, content_ranges, self._mask_token_id, effective_mask_ratio
                    )
                elif self.mask_strategy == "span":
                    masked_chunk = _span_mask(
                        chunk, content_ranges, self._mask_token_id,
                        effective_mask_ratio, self.span_mean_length, self.span_max_length
                    )
                else:
                    raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")
                context_ids[i, :L] = masked_chunk
            else:
                # No masking or no content ranges — context_ids == conversation_ids
                context_ids[i, :L] = chunk

            # Distillation: use a distant chunk, apply same label logic
            distill_chunk = s["distill_chunk"]  # (L,)
            distill_conversation_ids[i] = distill_chunk
            distill_content_ranges = _find_text_content_ranges(
                distill_chunk, self._end_think_id, self._newline_newline_id,
                self._im_end_id, self._im_start_id, self._eos_id,
            )
            for start, end in distill_content_ranges:
                distill_labels[i, start:end] = distill_chunk[start:end]
                if end < L and distill_chunk[end].item() == self._im_end_id:
                    distill_labels[i, end] = distill_chunk[end]

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
# Factory function
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """Create OldPretrainMaskDataset and OldPretrainMaskCollator."""
    data_cfg = cfg.data
    context_seq_len = data_cfg.context_seq_length
    conv_seq_len = data_cfg.conv_seq_length

    if context_seq_len != conv_seq_len:
        raise ValueError(
            f"context_seq_length ({context_seq_len}) must equal conv_seq_length ({conv_seq_len})"
        )

    _raw_cache = data_cfg.get("cache_dir", "cache/oldpretrainmask_tokens")
    cache_dir = _raw_cache if os.path.isabs(_raw_cache) else os.path.join(_project_root, _raw_cache)

    # Big chunk size: -1 means legacy behavior (one long stream with eos separators)
    big_chunk_size = data_cfg.get("big_chunk_size", -1)

    # Read mask settings from config (no backward compatibility with old 'ratio' field)
    mask_cfg = data_cfg.mask
    mask_ratio_start = mask_cfg.ratio_start
    mask_ratio_end = mask_cfg.ratio_end
    mask_ratio_std = mask_cfg.ratio_std
    mask_strategy = mask_cfg.strategy
    span_mean_length = mask_cfg.get("span_mean_length", 3)
    span_max_length = mask_cfg.get("span_max_length", 10)

    logger.info(
        f"[oldpretrainmask] Creating dataset: chunk_length={context_seq_len}, "
        f"big_chunk_size={big_chunk_size}, "
        f"cache='{cache_dir}', mask_ratio={mask_ratio_start}->{mask_ratio_end} "
        f"(std={mask_ratio_std}), strategy={mask_strategy}"
    )

    train_ds = OldPretrainMaskDataset(
        model_path=model_path, cache_dir=cache_dir, chunk_length=context_seq_len,
        big_chunk_size=big_chunk_size,
    )
    collator = OldPretrainMaskCollator(
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
    """Debug: inspect first few samples."""
    from utils.mydata import resolve_pad_token_id, debug_dataset
    from omegaconf import OmegaConf

    if "model" not in cfg:
        cfg = OmegaConf.merge(cfg, {"model": {"path": model_path}})

    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    num_mem_token = 10
    dataset, collator = create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)

    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer)

    mask_cfg = cfg.data.mask
    metadata = {
        "context_seq_len": cfg.data.context_seq_length,
        "conv_seq_len": cfg.data.conv_seq_length,
        "num_chunks": len(dataset),
        "shuffle": cfg.data.get("shuffle", False),
        "mask_ratio_start": mask_cfg.ratio_start,
        "mask_ratio_end": mask_cfg.ratio_end,
        "mask_ratio_std": mask_cfg.ratio_std,
        "mask_strategy": mask_cfg.strategy,
    }
    debug_dataset(
        dataset=dataset, collator=collator, tokenizer=tokenizer,
        dataset_name=cfg.data.get("name", "oldpretrainmask"),
        metadata=metadata, num_samples=5, num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="oldpretrainmask dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true")
    group.add_argument("--preprocess", action="store_true")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/oldpretrainmask.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    data_cfg = OmegaConf.load(args.config)
    _base_yaml = os.path.join("configs", "base.yaml")
    _base_cfg = OmegaConf.load(_base_yaml) if os.path.exists(_base_yaml) else OmegaConf.create({})
    _dataset_seed = _base_cfg.get("seed", {}).get("dataset", 42)
    cfg = OmegaConf.create({"data": data_cfg, "seed": {"dataset": _dataset_seed}})

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
        preprocess(cfg, model_path)
    elif args.debug:
        if model_path is None:
            print("ERROR: --model_path required for --debug")
            sys.exit(1)
        debug(cfg, model_path)
