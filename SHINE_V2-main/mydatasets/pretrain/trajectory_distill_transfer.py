#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory Distill Transfer Dataset Module for SHINE_V2

Loads trajectories from SHINE_SWE_DISTILLATION, tokenizes each trajectory
using the chat template, caches the results (shared cache with recon dataset),
and groups trajectories by repo for transfer learning.

For each repo:
    - All trajectories belonging to that repo are collected and shuffled.
    - Each trajectory becomes one data sample where:
        - context_ids = this trajectory's tokens
        - conversation_ids = the NEXT trajectory's tokens (circular: last -> first)
    - Labels mirror conversation_ids (with padding masked as -100).

Data ordering:
    Trajectories from the same repo are kept contiguous (never shuffled globally).
    This ensures that after DP splitting, each rank gets contiguous repo blocks.

Each sample also includes extra_info = {"repo": repo_name} for training-time
behavior modification (e.g., detach_state reset logic).

Preprocessing:
    Tokenizes all trajectories in parallel using multiprocessing, applies
    the chat template via the tokenizer, and stores results in a cache
    directory as .npz files. Uses file locking to avoid conflicts when
    both recon and transfer datasets preprocess simultaneously.

Dataset:
    Loads cached tokens and repo metadata, groups by repo, filters by
    max_token_length, and creates consecutive-pair samples.

Unified factory interface:
    create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
        -> (TrajectoryDistillTransferDataset, TrajectoryDistillTransferCollator)
"""

from __future__ import annotations

import os
import json
import random
import logging
import time
import multiprocessing as mp
import fcntl
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch

# Ensure project root is on sys.path so that local package imports work
import sys as _sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from mydatasets.base import BaseDataset, BaseCollator
from utils.mytokenizer import create_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_FILE = "all_trajectories.npz"   # Single cache file with all tokenized trajectories
REPO_CACHE_FILE = "repo_metadata.json"  # Repo info for each trajectory
MANIFEST_FILE = "manifest.json"
VERIFIED_FILE = "VERIFIED"
LOCK_FILE = ".cache_lock"


# ---------------------------------------------------------------------------
# Parallel tokenization worker
# ---------------------------------------------------------------------------

def _init_worker(tokenizer_path: str, tokenizer_cfg_dict: dict):
    """Pool initializer: load tokenizer once per worker."""
    global _worker_tokenizer
    from omegaconf import OmegaConf
    _worker_tokenizer = create_tokenizer(
        tokenizer_path, tokenizer_cfg=OmegaConf.create(tokenizer_cfg_dict)
    )


def _fix_tools(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """
    Fix tool definitions loaded from arrow files.
    The 'properties' field in parameters may be stored as a JSON string
    instead of a dict — parse it if needed.
    """
    if not tools:
        return None
    fixed_tools = []
    for tool in tools:
        tool = dict(tool)  # shallow copy
        if "function" in tool and tool["function"]:
            func = dict(tool["function"])
            if "parameters" in func and func["parameters"]:
                params = dict(func["parameters"])
                # Fix properties: may be a JSON string instead of dict
                if "properties" in params and isinstance(params["properties"], str):
                    try:
                        params["properties"] = json.loads(params["properties"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                func["parameters"] = params
            tool["function"] = func
        fixed_tools.append(tool)
    return fixed_tools


def _fix_messages(messages: List[Dict]) -> List[Dict]:
    """
    Fix messages loaded from arrow files.
    - Remove empty string fields (name, tool_call_id) that should be None
    - Ensure tool_calls is properly formatted
    """
    fixed_messages = []
    for msg in messages:
        msg = dict(msg)  # shallow copy
        # Remove empty string fields that should be absent/None
        for key in ("name", "tool_call_id"):
            if key in msg and msg[key] == "":
                del msg[key]
        # Remove empty reasoning_content
        if "reasoning_content" in msg and msg["reasoning_content"] == "":
            del msg["reasoning_content"]
        # Remove empty content for tool-calling messages
        if msg.get("role") == "assistant" and msg.get("content") == "" and msg.get("tool_calls"):
            del msg["content"]
        # Fix tool_calls if present
        if "tool_calls" in msg:
            if not msg["tool_calls"]:
                del msg["tool_calls"]
            else:
                # Ensure tool_calls entries are properly structured
                fixed_tc = []
                for tc in msg["tool_calls"]:
                    tc = dict(tc)
                    if "function" in tc and tc["function"]:
                        func = dict(tc["function"])
                        # Parse arguments from JSON string to dict
                        if "arguments" in func and isinstance(func["arguments"], str):
                            try:
                                func["arguments"] = json.loads(func["arguments"])
                            except (json.JSONDecodeError, TypeError):
                                pass
                        tc["function"] = func
                    fixed_tc.append(tc)
                msg["tool_calls"] = fixed_tc
        fixed_messages.append(msg)
    return fixed_messages


def _tokenize_trajectory(args: Tuple) -> Optional[Tuple[int, np.ndarray]]:
    """
    Tokenize a single trajectory (messages + tools) using the chat template.

    Args:
        args: (index, messages, tools)

    Returns:
        (index, token_ids_array) on success, None on failure.
    """
    idx, messages, tools = args
    try:
        global _worker_tokenizer
        # Fix data format issues from arrow storage
        fixed_messages = _fix_messages(messages)
        fixed_tools = _fix_tools(tools)

        # Apply chat template to get text, then tokenize manually
        text = _worker_tokenizer.apply_chat_template(
            fixed_messages,
            tools=fixed_tools,
            add_generation_prompt=False,
            tokenize=False,
        )
        token_ids = _worker_tokenizer.encode(text, add_special_tokens=False)
        return (idx, np.array(token_ids, dtype=np.int32))
    except Exception as e:
        logger.warning(f"[trajectory_distill_transfer] Failed to tokenize trajectory {idx}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _load_all_shards(data_path: str) -> List[Dict[str, Any]]:
    """
    Load all arrow shard directories under data_path and return a list of rows.
    Each row is a dict with 'messages', 'tools', and 'repo' keys.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)

    all_rows = []
    shard_dirs = sorted([
        d for d in os.listdir(abs_data_path)
        if os.path.isdir(os.path.join(abs_data_path, d)) and not d.startswith(".")
    ])

    for shard_dir in shard_dirs:
        shard_path = os.path.join(abs_data_path, shard_dir)
        # Find arrow files in the shard directory
        arrow_files = sorted([
            f for f in os.listdir(shard_path) if f.endswith(".arrow")
        ])
        for arrow_file in arrow_files:
            arrow_path = os.path.join(shard_path, arrow_file)
            f = pa.memory_map(arrow_path, "r")
            reader = ipc.open_stream(f)
            table = reader.read_all()
            # Convert entire table to columnar dict at once (much faster than row-by-row)
            col_dict = table.to_pydict()
            has_tools = "tools" in col_dict
            has_repo = "repo" in col_dict
            num_rows = table.num_rows
            messages_col = col_dict["messages"]
            tools_col = col_dict["tools"] if has_tools else [None] * num_rows
            repo_col = col_dict["repo"] if has_repo else ["unknown"] * num_rows
            for i in range(num_rows):
                all_rows.append({
                    "messages": messages_col[i],
                    "tools": tools_col[i],
                    "repo": repo_col[i],
                })
            f.close()

    logger.info(f"[trajectory_distill_transfer] Loaded {len(all_rows)} trajectories from {len(shard_dirs)} shards")
    return all_rows


# ---------------------------------------------------------------------------
# Preprocessing: tokenize all trajectories and cache
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Tokenize all trajectories in parallel and cache the results.
    Uses file locking to avoid conflicts when both recon and transfer
    datasets preprocess simultaneously (they share the same cache dir).

    Cache layout::
        {cache_dir}/
            manifest.json              # metadata (num_trajectories, etc.)
            all_trajectories.npz       # tokens and offsets for all trajectories
            repo_metadata.json         # repo name for each trajectory (indexed)
            VERIFIED                   # marker written after successful caching

    The .npz file stores:
        - tokens: 1-D int32 array (concatenation of all trajectory token sequences)
        - offsets: 1-D int64 array (start offset of each trajectory in tokens array)
        - lengths: 1-D int32 array (length of each trajectory)
    """
    import io
    from omegaconf import OmegaConf

    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "trajectory_distill_transfer")
    data_path = data_cfg.get("data_path", "data/SHINE_SWE_DISTILLATION")
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_distill_recon_tokens")
    num_workers = data_cfg.get("preprocess_workers", 32)

    # Resolve paths
    abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
    os.makedirs(abs_cache_dir, exist_ok=True)

    # Output file for logging
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

    # Use file locking to prevent concurrent writes
    lock_path = os.path.join(abs_cache_dir, LOCK_FILE)
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Check if already verified (after acquiring lock)
        verified_path = os.path.join(abs_cache_dir, VERIFIED_FILE)
        repo_cache_path = os.path.join(abs_cache_dir, REPO_CACHE_FILE)

        if os.path.exists(verified_path) and os.path.exists(repo_cache_path):
            _print(f"  Cache already verified at: {abs_cache_dir}")
            _print(f"  Repo metadata already exists.")
            _print(f"  Skipping preprocessing.")
            _print(sep)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(buf.getvalue())
            buf.close()
            return

        # Load all trajectories
        _print(f"  Loading trajectories from: {data_path}")
        t0 = time.time()
        all_rows = _load_all_shards(data_path)
        num_trajectories = len(all_rows)
        _print(f"  Loaded {num_trajectories} trajectories in {time.time() - t0:.1f}s")

        # Check if token cache already exists (recon may have created it)
        cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
        if os.path.exists(verified_path) and os.path.exists(cache_path):
            _print(f"  Token cache already exists (created by recon dataset).")
            _print(f"  Only creating repo metadata...")
        else:
            # Prepare tokenizer config for workers
            tokenizer_cfg_dict = OmegaConf.to_container(cfg.tokenizer, resolve=True)

            # Prepare work items
            work_items = [
                (i, row["messages"], row["tools"])
                for i, row in enumerate(all_rows)
            ]

            # Parallel tokenization
            _print(f"  Tokenizing with {num_workers} workers...")
            t0 = time.time()

            results = [None] * num_trajectories
            num_failed = 0

            with mp.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(model_path, tokenizer_cfg_dict),
            ) as pool:
                for result in pool.imap_unordered(_tokenize_trajectory, work_items, chunksize=64):
                    if result is not None:
                        idx, token_ids = result
                        results[idx] = token_ids
                    else:
                        num_failed += 1

            # Filter out failed tokenizations
            valid_results = [(i, r) for i, r in enumerate(results) if r is not None]
            _print(f"  Tokenized {len(valid_results)} trajectories in {time.time() - t0:.1f}s")
            if num_failed > 0:
                _print(f"  WARNING: {num_failed} trajectories failed tokenization")

            if len(valid_results) == 0:
                _print(f"  ERROR: All trajectories failed tokenization. Cannot create cache.")
                _print(sep)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(buf.getvalue())
                buf.close()
                raise RuntimeError("All trajectories failed tokenization.")

            # Build concatenated arrays
            lengths = np.array([len(r) for _, r in valid_results], dtype=np.int32)
            offsets = np.zeros(len(valid_results) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(lengths.astype(np.int64))

            total_tokens = int(offsets[-1])
            _print(f"  Total tokens: {total_tokens:,}")
            _print(f"  Token length stats: min={lengths.min()}, max={lengths.max()}, "
                   f"mean={lengths.mean():.0f}, median={np.median(lengths):.0f}")

            # Concatenate all tokens
            tokens = np.concatenate([r for _, r in valid_results])

            # Save cache
            _print(f"  Saving token cache to: {cache_path}")
            np.savez(
                cache_path,
                tokens=tokens,
                offsets=offsets,
                lengths=lengths,
            )

            # Save manifest
            manifest = {
                "num_trajectories": len(valid_results),
                "total_tokens": total_tokens,
                "num_failed": num_failed,
                "data_path": data_path,
                "model_path": model_path,
                "length_stats": {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": float(lengths.mean()),
                    "median": float(np.median(lengths)),
                },
            }
            manifest_path = os.path.join(abs_cache_dir, MANIFEST_FILE)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # Write verified marker
            with open(verified_path, "w") as f:
                f.write(f"verified at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            _print(f"  Token cache verified and saved.")

        # Save repo metadata (maps trajectory index -> repo name)
        # This uses the same ordering as the token cache (valid trajectories only)
        # We need to know which indices were valid
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            num_cached = len(data["lengths"])
        else:
            num_cached = len(valid_results)

        # Build repo list matching the cache order
        # If token cache was created by recon (which doesn't save repo info),
        # we need to re-derive the mapping. The cache stores trajectories in
        # the order they were successfully tokenized (original index order,
        # with failures removed).
        repo_list = []
        failed_indices = set()
        if 'valid_results' in dir():
            # We just tokenized — use valid_results ordering
            for orig_idx, _ in valid_results:
                repo_list.append(all_rows[orig_idx]["repo"])
        else:
            # Token cache was created by recon — we need to re-derive
            # by attempting tokenization order (same deterministic order)
            # Since recon uses the same _load_all_shards, the ordering is identical.
            # We just need to figure out which ones failed.
            # Load the manifest to get num_failed
            manifest_path = os.path.join(abs_cache_dir, MANIFEST_FILE)
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            if manifest["num_trajectories"] == num_trajectories:
                # All succeeded — simple case
                repo_list = [row["repo"] for row in all_rows]
            else:
                # Some failed — we need to re-tokenize to find which ones
                # For efficiency, just do a quick dry-run with single worker
                _print(f"  Re-deriving valid trajectory indices...")
                tokenizer_cfg_dict = OmegaConf.to_container(cfg.tokenizer, resolve=True)
                work_items = [
                    (i, row["messages"], row["tools"])
                    for i, row in enumerate(all_rows)
                ]
                with mp.Pool(
                    processes=num_workers,
                    initializer=_init_worker,
                    initargs=(model_path, tokenizer_cfg_dict),
                ) as pool:
                    for result in pool.imap_unordered(_tokenize_trajectory, work_items, chunksize=64):
                        if result is None:
                            pass  # failed
                        else:
                            idx, _ = result
                            # We only need to know which succeeded
                # Actually, simpler: just check lengths match
                # The cache has exactly num_cached entries in original index order
                # (skipping failures). We can rebuild by trying each and checking.
                # But this is complex. Instead, just re-tokenize and save repo info.
                _print(f"  WARNING: Cannot determine valid indices without re-tokenization.")
                _print(f"  Re-tokenizing to build repo metadata...")
                results_check = [None] * num_trajectories
                with mp.Pool(
                    processes=num_workers,
                    initializer=_init_worker,
                    initargs=(model_path, tokenizer_cfg_dict),
                ) as pool:
                    for result in pool.imap_unordered(_tokenize_trajectory, work_items, chunksize=64):
                        if result is not None:
                            idx, _ = result
                            results_check[idx] = True
                for i in range(num_trajectories):
                    if results_check[i] is not None:
                        repo_list.append(all_rows[i]["repo"])

        assert len(repo_list) == num_cached, (
            f"Repo list length ({len(repo_list)}) != cached trajectories ({num_cached})"
        )

        # Save repo metadata
        repo_metadata = {
            "repos": repo_list,
            "num_trajectories": num_cached,
        }
        with open(repo_cache_path, "w", encoding="utf-8") as f:
            json.dump(repo_metadata, f)
        _print(f"  Saved repo metadata ({len(set(repo_list))} unique repos)")

        _print(f"  Preprocessing complete.")
        _print(sep)

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[trajectory_distill_transfer] Preprocessing complete. Output: {output_path}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryDistillTransferDataset(BaseDataset):
    """
    Dataset that loads pre-tokenized trajectories from cache, groups them
    by repo, and creates consecutive-pair samples for transfer learning.

    For each repo, trajectories are shuffled, then each trajectory i is paired
    with trajectory (i+1) % num_in_repo:
        - context_ids = trajectory[i] tokens
        - conversation_ids = trajectory[(i+1) % n] tokens (the "next" one)
        - labels = same as conversation_ids (padding masked as -100)

    Samples from the same repo are kept contiguous. Repos are shuffled each epoch.

    Per-epoch reshuffling:
        Calling set_epoch(epoch) re-shuffles both the repo order and the
        within-repo trajectory order, producing different pairings each epoch.
        The seed for each epoch is derived from (base_seed + epoch) to ensure
        reproducibility and cross-rank consistency.

    Repo filtering:
        If repo_names is provided, only those repos are included. This is used
        to create separate train/val datasets at the repo level.
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        max_token_length: int,
        seed: int = 42,
        repo_names: Optional[List[str]] = None,
    ):
        super().__init__(model_path)
        self.max_token_length = max_token_length
        self._base_seed = seed

        # Load cache
        abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
        verified_path = os.path.join(abs_cache_dir, VERIFIED_FILE)
        if not os.path.exists(verified_path):
            raise RuntimeError(
                f"Cache not verified at {abs_cache_dir}. "
                f"Run preprocessing first: python mydatasets/pretrain/trajectory_distill_transfer.py --preprocess"
            )

        repo_cache_path = os.path.join(abs_cache_dir, REPO_CACHE_FILE)
        if not os.path.exists(repo_cache_path):
            raise RuntimeError(
                f"Repo metadata not found at {repo_cache_path}. "
                f"Run preprocessing for transfer dataset first."
            )

        cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
        data = np.load(cache_path)
        self._tokens = data["tokens"]
        self._offsets = data["offsets"]
        self._lengths = data["lengths"]

        # Load repo metadata
        with open(repo_cache_path, "r") as f:
            repo_metadata = json.load(f)
        repo_list = repo_metadata["repos"]

        # Group trajectories by repo, filtering by max_token_length
        all_repo_to_valid_indices: Dict[str, List[int]] = defaultdict(list)
        for i, repo in enumerate(repo_list):
            if self._lengths[i] <= max_token_length:
                all_repo_to_valid_indices[repo].append(i)

        # Apply repo filter if provided
        if repo_names is not None:
            repo_names_set = set(repo_names)
            self._repo_to_valid_indices = {
                k: v for k, v in all_repo_to_valid_indices.items()
                if k in repo_names_set
            }
        else:
            self._repo_to_valid_indices = dict(all_repo_to_valid_indices)

        # Pre-load all valid trajectory tokens into memory for fast access
        self._trajectory_tokens: Dict[int, torch.Tensor] = {}
        for indices in self._repo_to_valid_indices.values():
            for i in indices:
                start = int(self._offsets[i])
                end = int(self._offsets[i + 1])
                self._trajectory_tokens[i] = torch.from_numpy(
                    self._tokens[start:end].astype(np.int64)
                )

        # Build initial samples (epoch 0)
        self.samples: List[Dict[str, Any]] = []
        self._build_samples(seed)

        logger.info(
            f"[trajectory_distill_transfer] Created {len(self.samples)} samples "
            f"from {len(self._repo_to_valid_indices)} repos "
            f"(max_token_length={max_token_length})"
        )

    def _build_samples(self, seed: int):
        """
        Build the samples list by shuffling repo order and within-repo
        trajectory order using the given seed.

        Args:
            seed: Random seed for this epoch's shuffling.
        """
        rng = random.Random(seed)
        self.samples = []

        # Shuffle repo order
        repo_names = list(self._repo_to_valid_indices.keys())
        rng.shuffle(repo_names)

        for repo_name in repo_names:
            valid_indices = list(self._repo_to_valid_indices[repo_name])

            if len(valid_indices) < 2:
                # Need at least 2 trajectories to form pairs
                # If only 1, use it as both context and conversation
                if len(valid_indices) == 1:
                    i = valid_indices[0]
                    token_ids = self._trajectory_tokens[i]
                    self.samples.append({
                        "context_token_ids": token_ids,
                        "conversation_token_ids": token_ids,
                        "repo": repo_name,
                    })
                continue

            # Shuffle within repo
            rng.shuffle(valid_indices)

            # Create consecutive pairs
            for pos in range(len(valid_indices)):
                ctx_idx = valid_indices[pos]
                conv_idx = valid_indices[(pos + 1) % len(valid_indices)]

                self.samples.append({
                    "context_token_ids": self._trajectory_tokens[ctx_idx],
                    "conversation_token_ids": self._trajectory_tokens[conv_idx],
                    "repo": repo_name,
                })

    def set_epoch(self, epoch: int):
        """
        Re-shuffle repo order and within-repo trajectory order for the
        given epoch. This produces different pairings each epoch while
        maintaining repo-contiguous ordering.

        Must be called before iterating the dataset for a new epoch.
        All DP ranks must call this with the same epoch to ensure
        consistent dataset length and ordering.

        Args:
            epoch: The epoch number (0-indexed).
        """
        epoch_seed = self._base_seed + epoch
        self._build_samples(epoch_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "context_token_ids": sample["context_token_ids"],
            "conversation_token_ids": sample["conversation_token_ids"],
            "repo": sample["repo"],
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Label masking utilities
# ---------------------------------------------------------------------------

# Token IDs for label masking (Qwen3 tokenizer)
_THINK_TOKEN_ID = 248068       # <think>
_END_THINK_TOKEN_ID = 248069   # </think>
_NEWLINE_TOKEN_ID = 198        # '\n'
_DOUBLE_NEWLINE_TOKEN_ID = 271 # '\n\n'
_IM_END_TOKEN_ID = 248046      # <|im_end|>


def _compute_masked_labels(conv_tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute labels with masking: only mask <think> itself, then unmask
    starting from the \n or \n\n that follows, up to and including
    <|im_end|> for each assistant turn. Everything else is -100.

    The pattern we look for:
        <think> (248068) followed by \n (198) or \n\n (271)
    Only <think> is masked. The \n/\n\n and all subsequent tokens are kept
    in labels until we hit <|im_end|> (248046, inclusive). Then we mask again
    until the next <think> token.

    Args:
        conv_tokens: 1-D LongTensor of conversation token ids (no padding).

    Returns:
        labels: 1-D LongTensor same shape as conv_tokens, with -100 for
                masked positions.
    """
    labels = torch.full_like(conv_tokens, -100)
    length = conv_tokens.size(0)
    in_valid_region = False
    i = 0

    while i < length:
        if not in_valid_region:
            # Look for <think> token
            if conv_tokens[i].item() == _THINK_TOKEN_ID:
                # Mask <think> itself, then enter valid region
                i += 1
                in_valid_region = True
            else:
                i += 1
        else:
            # In valid region: keep tokens until <|im_end|> (inclusive)
            labels[i] = conv_tokens[i]
            if conv_tokens[i].item() == _IM_END_TOKEN_ID:
                in_valid_region = False
            i += 1

    return labels


class TrajectoryDistillTransferCollator(BaseCollator):
    """
    Collator that pads trajectories to fixed lengths for transfer learning.

    Each trajectory is right-padded to max_token_length.
    Labels only include tokens after <think>\\n up to and including <|im_end|>
    for each assistant turn. All other positions are masked as -100.

    Produced batch keys:
        context_ids:      (B, context_total_len)  where context_total_len = max_token_length + num_mem_token
        conversation_ids: (B, max_token_length)
        labels:           (B, max_token_length)
        context_lengths:  (B,)  actual number of valid tokens per context sample
        extra_info:       list of dicts, each with {"repo": repo_name}
    """

    def __init__(
        self,
        model_path: str,
        max_token_length: int,
        pad_token_id: int = 0,
        num_mem_token: int = 0,
    ):
        super().__init__(model_path)
        self.max_token_length = max_token_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch_size = len(samples)
        max_len = self.max_token_length
        num_mem = self.num_mem_token

        # context_total_len includes space for mem_token placeholders
        context_total_len = max_len + num_mem

        # Pre-allocate tensors (right-padded)
        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        context_lengths = torch.zeros(batch_size, dtype=torch.long)
        extra_info_list = []

        for i, s in enumerate(samples):
            # Context (this trajectory)
            ctx_len = min(s["context_token_ids"].size(0), max_len)
            context_ids[i, :ctx_len] = s["context_token_ids"][:ctx_len]
            context_lengths[i] = ctx_len

            # Conversation (next trajectory in the same repo)
            conv_tokens = s["conversation_token_ids"][:max_len]
            conv_len = conv_tokens.size(0)
            conversation_ids[i, :conv_len] = conv_tokens

            # Compute masked labels: only keep tokens after <think>\n
            # up to and including <|im_end|>
            masked_labels = _compute_masked_labels(conv_tokens)
            labels[i, :conv_len] = masked_labels

            # Extra info
            extra_info_list.append({"repo": s["repo"]})

        return [{
            "context_ids": context_ids,
            "conversation_ids": conversation_ids,
            "labels": labels,
            "context_lengths": context_lengths,
            "extra_info": extra_info_list,
        }]


# ---------------------------------------------------------------------------
# Train/Val repo-level split helper
# ---------------------------------------------------------------------------

def _get_train_val_repo_split(
    cache_dir: str,
    max_token_length: int,
    validation_split_num: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split repos into train and val sets based on validation_split_num.

    Strategy: shuffle all repos with a fixed seed, then greedily assign
    repos to the val set (in shuffled order) until the total number of
    val samples reaches validation_split_num. The remaining repos go to
    the train set.

    This ensures:
    - Val set is a fixed set of complete repos (repo-contiguous)
    - Train set contains all other repos
    - The split is deterministic given the same seed

    Args:
        cache_dir: Path to the cache directory.
        max_token_length: Maximum token length for filtering.
        validation_split_num: Target number of validation samples.
        seed: Random seed for the split.

    Returns:
        (train_repo_names, val_repo_names)
    """
    abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
    cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
    repo_cache_path = os.path.join(abs_cache_dir, REPO_CACHE_FILE)

    data = np.load(cache_path)
    lengths = data["lengths"]

    with open(repo_cache_path, "r") as f:
        repo_metadata = json.load(f)
    repo_list = repo_metadata["repos"]

    # Count valid samples per repo
    repo_sample_counts: Dict[str, int] = defaultdict(int)
    for i, repo in enumerate(repo_list):
        if lengths[i] <= max_token_length:
            repo_sample_counts[repo] += 1

    # Shuffle repos deterministically
    rng = random.Random(seed)
    all_repos = list(repo_sample_counts.keys())
    rng.shuffle(all_repos)

    # Greedily assign repos to val until we reach validation_split_num
    val_repos = []
    val_total = 0
    for repo in all_repos:
        if val_total >= validation_split_num:
            break
        count = repo_sample_counts[repo]
        if count > 0:
            val_repos.append(repo)
            val_total += count

    val_repo_set = set(val_repos)
    train_repos = [r for r in all_repos if r not in val_repo_set]

    logger.info(
        f"[trajectory_distill_transfer] Repo-level split: "
        f"train={len(train_repos)} repos, val={len(val_repos)} repos "
        f"(val_samples≈{val_total}, target={validation_split_num})"
    )

    return train_repos, val_repos


# ---------------------------------------------------------------------------
# Factory function -- unified interface
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a TrajectoryDistillTransferDataset and its collator.

    If validation_split_num is set, the train dataset only contains train repos
    (val repos are excluded). The val dataset is created separately via
    create_val_dataset().

    Parameters from cfg.data (trajectory_distill_transfer.yaml):
        - max_token_length:   maximum token length for filtering trajectories
        - cache_dir:          directory containing preprocessed cache
        - validation_split_num: target number of validation samples (repo-level split)

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (TrajectoryDistillTransferDataset, TrajectoryDistillTransferCollator)
    """
    data_cfg = cfg.data

    max_token_length = data_cfg.max_token_length
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_distill_recon_tokens")
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)

    logger.info(
        f"[trajectory_distill_transfer] Creating dataset: "
        f"max_token_length={max_token_length}, cache_dir={cache_dir}"
    )

    # Determine train repos (exclude val repos if validation is enabled)
    train_repo_names = None
    if validation_split_num > 0:
        train_repos, _ = _get_train_val_repo_split(
            cache_dir=cache_dir,
            max_token_length=max_token_length,
            validation_split_num=validation_split_num,
            seed=seed,
        )
        train_repo_names = train_repos

    dataset = TrajectoryDistillTransferDataset(
        model_path=model_path,
        cache_dir=cache_dir,
        max_token_length=max_token_length,
        seed=seed,
        repo_names=train_repo_names,
    )

    collator = TrajectoryDistillTransferCollator(
        model_path=model_path,
        max_token_length=max_token_length,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
    )

    return dataset, collator


def create_val_dataset(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a validation dataset with a fixed set of repos.

    The val dataset uses a fixed seed (no per-epoch reshuffling) and contains
    only the repos assigned to validation by _get_train_val_repo_split().

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding (unused, for API compatibility).
        num_mem_token: Number of memory token placeholders (unused, for API compatibility).

    Returns:
        TrajectoryDistillTransferDataset or None if validation is disabled.
    """
    data_cfg = cfg.data
    max_token_length = data_cfg.max_token_length
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_distill_recon_tokens")
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)

    if validation_split_num <= 0:
        return None

    _, val_repos = _get_train_val_repo_split(
        cache_dir=cache_dir,
        max_token_length=max_token_length,
        validation_split_num=validation_split_num,
        seed=seed,
    )

    if not val_repos:
        return None

    # Val dataset uses a fixed seed (no set_epoch reshuffling needed)
    val_dataset = TrajectoryDistillTransferDataset(
        model_path=model_path,
        cache_dir=cache_dir,
        max_token_length=max_token_length,
        seed=seed,
        repo_names=val_repos,
    )

    logger.info(
        f"[trajectory_distill_transfer] Val dataset: {len(val_dataset)} samples "
        f"from {len(val_repos)} repos"
    )

    return val_dataset


# ---------------------------------------------------------------------------
# Debug -- inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Create the dataset + collator, then call the generic
    ``debug_dataset`` utility to print aligned per-token tables.
    Also outputs per-repo trajectory statistics for both train and val sets.

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
        "max_token_length": data_cfg.max_token_length,
        "cache_dir": data_cfg.get("cache_dir", "cache/trajectory_distill_recon_tokens"),
        "num_samples_in_dataset": len(dataset),
    }

    dataset_name = data_cfg.get("name", "trajectory_distill_transfer")

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        metadata=metadata,
        num_samples=5,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )

    # ---- Append per-repo trajectory statistics to debug file ----
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_distill_recon_tokens")
    max_token_length = data_cfg.max_token_length
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)

    abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
    cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
    repo_cache_path = os.path.join(abs_cache_dir, REPO_CACHE_FILE)

    data = np.load(cache_path)
    lengths = data["lengths"]

    with open(repo_cache_path, "r") as f:
        repo_metadata = json.load(f)
    repo_list = repo_metadata["repos"]

    # Count total and valid (<=max_token_length) trajectories per repo
    repo_total_counts: Dict[str, int] = defaultdict(int)
    repo_valid_counts: Dict[str, int] = defaultdict(int)
    for i, repo in enumerate(repo_list):
        repo_total_counts[repo] += 1
        if lengths[i] <= max_token_length:
            repo_valid_counts[repo] += 1

    # Determine train/val split
    if validation_split_num > 0:
        train_repos, val_repos = _get_train_val_repo_split(
            cache_dir=cache_dir,
            max_token_length=max_token_length,
            validation_split_num=validation_split_num,
            seed=seed,
        )
    else:
        train_repos = list(repo_total_counts.keys())
        val_repos = []

    train_repo_set = set(train_repos)
    val_repo_set = set(val_repos)

    # Write to debug file (append mode)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{dataset_name}_debug.txt")

    with open(output_path, "a", encoding="utf-8") as f:
        sep = "=" * 120
        f.write(f"\n\n{sep}\n")
        f.write(f"  REPO STATISTICS (valid <= max_token_length / total)\n")
        f.write(f"  max_token_length = {max_token_length}\n")
        f.write(f"{sep}\n\n")

        # Train set
        f.write(f"  TRAIN SET: {len(train_repos)} repos\n")
        f.write(f"  {'─' * 100}\n")
        for repo in sorted(train_repos):
            valid = repo_valid_counts.get(repo, 0)
            total = repo_total_counts.get(repo, 0)
            f.write(f"    {repo}: {valid}/{total}\n")
        train_valid_total = sum(repo_valid_counts.get(r, 0) for r in train_repos)
        train_total_total = sum(repo_total_counts.get(r, 0) for r in train_repos)
        f.write(f"  {'─' * 100}\n")
        f.write(f"  Train total: {train_valid_total}/{train_total_total}\n\n")

        # Val set
        if val_repos:
            f.write(f"  VAL SET: {len(val_repos)} repos\n")
            f.write(f"  {'─' * 100}\n")
            for repo in sorted(val_repos):
                valid = repo_valid_counts.get(repo, 0)
                total = repo_total_counts.get(repo, 0)
                f.write(f"    {repo}: {valid}/{total}\n")
            val_valid_total = sum(repo_valid_counts.get(r, 0) for r in val_repos)
            val_total_total = sum(repo_total_counts.get(r, 0) for r in val_repos)
            f.write(f"  {'─' * 100}\n")
            f.write(f"  Val total: {val_valid_total}/{val_total_total}\n\n")
        else:
            f.write(f"  VAL SET: None (validation_split_num <= 0)\n\n")

        # Overall summary
        all_valid = sum(repo_valid_counts.values())
        all_total = sum(repo_total_counts.values())
        f.write(f"  OVERALL: {len(repo_total_counts)} repos, {all_valid}/{all_total} trajectories\n")
        f.write(f"{sep}\n")

    print(f"\n[debug] Repo statistics appended to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/pretrain/trajectory_distill_transfer.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Trajectory Distill Transfer dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Preprocess: tokenize and cache all trajectories")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/trajectory_distill_transfer.yaml",
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

    # Load tokenizer config
    _tokenizer_yaml = os.path.join("configs", "tokenizer", "origin.yaml")
    _tokenizer_cfg = OmegaConf.load(_tokenizer_yaml) if os.path.exists(_tokenizer_yaml) else OmegaConf.create({})

    cfg = OmegaConf.create({
        "data": data_cfg,
        "seed": {"dataset": _dataset_seed},
        "tokenizer": _tokenizer_cfg,
    })

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
        if model_path is None:
            print("ERROR: --model_path is required for --preprocess mode (or set in configs/model/*.yaml).")
            sys.exit(1)
        preprocess(cfg, model_path)
    elif args.debug:
        if model_path is None:
            print("ERROR: --model_path is required for --debug mode (or set in configs/model/*.yaml).")
            sys.exit(1)
        debug(cfg, model_path)
