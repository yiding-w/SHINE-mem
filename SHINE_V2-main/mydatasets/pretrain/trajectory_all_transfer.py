#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory All Transfer Dataset Module for SHINE_V2

Loads trajectories from SHINE_SWE_DISTILLATION + SHINE_SWE_OPENSOURCE,
tokenizes each trajectory using the chat template, caches the results,
and groups trajectories by repo for transfer learning.

Key difference from trajectory_distill_transfer:
    - Combines multiple data sources (DISTILLATION + OPENSOURCE)
    - OPENSOURCE trajectories are filtered by correctness="correct" (only correct ones
      are tokenized and used for training)
    - Debug statistics show valid/correct/total counts per repo

For each repo:
    - All correct trajectories belonging to that repo are collected and shuffled.
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
    Tokenizes all correct trajectories in parallel using multiprocessing, applies
    the chat template via the tokenizer, and stores results in a cache
    directory as .npz files. Uses file locking to avoid conflicts.

Dataset:
    Loads cached tokens and repo metadata, groups by repo, filters by
    max_token_length, and creates consecutive-pair samples.

Unified factory interface:
    create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
        -> (TrajectoryAllTransferDataset, TrajectoryAllTransferCollator)
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
from tqdm import tqdm

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
        args: (index, messages, tools) or (index, messages, tools, tmp_dir)

    Returns:
        (index, token_ids_array) on success, None on failure.
        If tmp_dir is provided, writes tokens to a file and returns (index, length).
    """
    if len(args) == 4:
        idx, messages, tools, tmp_dir = args
    else:
        idx, messages, tools = args
        tmp_dir = None

    try:
        global _worker_tokenizer
        # Fix data format issues from arrow storage
        fixed_messages = _fix_messages(messages)
        fixed_tools = _fix_tools(tools)

        # Apply chat template and tokenize in one step (avoids intermediate string alloc)
        token_ids = _worker_tokenizer.apply_chat_template(
            fixed_messages,
            tools=fixed_tools,
            add_generation_prompt=False,
            tokenize=True,
            preserve_thinking=True,
        )
        # Handle different return types from apply_chat_template
        if hasattr(token_ids, 'input_ids'):
            # BatchEncoding or dict-like
            token_ids = token_ids['input_ids']
        elif isinstance(token_ids, dict):
            token_ids = token_ids['input_ids']
        # token_ids should now be a list of ints
        token_arr = np.array(token_ids, dtype=np.int32)

        if tmp_dir is not None:
            # Write to temp file to avoid sending large arrays through pipe
            tmp_path = os.path.join(tmp_dir, f"{idx}.npy")
            np.save(tmp_path, token_arr)
            return (idx, len(token_arr))
        else:
            return (idx, token_arr)
    except Exception as e:
        logger.warning(f"[trajectory_all_transfer] Failed to tokenize trajectory {idx}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _load_all_shards_single(data_path: str) -> List[Dict[str, Any]]:
    """
    Load all arrow shard directories under a single data_path and return a list of rows.
    Each row is a dict with 'messages', 'tools', 'repo', and 'correctness' keys.
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
            has_correctness = "correctness" in col_dict
            has_tools = "tools" in col_dict
            has_repo = "repo" in col_dict
            num_rows = table.num_rows
            messages_col = col_dict["messages"]
            tools_col = col_dict["tools"] if has_tools else [None] * num_rows
            repo_col = col_dict["repo"] if has_repo else ["unknown"] * num_rows
            correctness_col = col_dict["correctness"] if has_correctness else [None] * num_rows
            for i in range(num_rows):
                all_rows.append({
                    "messages": messages_col[i],
                    "tools": tools_col[i],
                    "repo": repo_col[i],
                    "correctness": correctness_col[i],
                    "source": data_path,
                })
            f.close()

    logger.info(f"[trajectory_all_transfer] Loaded {len(all_rows)} trajectories from {len(shard_dirs)} shards in {data_path}")
    return all_rows


def _load_all_shards(data_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load all arrow shard directories from multiple data paths and return a combined list.
    Each row is a dict with 'messages', 'tools', 'repo', 'correctness', and 'source' keys.
    """
    all_rows = []
    for data_path in data_paths:
        rows = _load_all_shards_single(data_path)
        all_rows.extend(rows)
    logger.info(f"[trajectory_all_transfer] Total loaded: {len(all_rows)} trajectories from {len(data_paths)} sources")
    return all_rows


# ---------------------------------------------------------------------------
# Preprocessing: tokenize all trajectories and cache
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Tokenize all correct trajectories in parallel and cache the results.
    Uses file locking to avoid conflicts.

    For DISTILLATION data: all trajectories are considered correct (no correctness field).
    For OPENSOURCE data: only trajectories with correctness="correct" are tokenized.

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
    dataset_name = data_cfg.get("name", "trajectory_all_transfer")
    data_paths = list(data_cfg.get("data_paths", ["data/SHINE_SWE_DISTILLATION", "data/SHINE_SWE_OPENSOURCE"]))
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_tokens")
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
    _print(f"  PREPROCESS -- Dataset: {dataset_name}")
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

        # Load all trajectories from all data paths
        # STREAMING MODE: process one arrow file at a time to avoid loading 165GB+ into memory
        # We save lightweight metadata (repo, correctness) per trajectory to a JSONL file,
        # and only keep messages/tools in memory for the current shard being tokenized.
        _print(f"  Loading trajectories from: {data_paths} (streaming mode)")

        # Prepare tokenizer config for workers
        tokenizer_cfg_dict = OmegaConf.to_container(cfg.tokenizer, resolve=True)

        # Create temp directory for storing tokenized results (avoids pipe OOM)
        import shutil
        tmp_dir = os.path.join(abs_cache_dir, "_tmp_tokens")
        os.makedirs(tmp_dir, exist_ok=True)

        # Trajectory metadata file: stores (repo, correctness, source) per global index
        # This is lightweight (~10MB for 100k trajectories) and enables resume without reloading data
        meta_path = os.path.join(tmp_dir, "_trajectory_meta.json")

        # Resume support: check which trajectories are already tokenized
        already_done = set()
        _print(f"  Scanning temp dir for existing results...")
        t_scan = time.time()
        for entry in os.scandir(tmp_dir):
            if entry.name.endswith(".npy") and entry.is_file(follow_symlinks=False):
                try:
                    idx = int(entry.name[:-4])
                    already_done.add(idx)
                except ValueError:
                    pass
        _print(f"  Found {len(already_done)} existing files in {time.time() - t_scan:.1f}s")

        # Load or build trajectory metadata
        # The metadata maps global_index -> {repo, correctness}
        # On first run, we stream through all shards to build it
        # On resume, we load it from disk
        trajectory_meta = None  # list of dicts: [{repo, correctness, source}, ...]
        if os.path.exists(meta_path):
            _print(f"  Loading trajectory metadata from cache...")
            with open(meta_path, "r") as f:
                trajectory_meta = json.load(f)
            _print(f"  Loaded metadata for {len(trajectory_meta)} trajectories")

        if trajectory_meta is None:
            # First run: stream through all shards to build metadata
            _print(f"  Building trajectory metadata (first run)...")
            import pyarrow as pa
            import pyarrow.ipc as ipc

            trajectory_meta = []  # [{repo, correctness, source}, ...]
            shard_info_list = []  # [(arrow_path, global_start, num_rows), ...]
            for data_path in data_paths:
                abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)
                shard_dirs = sorted([
                    d for d in os.listdir(abs_data_path)
                    if os.path.isdir(os.path.join(abs_data_path, d)) and not d.startswith(".")
                ])
                for shard_dir in shard_dirs:
                    shard_path = os.path.join(abs_data_path, shard_dir)
                    arrow_files = sorted([
                        af for af in os.listdir(shard_path) if af.endswith(".arrow")
                    ])
                    for arrow_file in arrow_files:
                        arrow_path = os.path.join(shard_path, arrow_file)
                        f = pa.memory_map(arrow_path, "r")
                        reader = ipc.open_stream(f)
                        table = reader.read_all()
                        # Only read lightweight columns (repo, correctness) - skip messages/tools
                        schema_names = set(table.schema.names)
                        has_correctness = "correctness" in schema_names
                        has_repo = "repo" in schema_names
                        num_rows = table.num_rows
                        # Record shard info for later use
                        shard_info_list.append((arrow_path, len(trajectory_meta), num_rows))
                        # Select only needed columns to avoid loading messages/tools into memory
                        if has_repo:
                            repo_col = table.column("repo").to_pylist()
                        else:
                            repo_col = ["unknown"] * num_rows
                        if has_correctness:
                            correctness_col = table.column("correctness").to_pylist()
                        else:
                            correctness_col = [None] * num_rows
                        for i in range(num_rows):
                            trajectory_meta.append({
                                "repo": repo_col[i],
                                "correctness": correctness_col[i],
                                "source": f"{data_path}/{shard_dir}",
                            })
                        f.close()
                        del table
                _print(f"    {data_path}: scanned, total so far: {len(trajectory_meta)}")

            # Save metadata for future resumes
            with open(meta_path, "w") as f:
                json.dump(trajectory_meta, f)
            # Save shard info for fast tokenization resume
            shard_info_path = os.path.join(tmp_dir, "_shard_info.json")
            with open(shard_info_path, "w") as f:
                json.dump(shard_info_list, f)
            _print(f"  Saved trajectory metadata ({len(trajectory_meta)} entries, {len(shard_info_list)} shards)")

        num_total = len(trajectory_meta)
        _print(f"  Total trajectories: {num_total}")

        # Determine which global indices are "correct" (correctness == "correct")
        # Also skip trajectories with repo=None (cannot be grouped by repo for training)
        correct_indices = []  # global indices of correct trajectories
        num_skipped_incorrect = 0
        num_skipped_unverified = 0
        num_skipped_no_repo = 0
        num_correctness_unknown = 0
        for global_idx, meta in enumerate(trajectory_meta):
            repo = meta.get("repo")
            correctness = meta.get("correctness")
            # Skip trajectories with no repo (cannot be used in repo-grouped training)
            if repo is None:
                num_skipped_no_repo += 1
                continue
            if correctness == "correct":
                correct_indices.append(global_idx)
            elif correctness is None:
                # No correctness field (e.g., DISTILLATION data) - treat as correct
                correct_indices.append(global_idx)
                num_correctness_unknown += 1
            elif correctness == "unverified":
                num_skipped_unverified += 1
            else:
                # "incorrect" or other values
                num_skipped_incorrect += 1

        num_correct = len(correct_indices)
        _print(f"  Correct (correctness='correct' or unknown): {num_correct}")
        _print(f"  Skipped incorrect: {num_skipped_incorrect}")
        _print(f"  Skipped unverified: {num_skipped_unverified}")
        _print(f"  Skipped no repo (repo=None): {num_skipped_no_repo}")
        if num_correctness_unknown > 0:
            _print(f"  NOTE: {num_correctness_unknown} trajectories have no 'correctness' field (treated as correct)")

        # Build reverse map: global_index -> correct_position
        global_to_pos = {gidx: pos for pos, gidx in enumerate(correct_indices)}

        # Determine which correct positions still need tokenization
        # Tmp files are named by correct_position (0..num_correct-1) for backward compatibility
        all_correct_positions = set(range(num_correct))
        to_process_positions = sorted(all_correct_positions - already_done)
        num_to_process = len(to_process_positions)
        num_already_done = len(already_done & all_correct_positions)

        if num_already_done > 0:
            _print(f"  Resuming: {num_already_done} correct trajectories already tokenized, "
                   f"{num_to_process} remaining.")
        else:
            _print(f"  Tokenizing {num_correct} correct trajectories with {num_workers} workers...")
        _print(f"  (Using temp dir for resume support: {tmp_dir})")
        t0 = time.time()

        # Length index for fast resume (keyed by correct_position)
        length_index_path = os.path.join(tmp_dir, "_lengths.json")
        length_index = {}
        if os.path.exists(length_index_path):
            try:
                with open(length_index_path, "r") as f:
                    length_index = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                length_index = {}

        # Results array: indexed by correct_position (0..num_correct-1)
        results = [None] * num_correct
        num_failed = 0

        # Pre-fill results for already-done items using length index
        for pos in range(num_correct):
            if pos in length_index:
                results[pos] = length_index[pos]
            elif pos in already_done:
                # Length not in index, will need to reload
                pass

        # For items in already_done but not in length_index, load lengths in parallel
        need_reload_positions = [pos for pos in (already_done & all_correct_positions) if pos not in length_index]
        if need_reload_positions:
            _print(f"  Loading lengths for {len(need_reload_positions)} files not in index...")

            def _get_length(pos):
                tmp_path_l = os.path.join(tmp_dir, f"{pos}.npy")
                try:
                    arr = np.load(tmp_path_l)
                    return (pos, len(arr))
                except Exception:
                    return (pos, None)

            from concurrent.futures import ThreadPoolExecutor as _TPE
            with _TPE(max_workers=min(32, num_workers)) as executor:
                for pos, length in tqdm(
                    executor.map(_get_length, need_reload_positions),
                    total=len(need_reload_positions),
                    desc="  Loading lengths",
                    unit="file",
                ):
                    if length is not None:
                        length_index[pos] = length
                        results[pos] = length
                    else:
                        # Corrupted file, add to reprocess list
                        tmp_path_c = os.path.join(tmp_dir, f"{pos}.npy")
                        if os.path.exists(tmp_path_c):
                            os.remove(tmp_path_c)
                        to_process_positions.append(pos)
                        num_to_process += 1

        # --- Streaming tokenization: process one arrow file at a time ---
        if num_to_process > 0:
            _print(f"  Streaming tokenization: {num_to_process} trajectories to process...")
            import pyarrow as pa
            import pyarrow.ipc as ipc
            from bisect import bisect_left, bisect_right

            # Build sorted list of global indices that need processing (for fast range queries)
            to_process_global_sorted = sorted(correct_indices[pos] for pos in to_process_positions)

            # Load shard info (saved during metadata building)
            shard_info_path = os.path.join(tmp_dir, "_shard_info.json")
            if os.path.exists(shard_info_path):
                with open(shard_info_path, "r") as f:
                    shard_info_list = json.load(f)
                # Convert lists back to tuples
                shard_info_list = [(s[0], s[1], s[2]) for s in shard_info_list]
            else:
                # Fallback: re-scan (should not happen if metadata was built correctly)
                _print(f"  WARNING: shard_info not found, re-scanning...")
                shard_info_list = []
                _global_counter = 0
                for data_path in data_paths:
                    abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)
                    shard_dirs = sorted([
                        d for d in os.listdir(abs_data_path)
                        if os.path.isdir(os.path.join(abs_data_path, d)) and not d.startswith(".")
                    ])
                    for shard_dir in shard_dirs:
                        shard_path = os.path.join(abs_data_path, shard_dir)
                        arrow_files = sorted([
                            af for af in os.listdir(shard_path) if af.endswith(".arrow")
                        ])
                        for arrow_file in arrow_files:
                            arrow_path = os.path.join(shard_path, arrow_file)
                            f_arrow = pa.memory_map(arrow_path, "r")
                            reader = ipc.open_stream(f_arrow)
                            num_rows = 0
                            for batch in reader:
                                num_rows += batch.num_rows
                            f_arrow.close()
                            shard_info_list.append((arrow_path, _global_counter, num_rows))
                            _global_counter += num_rows
                # Save for next time
                with open(shard_info_path, "w") as f:
                    json.dump(shard_info_list, f)

            # Filter to only shards that have work items (using bisect for O(log n) lookup)
            shards_with_work = []
            for arrow_path, shard_start, num_rows in shard_info_list:
                shard_end = shard_start + num_rows
                # Find if any element in to_process_global_sorted falls in [shard_start, shard_end)
                left_idx = bisect_left(to_process_global_sorted, shard_start)
                if left_idx < len(to_process_global_sorted) and to_process_global_sorted[left_idx] < shard_end:
                    shards_with_work.append((arrow_path, shard_start, num_rows))

            _print(f"  {len(shards_with_work)}/{len(shard_info_list)} shards contain work items")

            # Convert to set for O(1) membership test during iteration
            to_process_global_set = set(to_process_global_sorted)
            del to_process_global_sorted

            # Generator that only opens shards known to have work
            # NOTE: We process shard-by-shard to bound memory usage.
            # Python's imap_unordered eagerly consumes generators via an internal thread,
            # so a generator alone doesn't help with memory. Instead, we submit batches
            # per shard and collect results before moving to the next shard.
            with mp.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(model_path, tokenizer_cfg_dict),
            ) as pool:
                pbar = tqdm(total=num_to_process, desc="  Tokenizing", unit="traj")

                for shard_idx, (arrow_path, shard_start, num_rows) in enumerate(shards_with_work):
                    # Find which global indices in this shard need processing
                    shard_end = shard_start + num_rows
                    shard_needs_set = set()
                    for gidx in range(shard_start, shard_end):
                        if gidx in to_process_global_set:
                            shard_needs_set.add(gidx)

                    if not shard_needs_set:
                        continue

                    # Read shard batch-by-batch to avoid loading entire table into memory
                    f_a = pa.memory_map(arrow_path, "r")
                    reader = ipc.open_stream(f_a)
                    batch_offset = shard_start  # global index of first row in current batch
                    shard_work = []

                    for batch in reader:
                        batch_size = batch.num_rows
                        batch_end = batch_offset + batch_size

                        # Check if any needed indices fall in this batch
                        needs_in_batch = []
                        for gidx in range(batch_offset, batch_end):
                            if gidx in shard_needs_set:
                                needs_in_batch.append(gidx)

                        if needs_in_batch:
                            # Convert only this batch to python
                            col_names = batch.schema.names
                            has_tools = "tools" in col_names
                            messages_col = batch.column("messages").to_pylist()
                            tools_col = batch.column("tools").to_pylist() if has_tools else None

                            for gidx in needs_in_batch:
                                local_idx = gidx - batch_offset
                                pos = global_to_pos[gidx]
                                tool_val = tools_col[local_idx] if tools_col is not None else None
                                shard_work.append((pos, messages_col[local_idx], tool_val, tmp_dir))

                            del messages_col, tools_col

                        batch_offset = batch_end

                        # Submit in chunks to avoid accumulating too much in memory
                        if len(shard_work) >= 1024:
                            for result in pool.imap_unordered(_tokenize_trajectory, shard_work, chunksize=8):
                                if result is not None:
                                    pos, length = result
                                    length_index[pos] = length
                                    results[pos] = length
                                else:
                                    num_failed += 1
                                pbar.update(1)
                            shard_work = []

                    f_a.close()

                    # Process remaining work items for this shard
                    if shard_work:
                        for result in pool.imap_unordered(_tokenize_trajectory, shard_work, chunksize=8):
                            if result is not None:
                                pos, length = result
                                length_index[pos] = length
                                results[pos] = length
                            else:
                                num_failed += 1
                            pbar.update(1)
                        del shard_work

                pbar.close()

            # Save updated length index for fast future resumes
            with open(length_index_path, "w") as f:
                json.dump(length_index, f)
            _print(f"  Saved length index ({len(length_index)} entries)")
        else:
            _print(f"  All trajectories already tokenized (resume complete).")

        # Filter out failed tokenizations
        # valid_positions are correct_positions (0..num_correct-1) that succeeded
        valid_positions = [pos for pos, r in enumerate(results) if r is not None]
        valid_lengths = [results[pos] for pos in valid_positions]
        _print(f"  Tokenized {len(valid_positions)} trajectories in {time.time() - t0:.1f}s")
        if num_failed > 0:
            _print(f"  WARNING: {num_failed} trajectories failed tokenization")

        if len(valid_positions) == 0:
            _print(f"  ERROR: All trajectories failed tokenization. Cannot create cache.")
            _print(sep)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(buf.getvalue())
            buf.close()
            raise RuntimeError("All trajectories failed tokenization.")

        # Build concatenated arrays by reading from temp files
        _print(f"  Assembling token cache from {len(valid_positions)} temp files...")
        lengths = np.array(valid_lengths, dtype=np.int32)
        offsets = np.zeros(len(valid_positions) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths.astype(np.int64))

        total_tokens = int(offsets[-1])
        _print(f"  Total tokens: {total_tokens:,}")
        _print(f"  Token length stats: min={lengths.min()}, max={lengths.max()}, "
               f"mean={lengths.mean():.0f}, median={np.median(lengths):.0f}")

        # Concatenate all tokens from temp files (parallelized I/O with progress bar)
        # Tmp files are named by correct_position (0..num_correct-1)
        tokens = np.empty(total_tokens, dtype=np.int32)

        def _load_and_place(args):
            """Load a single .npy file and place it into the pre-allocated tokens array."""
            assembly_idx, correct_pos, start, end = args
            tmp_path = os.path.join(tmp_dir, f"{correct_pos}.npy")
            token_arr = np.load(tmp_path)
            tokens[start:end] = token_arr

        # Use threads for parallel I/O (GIL is released during np.load file I/O)
        from concurrent.futures import ThreadPoolExecutor
        load_args = [
            (i, correct_pos, int(offsets[i]), int(offsets[i + 1]))
            for i, correct_pos in enumerate(valid_positions)
        ]
        with ThreadPoolExecutor(max_workers=min(32, num_workers)) as executor:
            list(tqdm(
                executor.map(_load_and_place, load_args),
                total=len(load_args),
                desc="  Assembling",
                unit="file",
            ))

        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _print(f"  Cleaned up temp directory.")

        # Save cache
        cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
        _print(f"  Saving token cache to: {cache_path}")
        np.savez(
            cache_path,
            tokens=tokens,
            offsets=offsets,
            lengths=lengths,
        )

        # Build repo list matching the cache order (valid tokenized correct trajectories)
        # Use trajectory_meta (lightweight, already in memory) to get repo names
        repo_list = []
        source_list = []  # source (data_path/shard_dir) for each trajectory in cache order
        for correct_pos in valid_positions:
            global_idx = correct_indices[correct_pos]
            repo_list.append(trajectory_meta[global_idx]["repo"])
            source_list.append(trajectory_meta[global_idx].get("source", "unknown"))

        assert len(repo_list) == len(valid_positions), (
            f"Repo list length ({len(repo_list)}) != valid results ({len(valid_positions)})"
        )

        # Build per-source statistics
        # source_stats_cache: {source_key: {total, correct}} where correct = in cache
        source_stats_cache: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for entry in trajectory_meta:
            src = entry.get("source", "unknown")
            source_stats_cache[src]["total"] += 1
        for src in source_list:
            source_stats_cache[src]["correct"] += 1

        # Save repo metadata
        repo_metadata = {
            "repos": repo_list,
            "sources": source_list,
            "source_stats": dict(source_stats_cache),
            "num_trajectories": len(valid_positions),
            "num_total_raw": num_total,
            "num_correct": num_correct,
            "num_skipped_incorrect": num_skipped_incorrect,
            "num_skipped_unverified": num_skipped_unverified,
            "num_skipped_no_repo": num_skipped_no_repo,
            "num_failed_tokenization": num_failed,
        }
        with open(repo_cache_path, "w", encoding="utf-8") as f:
            json.dump(repo_metadata, f)
        _print(f"  Saved repo metadata ({len(set(repo_list))} unique repos, {len(source_stats_cache)} sources)")

        # Save manifest
        manifest = {
            "num_trajectories": len(valid_positions),
            "total_tokens": total_tokens,
            "num_failed": num_failed,
            "num_total_raw": num_total,
            "num_correct": num_correct,
            "num_skipped_incorrect": num_skipped_incorrect,
            "num_skipped_unverified": num_skipped_unverified,
            "num_skipped_no_repo": num_skipped_no_repo,
            "data_paths": data_paths,
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

        _print(f"  Cache verified and saved.")
        _print(f"  Preprocessing complete.")
        _print(sep)

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[trajectory_all_transfer] Preprocessing complete. Output: {output_path}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryAllTransferDataset(BaseDataset):
    """
    Dataset that loads pre-tokenized trajectories from cache, groups them
    by repo, and creates consecutive-pair samples for transfer learning.

    For each repo, trajectories are shuffled, then each trajectory i is paired
    with trajectory (i+1) % num_in_repo:
        - context_ids = trajectory[i] tokens
        - conversation_ids = trajectory[(i+1) % n] tokens (the "next" one)
        - labels = same as conversation_ids (padding masked as -100)

    Chunk-based ordering (continuous_repo_num):
        When continuous_repo_num > 0, each repo's samples are split into
        chunks with sizes sampled from Poisson(N), where N is the target
        average. Repos with fewer than N trajectories are kept intact (not
        split). Tiny tail chunks (< N/2) are merged into the previous chunk.
        All chunks from all repos are then globally shuffled, so chunks from
        the same repo may be separated by chunks from other repos. Only
        within a single chunk are trajectories from the same repo guaranteed
        to be contiguous.
        When continuous_repo_num <= 0, the original behavior is preserved
        (entire repo kept contiguous).

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
        continuous_repo_num: int = -1,
    ):
        super().__init__(model_path)
        self.max_token_length = max_token_length
        self._base_seed = seed
        self._continuous_repo_num = continuous_repo_num

        # Load cache
        abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
        verified_path = os.path.join(abs_cache_dir, VERIFIED_FILE)
        if not os.path.exists(verified_path):
            raise RuntimeError(
                f"Cache not verified at {abs_cache_dir}. "
                f"Run preprocessing first: python mydatasets/pretrain/trajectory_all_transfer.py --preprocess"
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
            f"[trajectory_all_transfer] Created {len(self.samples)} samples "
            f"from {len(self._repo_to_valid_indices)} repos "
            f"(max_token_length={max_token_length}, "
            f"continuous_repo_num={continuous_repo_num})"
        )

    def _build_samples(self, seed: int):
        """
        Build the samples list by shuffling repo order and within-repo
        trajectory order using the given seed.

        When continuous_repo_num > 0, long repos are split into chunks whose
        sizes are sampled from Poisson(N) (where N is continuous_repo_num).
        This means the average consecutive run of the same repo is
        approximately N, with natural variation (variance = N). Repos with
        fewer than N trajectories are NOT split. Tiny tail chunks (< N/2)
        are merged into the preceding chunk to avoid degenerate small blocks.
        All chunks (from all repos) are then globally shuffled, so different
        repos' chunks are interleaved.

        Args:
            seed: Random seed for this epoch's shuffling.
        """
        rng = random.Random(seed)
        self.samples = []

        # Shuffle repo order
        repo_names = list(self._repo_to_valid_indices.keys())
        rng.shuffle(repo_names)

        # Collect chunks: each chunk is a list of samples from the same repo
        all_chunks: List[List[Dict[str, Any]]] = []

        for repo_name in repo_names:
            valid_indices = list(self._repo_to_valid_indices[repo_name])

            if len(valid_indices) < 2:
                # Need at least 2 trajectories to form pairs
                # If only 1, use it as both context and conversation
                if len(valid_indices) == 1:
                    i = valid_indices[0]
                    token_ids = self._trajectory_tokens[i]
                    all_chunks.append([{
                        "context_token_ids": token_ids,
                        "conversation_token_ids": token_ids,
                        "repo": repo_name,
                    }])
                continue

            # Shuffle within repo
            rng.shuffle(valid_indices)

            # Create consecutive pairs for the entire repo
            repo_samples = []
            for pos in range(len(valid_indices)):
                ctx_idx = valid_indices[pos]
                conv_idx = valid_indices[(pos + 1) % len(valid_indices)]
                repo_samples.append({
                    "context_token_ids": self._trajectory_tokens[ctx_idx],
                    "conversation_token_ids": self._trajectory_tokens[conv_idx],
                    "repo": repo_name,
                })

            # Split into chunks if continuous_repo_num is set
            # Chunk sizes are sampled from Poisson(N) to produce natural variation
            # around the target average. Only repos longer than N are split.
            # Tiny tail chunks (< 0.5*N) are merged into the previous chunk.
            target = self._continuous_repo_num
            if target > 0 and len(repo_samples) > target:
                np_rng = np.random.RandomState(rng.randint(0, 2**31))
                start = 0
                while start < len(repo_samples):
                    remaining = len(repo_samples) - start
                    # If remaining is small enough, just take it all
                    if remaining <= target:
                        all_chunks.append(repo_samples[start:])
                        break
                    cs = max(1, int(np_rng.poisson(target)))
                    end = min(start + cs, len(repo_samples))
                    # Merge tiny tail: if what's left after this chunk is < 0.5*N,
                    # extend this chunk to include the tail
                    leftover = len(repo_samples) - end
                    if 0 < leftover < max(1, target // 2):
                        end = len(repo_samples)
                    all_chunks.append(repo_samples[start:end])
                    start = end
            else:
                # Keep entire repo as one chunk (original behavior or repo is small)
                all_chunks.append(repo_samples)

        # Globally shuffle all chunks
        rng.shuffle(all_chunks)

        # Flatten chunks into the final samples list
        for chunk in all_chunks:
            self.samples.extend(chunk)

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


class TrajectoryAllTransferCollator(BaseCollator):
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
        f"[trajectory_all_transfer] Repo-level split: "
        f"train={len(train_repos)} repos, val={len(val_repos)} repos "
        f"(val_samples={val_total}, target={validation_split_num})"
    )

    return train_repos, val_repos


# ---------------------------------------------------------------------------
# Factory function -- unified interface
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a TrajectoryAllTransferDataset and its collator.

    If validation_split_num is set, the train dataset only contains train repos
    (val repos are excluded). The val dataset is created separately via
    create_val_dataset().

    Parameters from cfg.data (trajectory_all_transfer.yaml):
        - max_token_length:   maximum token length for filtering trajectories
        - cache_dir:          directory containing preprocessed cache
        - validation_split_num: target number of validation samples (repo-level split)

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (TrajectoryAllTransferDataset, TrajectoryAllTransferCollator)
    """
    data_cfg = cfg.data

    max_token_length = data_cfg.max_token_length
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_tokens")
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)
    continuous_repo_num = data_cfg.get("continuous_repo_num", -1)

    logger.info(
        f"[trajectory_all_transfer] Creating dataset: "
        f"max_token_length={max_token_length}, cache_dir={cache_dir}, "
        f"continuous_repo_num={continuous_repo_num}"
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

    dataset = TrajectoryAllTransferDataset(
        model_path=model_path,
        cache_dir=cache_dir,
        max_token_length=max_token_length,
        seed=seed,
        repo_names=train_repo_names,
        continuous_repo_num=continuous_repo_num,
    )

    collator = TrajectoryAllTransferCollator(
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
        TrajectoryAllTransferDataset or None if validation is disabled.
    """
    data_cfg = cfg.data
    max_token_length = data_cfg.max_token_length
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_tokens")
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)
    continuous_repo_num = data_cfg.get("continuous_repo_num", -1)

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
    val_dataset = TrajectoryAllTransferDataset(
        model_path=model_path,
        cache_dir=cache_dir,
        max_token_length=max_token_length,
        seed=seed,
        repo_names=val_repos,
        continuous_repo_num=continuous_repo_num,
    )

    logger.info(
        f"[trajectory_all_transfer] Val dataset: {len(val_dataset)} samples "
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

    Statistics format per repo: valid/correct/total
        - valid: trajectories with length <= max_token_length (among correct ones)
        - correct: trajectories with correctness="correct" (successfully tokenized)
        - total: all trajectories in the raw data for this repo

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
        "cache_dir": data_cfg.get("cache_dir", "cache/trajectory_all_transfer_tokens"),
        "num_samples_in_dataset": len(dataset),
    }

    dataset_name = data_cfg.get("name", "trajectory_all_transfer")

    # --- Reorder samples so that debug_dataset shows both openai-style
    #     (has <tool_call> token 248058) and xml-style (no <tool_call> token) ---
    # Uniformly sample from different parts of the dataset (not just the first few)
    TOOL_CALL_TOKEN_ID = 248058
    NUM_PER_STYLE = 5  # 5 openai + 5 xml = 10 total
    openai_indices = []
    xml_indices = []
    total_samples = len(dataset.samples)
    # Sample at regular intervals to get diverse sources
    step = max(1, total_samples // 200)  # check ~200 samples spread across dataset
    for idx in range(0, total_samples, step):
        ctx_tokens = dataset.samples[idx]["context_token_ids"]
        has_tool_call = (ctx_tokens == TOOL_CALL_TOKEN_ID).any().item()
        if has_tool_call:
            if len(openai_indices) < NUM_PER_STYLE:
                openai_indices.append(idx)
        else:
            if len(xml_indices) < NUM_PER_STYLE:
                xml_indices.append(idx)
        if len(openai_indices) >= NUM_PER_STYLE and len(xml_indices) >= NUM_PER_STYLE:
            break

    # Reorder: put selected openai + xml samples at the front
    selected_indices = openai_indices + xml_indices
    if selected_indices:
        original_samples = dataset.samples
        reordered = [original_samples[i] for i in selected_indices]
        remaining = [s for idx, s in enumerate(original_samples) if idx not in set(selected_indices)]
        dataset.samples = reordered + remaining

    num_debug_samples = len(selected_indices) if selected_indices else 5
    print(f"[debug] Selected {len(openai_indices)} openai-style + {len(xml_indices)} xml-style samples for table visualization (step={step})")

    # Add sample layout info to metadata
    metadata["sample_layout"] = (
        f"Sample 0~{len(openai_indices)-1} = OpenAI-style (has <tool_call> token), "
        f"Sample {len(openai_indices)}~{len(selected_indices)-1} = XML-style (no <tool_call> token)"
    ) if selected_indices else "default (first N samples)"

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        metadata=metadata,
        num_samples=num_debug_samples,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )

    # ---- Append per-repo trajectory statistics to debug file ----
    # We need to reload raw data to get total counts (including incorrect/unverified)
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_tokens")
    max_token_length = data_cfg.max_token_length
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)
    data_paths = list(data_cfg.get("data_paths", ["data/SHINE_SWE_DISTILLATION", "data/SHINE_SWE_OPENSOURCE"]))

    abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
    cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
    repo_cache_path = os.path.join(abs_cache_dir, REPO_CACHE_FILE)

    data = np.load(cache_path)
    lengths = data["lengths"]

    with open(repo_cache_path, "r") as f:
        repo_metadata = json.load(f)
    repo_list = repo_metadata["repos"]
    source_list_cached = repo_metadata.get("sources")  # may be None if old cache
    source_stats_cached = repo_metadata.get("source_stats")  # may be None if old cache

    # Count valid (<=max_token_length) and correct (in cache) per repo
    repo_correct_counts: Dict[str, int] = defaultdict(int)  # correct = in cache (correctness=correct & tokenized)
    repo_valid_counts: Dict[str, int] = defaultdict(int)    # valid = correct & <= max_token_length
    for i, repo in enumerate(repo_list):
        repo_correct_counts[repo] += 1
        if lengths[i] <= max_token_length:
            repo_valid_counts[repo] += 1

    # ---- Per-source statistics ----
    # Best case: repo_metadata.json has "sources" list (new cache format)
    # Otherwise: fall back to _trajectory_meta.json or arrow scanning
    source_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0, "valid": 0})

    if source_list_cached is not None and len(source_list_cached) == len(repo_list):
        # Fast path: use sources list from repo_metadata.json
        print("\n[debug] Using per-source info from repo_metadata.json (fast path)")

        def _normalize_source_key(src: str) -> str:
            """Normalize source key: DISTILLATION as one group, OPENSOURCE per subset."""
            if "DISTILLATION" in src:
                return "SHINE_SWE_DISTILLATION"
            elif "OPENSOURCE" in src:
                return os.path.basename(src)
            else:
                return src

        # Compute valid per source directly from lengths + source_list
        source_correct_counts_local: Dict[str, int] = defaultdict(int)
        for i, src_raw in enumerate(source_list_cached):
            src = _normalize_source_key(src_raw)
            source_correct_counts_local[src] += 1
            if lengths[i] <= max_token_length:
                source_stats[src]["valid"] += 1
        # Fill correct counts
        for src, cnt in source_correct_counts_local.items():
            source_stats[src]["correct"] = cnt
        # Fill total counts from source_stats_cached if available
        if source_stats_cached:
            # Aggregate source_stats_cached using same normalization
            for src_raw, stats in source_stats_cached.items():
                src = _normalize_source_key(src_raw)
                source_stats[src]["total"] += stats.get("total", 0)
        else:
            # No total info available, set total = correct as fallback
            for src in source_stats:
                if source_stats[src]["total"] == 0:
                    source_stats[src]["total"] = source_stats[src]["correct"]

    # Load raw data to get total counts per repo (including incorrect/unverified)
    # Also collect per-source statistics if not already done
    tmp_dir = os.path.join(abs_cache_dir, "_tmp_tokens")
    meta_path = os.path.join(tmp_dir, "_trajectory_meta.json")
    repo_total_counts: Dict[str, int] = defaultdict(int)
    repos_with_unknown_correctness: set = set()
    source_correct_order: List[str] = []  # source name for each correct trajectory in cache order

    if os.path.exists(meta_path):
        print("\n[debug] Loading trajectory metadata from preprocess cache...")
        with open(meta_path, "r") as f:
            trajectory_meta = json.load(f)
        print(f"  Loaded {len(trajectory_meta)} entries from cache (fast path)")
        for entry in trajectory_meta:
            repo = entry["repo"]
            repo_total_counts[repo] += 1
            if entry["correctness"] is None:
                repos_with_unknown_correctness.add(repo)
            # Collect per-source stats only if not already done via source_list_cached
            if source_list_cached is None:
                source_raw = entry.get("source", "unknown")
                # Normalize source key: DISTILLATION as one group, OPENSOURCE per subset
                if "DISTILLATION" in source_raw:
                    source_key = "SHINE_SWE_DISTILLATION"
                elif "OPENSOURCE" in source_raw:
                    # Extract subset name (last path component)
                    source_key = os.path.basename(source_raw)
                else:
                    source_key = source_raw
                source_stats[source_key]["total"] += 1
                correctness = entry.get("correctness")
                if correctness == "correct" or correctness is None:
                    if repo is not None:
                        source_stats[source_key]["correct"] += 1
                        source_correct_order.append(source_key)
    else:
        # Slow path: stream through arrow files but only load lightweight columns
        print("\n[debug] Loading raw data for total trajectory counts (lightweight scan)...")
        import pyarrow as pa
        import pyarrow.ipc as ipc
        from tqdm import tqdm

        # First, collect all arrow file paths with their source info
        arrow_paths = []  # (arrow_path, data_path_basename, shard_dir)
        for data_path in data_paths:
            abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)
            data_path_base = os.path.basename(data_path)
            shard_dirs = sorted([
                d for d in os.listdir(abs_data_path)
                if os.path.isdir(os.path.join(abs_data_path, d)) and not d.startswith(".")
            ])
            for shard_dir in shard_dirs:
                shard_path = os.path.join(abs_data_path, shard_dir)
                arrow_files = sorted([
                    af for af in os.listdir(shard_path) if af.endswith(".arrow")
                ])
                for arrow_file in arrow_files:
                    arrow_paths.append((os.path.join(shard_path, arrow_file), data_path_base, shard_dir))

        # Stream through arrow files, only reading repo + correctness columns
        for arrow_path, data_path_base, shard_dir in tqdm(arrow_paths, desc="  Scanning shards", unit="shard"):
            # Source key: for DISTILLATION use parent name, for OPENSOURCE use shard_dir
            source_key = shard_dir if data_path_base == "SHINE_SWE_OPENSOURCE" else data_path_base
            f = pa.memory_map(arrow_path, "r")
            reader = ipc.open_stream(f)
            table = reader.read_all()
            schema_names = set(table.schema.names)
            num_rows = table.num_rows
            if "repo" in schema_names:
                repo_col = table.column("repo").to_pylist()
            else:
                repo_col = ["unknown"] * num_rows
            if "correctness" in schema_names:
                correctness_col = table.column("correctness").to_pylist()
            else:
                correctness_col = [None] * num_rows
            for i in range(num_rows):
                repo_total_counts[repo_col[i]] += 1
                if correctness_col[i] is None:
                    repos_with_unknown_correctness.add(repo_col[i])
                # Collect per-source stats only if not already done via source_list_cached
                if source_list_cached is None:
                    source_stats[source_key]["total"] += 1
                    correctness = correctness_col[i]
                    if correctness == "correct" or correctness is None:
                        if repo_col[i] is not None:
                            source_stats[source_key]["correct"] += 1
                            source_correct_order.append(source_key)
            f.close()
            del table

    # Compute per-source "valid" counts using lengths array and source_correct_order
    # (only needed if source_list_cached was not available)
    if source_list_cached is None:
        if source_correct_order and len(source_correct_order) == len(lengths):
            for i, src in enumerate(source_correct_order):
                if lengths[i] <= max_token_length:
                    source_stats[src]["valid"] += 1
        elif source_correct_order:
            # Length mismatch - try to use conversion_stats.json order as fallback
            print(f"  [warn] source_correct_order length ({len(source_correct_order)}) != "
                  f"lengths array ({len(lengths)}), using conversion_stats order fallback")
            # Infer valid counts from the order of sources in data_paths
            shard_correct_counts = []  # [(source_key, num_correct_rows)]
            for data_path in data_paths:
                abs_data_path = data_path if os.path.isabs(data_path) else os.path.join(_project_root, data_path)
                data_path_base = os.path.basename(data_path)
                shard_dirs = sorted([
                    d for d in os.listdir(abs_data_path)
                    if os.path.isdir(os.path.join(abs_data_path, d)) and not d.startswith(".")
                ])
                for shard_dir in shard_dirs:
                    source_key = shard_dir if data_path_base == "SHINE_SWE_OPENSOURCE" else data_path_base
                    if source_key in source_stats and source_stats[source_key]["correct"] > 0:
                        shard_correct_counts.append((source_key, source_stats[source_key]["correct"]))
            # Map cache positions to sources
            pos = 0
            for source_key, num_correct in shard_correct_counts:
                end = min(pos + num_correct, len(lengths))
                for i in range(pos, end):
                    if lengths[i] <= max_token_length:
                        source_stats[source_key]["valid"] += 1
                pos = end
        else:
            print("  [info] No per-source order info available, valid counts may be incomplete")

    # Determine train/val split
    if validation_split_num > 0:
        train_repos, val_repos = _get_train_val_repo_split(
            cache_dir=cache_dir,
            max_token_length=max_token_length,
            validation_split_num=validation_split_num,
            seed=seed,
        )
    else:
        train_repos = list(repo_correct_counts.keys())
        val_repos = []

    # Write to debug file (append mode)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{dataset_name}_debug.txt")

    with open(output_path, "a", encoding="utf-8") as f:
        sep = "=" * 120
        # ---- Per-source (dataset) statistics ----
        f.write(f"\n\n{sep}\n")
        f.write(f"  SOURCE STATISTICS (per data source / subset)\n")
        f.write(f"  max_token_length = {max_token_length}\n")
        f.write(f"  Format: valid (correct & <=max_token_length) / correct / total\n")
        f.write(f"{sep}\n\n")

        f.write(f"  {'Source':<70s} | {'Valid':>10s} | {'Correct':>10s} | {'Total':>10s} | {'%Valid/Corr':>12s} | {'%Corr/Total':>12s}\n")
        f.write(f"  {'-'*70}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}\n")

        total_valid_all = 0
        total_correct_all = 0
        total_total_all = 0
        for src in sorted(source_stats.keys()):
            s = source_stats[src]
            v, c, t = s["valid"], s["correct"], s["total"]
            total_valid_all += v
            total_correct_all += c
            total_total_all += t
            pct_vc = f"{100.0*v/c:.1f}%" if c > 0 else "N/A"
            pct_ct = f"{100.0*c/t:.1f}%" if t > 0 else "N/A"
            f.write(f"  {src:<70s} | {v:>10d} | {c:>10d} | {t:>10d} | {pct_vc:>12s} | {pct_ct:>12s}\n")

        f.write(f"  {'-'*70}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}\n")
        pct_vc_all = f"{100.0*total_valid_all/total_correct_all:.1f}%" if total_correct_all > 0 else "N/A"
        pct_ct_all = f"{100.0*total_correct_all/total_total_all:.1f}%" if total_total_all > 0 else "N/A"
        f.write(f"  {'TOTAL':<70s} | {total_valid_all:>10d} | {total_correct_all:>10d} | {total_total_all:>10d} | {pct_vc_all:>12s} | {pct_ct_all:>12s}\n")
        f.write(f"\n{sep}\n")

        # ---- Per-repo statistics ----
        f.write(f"\n\n{sep}\n")
        f.write(f"  REPO STATISTICS (valid <= max_token_length / correct / total)\n")
        f.write(f"  max_token_length = {max_token_length}\n")
        f.write(f"  data_paths = {data_paths}\n")
        f.write(f"{sep}\n\n")

        # Warnings for repos with unknown correctness field
        if repos_with_unknown_correctness:
            f.write(f"  WARNING: The following repos have trajectories with no 'correctness' field\n")
            f.write(f"     (treated as correct, but correctness cannot be verified):\n")
            for repo in sorted(repos_with_unknown_correctness, key=lambda x: x or ""):
                f.write(f"       - {repo}\n")
            f.write(f"\n")

        # Train set
        f.write(f"  TRAIN SET: {len(train_repos)} repos\n")
        f.write(f"  {'_' * 100}\n")
        for repo in sorted(train_repos, key=lambda x: x or ""):
            valid = repo_valid_counts.get(repo, 0)
            correct = repo_correct_counts.get(repo, 0)
            total = repo_total_counts.get(repo, 0)
            warning = " (correctness unknown)" if repo in repos_with_unknown_correctness else ""
            f.write(f"    {repo}: {valid}/{correct}/{total}{warning}\n")
        train_valid_total = sum(repo_valid_counts.get(r, 0) for r in train_repos)
        train_correct_total = sum(repo_correct_counts.get(r, 0) for r in train_repos)
        train_total_total = sum(repo_total_counts.get(r, 0) for r in train_repos)
        f.write(f"  {'_' * 100}\n")
        f.write(f"  Train total: {train_valid_total}/{train_correct_total}/{train_total_total}\n\n")

        # Val set
        if val_repos:
            f.write(f"  VAL SET: {len(val_repos)} repos\n")
            f.write(f"  {'_' * 100}\n")
            for repo in sorted(val_repos, key=lambda x: x or ""):
                valid = repo_valid_counts.get(repo, 0)
                correct = repo_correct_counts.get(repo, 0)
                total = repo_total_counts.get(repo, 0)
                warning = " (correctness unknown)" if repo in repos_with_unknown_correctness else ""
                f.write(f"    {repo}: {valid}/{correct}/{total}{warning}\n")
            val_valid_total = sum(repo_valid_counts.get(r, 0) for r in val_repos)
            val_correct_total = sum(repo_correct_counts.get(r, 0) for r in val_repos)
            val_total_total = sum(repo_total_counts.get(r, 0) for r in val_repos)
            f.write(f"  {'_' * 100}\n")
            f.write(f"  Val total: {val_valid_total}/{val_correct_total}/{val_total_total}\n\n")
        else:
            f.write(f"  VAL SET: None (validation_split_num <= 0)\n\n")

        # Overall summary
        all_valid = sum(repo_valid_counts.values())
        all_correct = sum(repo_correct_counts.values())
        all_total = sum(repo_total_counts.values())
        f.write(f"  OVERALL: {len(repo_total_counts)} repos, {all_valid}/{all_correct}/{all_total} trajectories\n")
        f.write(f"  (format: valid<=max_token_length / correct / total_raw)\n\n")

        # Warning about repo=None trajectories (excluded from training)
        num_none_repo_total = repo_total_counts.get(None, 0)
        num_none_repo_correct = repo_correct_counts.get(None, 0)
        num_none_repo_valid = repo_valid_counts.get(None, 0)
        if num_none_repo_total > 0:
            f.write(f"  {'!' * 100}\n")
            f.write(f"  WARNING: {num_none_repo_total} trajectories have repo=None and are EXCLUDED from training.\n")
            f.write(f"    - Total with repo=None: {num_none_repo_total}\n")
            f.write(f"    - Of which correctness='correct': {num_none_repo_correct}\n")
            f.write(f"    - Of which would be valid (<=max_token_length): {num_none_repo_valid}\n")
            f.write(f"    These trajectories cannot be grouped by repo and are skipped during preprocessing.\n")
            f.write(f"  {'!' * 100}\n")

        f.write(f"{sep}\n")

    print(f"\n[debug] Repo statistics appended to: {output_path}")

    # ---- Append reasoning_content statistics ----
    # Efficiently count trajectories with/without reasoning_content
    # using parallel arrow scanning.
    _append_reasoning_content_statistics(
        data_paths=data_paths,
        output_path=output_path,
    )

    # ---- Append tool_call style statistics and samples ----
    # Count and display OpenAI-style (structured tool_calls field) vs
    # XML-style (tool_call tags embedded in content) tool calls.
    _append_tool_call_style_stats(
        data_paths=data_paths,
        output_path=output_path,
        num_per_style=3,
    )


def _scan_arrow_for_tool_call_styles(arrow_path: str, sub: str) -> Dict[str, Any]:
    """
    Worker function: scan a single arrow file and return tool_call style counts
    plus up to 1 sample per style. Runs in a separate process for parallelism.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    counts = {
        "num_openai_style": 0,
        "num_xml_style": 0,
        "num_both_style": 0,
        "num_no_tool_call": 0,
        "num_total_scanned": 0,
        "msg_openai_style": 0,
        "msg_xml_style": 0,
    }
    openai_sample = None
    xml_sample = None
    both_sample = None
    no_tc_sample = None

    f = pa.memory_map(arrow_path, "r")
    try:
        reader = ipc.open_stream(f)
        for batch in reader:
            names = batch.schema.names
            n = batch.num_rows
            if "correctness" in names:
                corr = batch.column("correctness").to_pylist()
            else:
                corr = [None] * n
            msgs_col = batch.column("messages").to_pylist() if "messages" in names else [None] * n
            repos = batch.column("repo").to_pylist() if "repo" in names else [None] * n
            iids = (batch.column("instance_id").to_pylist()
                    if "instance_id" in names else [None] * n)
            for i in range(n):
                c = corr[i]
                if c is not None and c != "correct":
                    continue
                m = msgs_col[i]
                if m is None:
                    continue
                counts["num_total_scanned"] += 1
                # Classify trajectory
                has_openai = False
                has_xml = False
                for msg in m:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "assistant":
                        continue
                    tc = msg.get("tool_calls")
                    if tc and len(tc) > 0:
                        has_real_tc = any(
                            (isinstance(t, dict) and t.get("function"))
                            for t in tc
                        )
                        if has_real_tc:
                            has_openai = True
                            counts["msg_openai_style"] += 1
                    content = msg.get("content") or ""
                    if "<function=" in content:
                        if not tc or len(tc) == 0:
                            has_xml = True
                            counts["msg_xml_style"] += 1
                if has_openai and has_xml:
                    counts["num_both_style"] += 1
                elif has_openai:
                    counts["num_openai_style"] += 1
                elif has_xml:
                    counts["num_xml_style"] += 1
                else:
                    counts["num_no_tool_call"] += 1
                # Collect one sample per style per file (lightweight)
                if has_openai and not has_xml and openai_sample is None:
                    try:
                        fixed = _fix_messages(m)
                    except Exception:
                        fixed = m
                    openai_sample = {
                        "source_dataset": sub,
                        "repo": repos[i],
                        "instance_id": iids[i],
                        "messages": fixed,
                    }
                if has_xml and not has_openai and xml_sample is None:
                    try:
                        fixed = _fix_messages(m)
                    except Exception:
                        fixed = m
                    xml_sample = {
                        "source_dataset": sub,
                        "repo": repos[i],
                        "instance_id": iids[i],
                        "messages": fixed,
                    }
                if has_openai and has_xml and both_sample is None:
                    try:
                        fixed = _fix_messages(m)
                    except Exception:
                        fixed = m
                    both_sample = {
                        "source_dataset": sub,
                        "repo": repos[i],
                        "instance_id": iids[i],
                        "messages": fixed,
                    }
                if not has_openai and not has_xml and no_tc_sample is None:
                    try:
                        fixed = _fix_messages(m)
                    except Exception:
                        fixed = m
                    no_tc_sample = {
                        "source_dataset": sub,
                        "repo": repos[i],
                        "instance_id": iids[i],
                        "messages": fixed,
                    }
    finally:
        f.close()

    return {
        "counts": counts,
        "openai_sample": openai_sample,
        "xml_sample": xml_sample,
        "both_sample": both_sample,
        "no_tc_sample": no_tc_sample,
    }


def _append_tool_call_style_stats(
    data_paths: List[str],
    output_path: str,
    num_per_style: int = 3,
) -> None:
    """
    Scan arrow shards to count and sample two styles of tool calls:
      - OpenAI style: assistant messages with a non-empty ``tool_calls`` field
      - XML style: assistant messages with ``<tool_call>`` in content but no
        ``tool_calls`` field (or empty tool_calls)

    Counts are reported at the trajectory level (a trajectory is classified
    as OpenAI-style if ANY assistant message has a non-empty tool_calls field,
    XML-style if ANY assistant message has <tool_call> in content without
    tool_calls field, or both if it has both patterns).

    Uses multiprocessing to scan arrow files in parallel for speed.

    Appends statistics and sample trajectories to the debug file.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\n[debug] Scanning arrow shards for tool_call style statistics ...")

    # Collect all arrow file paths with their shard names
    arrow_tasks: List[Tuple[str, str]] = []  # (arrow_path, sub_name)
    project_root = _project_root
    for data_path in data_paths:
        abs_path = data_path if os.path.isabs(data_path) else os.path.join(project_root, data_path)
        if not os.path.isdir(abs_path):
            continue
        shard_dirs = sorted([
            d for d in os.listdir(abs_path)
            if os.path.isdir(os.path.join(abs_path, d)) and not d.startswith(".")
        ])
        for sub in shard_dirs:
            sub_path = os.path.join(abs_path, sub)
            arrow_files = sorted(
                os.path.join(sub_path, x)
                for x in os.listdir(sub_path) if x.endswith(".arrow")
            )
            for arrow_path in arrow_files:
                arrow_tasks.append((arrow_path, sub))

    print(f"  Found {len(arrow_tasks)} arrow files to scan, using parallel workers...")

    # Counters
    num_openai_style = 0
    num_xml_style = 0
    num_both_style = 0
    num_no_tool_call = 0
    num_total_scanned = 0
    msg_openai_style = 0
    msg_xml_style = 0

    # Samples
    openai_samples: List[Dict[str, Any]] = []
    xml_samples: List[Dict[str, Any]] = []
    both_samples: List[Dict[str, Any]] = []
    no_tc_samples: List[Dict[str, Any]] = []

    # Use ProcessPoolExecutor for true parallelism (bypasses GIL for pyarrow I/O)
    num_workers = min(16, len(arrow_tasks), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_scan_arrow_for_tool_call_styles, path, sub): (path, sub)
            for path, sub in arrow_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Scanning tool_call styles", unit="file"):
            result = future.result()
            c = result["counts"]
            num_openai_style += c["num_openai_style"]
            num_xml_style += c["num_xml_style"]
            num_both_style += c["num_both_style"]
            num_no_tool_call += c["num_no_tool_call"]
            num_total_scanned += c["num_total_scanned"]
            msg_openai_style += c["msg_openai_style"]
            msg_xml_style += c["msg_xml_style"]
            if result["openai_sample"] and len(openai_samples) < num_per_style:
                openai_samples.append(result["openai_sample"])
            if result["xml_sample"] and len(xml_samples) < num_per_style:
                xml_samples.append(result["xml_sample"])
            if result["both_sample"] and len(both_samples) < num_per_style:
                both_samples.append(result["both_sample"])
            if result["no_tc_sample"] and len(no_tc_samples) < num_per_style:
                no_tc_samples.append(result["no_tc_sample"])

    # ---- Write results to debug file ----
    def _shorten(s, limit: int = 600) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) <= limit:
            return s
        return s[:limit] + f" ... [truncated, total len={len(s)}]"

    def _render_msg_tc(msg: Dict) -> str:
        """Render a message focusing on tool_call details."""
        role = msg.get("role", "?")
        parts = [f"role={role}"]
        if "name" in msg:
            parts.append(f"name={msg['name']}")
        if "tool_call_id" in msg:
            parts.append(f"tool_call_id={msg['tool_call_id']}")
        content = msg.get("content")
        if content not in (None, ""):
            parts.append(f"content={_shorten(content, 500)!r}")
        if msg.get("tool_calls"):
            tc_brief = []
            for tc in msg["tool_calls"][:3]:
                fn = (tc or {}).get("function", {}) or {}
                args_str = _shorten(str(fn.get('arguments', '')), 300)
                tc_brief.append(f"{fn.get('name')}({args_str})")
            if len(msg["tool_calls"]) > 3:
                tc_brief.append(f"... +{len(msg['tool_calls'])-3} more")
            parts.append(f"tool_calls=[{', '.join(tc_brief)}]")
        return " | ".join(parts)

    with open(output_path, "a", encoding="utf-8") as f:
        sep = "=" * 120
        f.write(f"\n\n{sep}\n")
        f.write(f"  TOOL CALL STYLE STATISTICS (correct/unknown-correctness trajectories only)\n")
        f.write(f"{sep}\n\n")

        f.write(f"  Total trajectories scanned: {num_total_scanned}\n")
        f.write(f"  OpenAI-style only (structured tool_calls field, renders <tool_call> token after tokenize): {num_openai_style}\n")
        f.write(f"  XML-style only (<function= in content, no structured tool_calls field):                  {num_xml_style}\n")
        f.write(f"  Both styles in same trajectory:                  {num_both_style}\n")
        f.write(f"  No tool calls at all:                            {num_no_tool_call}\n")
        f.write(f"\n")
        f.write(f"  Per-message counts:\n")
        f.write(f"    Assistant messages with OpenAI-style tool_calls: {msg_openai_style}\n")
        f.write(f"    Assistant messages with XML-style tool_calls:    {msg_xml_style}\n")
        f.write(f"\n")

        # OpenAI-style samples
        f.write(f"  --- OPENAI-STYLE ONLY SAMPLES: {len(openai_samples)} trajectory(s) ---\n\n")
        if not openai_samples:
            f.write(f"    (none found)\n\n")
        for k, entry in enumerate(openai_samples):
            f.write(f"  [OpenAI #{k}] source_dataset={entry['source_dataset']!r} "
                    f"repo={entry['repo']!r} instance_id={entry['instance_id']!r}\n")
            msgs = entry["messages"] or []
            f.write(f"    num_messages={len(msgs)}\n")
            # Show assistant messages with tool_calls (up to 4)
            shown = 0
            for j, msg in enumerate(msgs):
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    f.write(f"      msg[{j:>3}] {_render_msg_tc(msg)}\n")
                    shown += 1
                    if shown >= 4:
                        remaining = sum(1 for mm in msgs[j+1:] if mm.get("role") == "assistant" and mm.get("tool_calls"))
                        if remaining > 0:
                            f.write(f"      ... +{remaining} more assistant messages with tool_calls\n")
                        break
            f.write("\n")

        # XML-style samples
        f.write(f"  --- XML-STYLE ONLY SAMPLES: {len(xml_samples)} trajectory(s) ---\n\n")
        if not xml_samples:
            f.write(f"    (none found)\n\n")
        for k, entry in enumerate(xml_samples):
            f.write(f"  [XML #{k}] source_dataset={entry['source_dataset']!r} "
                    f"repo={entry['repo']!r} instance_id={entry['instance_id']!r}\n")
            msgs = entry["messages"] or []
            f.write(f"    num_messages={len(msgs)}\n")
            # Show assistant messages with <function= in content (up to 4)
            shown = 0
            for j, msg in enumerate(msgs):
                if msg.get("role") == "assistant":
                    content = msg.get("content") or ""
                    tc = msg.get("tool_calls")
                    if "<function=" in content and (not tc or len(tc) == 0):
                        f.write(f"      msg[{j:>3}] {_render_msg_tc(msg)}\n")
                        shown += 1
                        if shown >= 4:
                            remaining = sum(
                                1 for mm in msgs[j+1:]
                                if mm.get("role") == "assistant"
                                and "<function=" in (mm.get("content") or "")
                                and (not mm.get("tool_calls") or len(mm.get("tool_calls")) == 0)
                            )
                            if remaining > 0:
                                f.write(f"      ... +{remaining} more assistant messages with XML tool_calls\n")
                            break
            f.write("\n")

        # Both-style samples
        f.write(f"  --- BOTH STYLES IN SAME TRAJECTORY SAMPLES: {len(both_samples)} trajectory(s) ---\n\n")
        if not both_samples:
            f.write(f"    (none found)\n\n")
        for k, entry in enumerate(both_samples):
            f.write(f"  [Both #{k}] source_dataset={entry['source_dataset']!r} "
                    f"repo={entry['repo']!r} instance_id={entry['instance_id']!r}\n")
            msgs = entry["messages"] or []
            f.write(f"    num_messages={len(msgs)}\n")
            # Show assistant messages with either style (up to 6)
            shown = 0
            for j, msg in enumerate(msgs):
                if msg.get("role") == "assistant":
                    tc = msg.get("tool_calls")
                    content = msg.get("content") or ""
                    has_openai_tc = tc and len(tc) > 0 and any(
                        isinstance(t, dict) and t.get("function") for t in tc
                    )
                    has_xml_tc = "<function=" in content and (not tc or len(tc) == 0)
                    if has_openai_tc or has_xml_tc:
                        style_tag = "[OpenAI]" if has_openai_tc else "[XML]"
                        f.write(f"      msg[{j:>3}] {style_tag} {_render_msg_tc(msg)}\n")
                        shown += 1
                        if shown >= 6:
                            f.write(f"      ... (more messages omitted)\n")
                            break
            f.write("\n")

        # No-tool-call samples
        f.write(f"  --- NO TOOL CALLS AT ALL SAMPLES: {len(no_tc_samples)} trajectory(s) ---\n\n")
        if not no_tc_samples:
            f.write(f"    (none found)\n\n")
        for k, entry in enumerate(no_tc_samples):
            f.write(f"  [NoTC #{k}] source_dataset={entry['source_dataset']!r} "
                    f"repo={entry['repo']!r} instance_id={entry['instance_id']!r}\n")
            msgs = entry["messages"] or []
            f.write(f"    num_messages={len(msgs)}\n")
            # Show first few assistant messages to illustrate the format
            shown = 0
            for j, msg in enumerate(msgs):
                if msg.get("role") == "assistant":
                    f.write(f"      msg[{j:>3}] {_render_msg_tc(msg)}\n")
                    shown += 1
                    if shown >= 4:
                        remaining = sum(1 for mm in msgs[j+1:] if mm.get("role") == "assistant")
                        if remaining > 0:
                            f.write(f"      ... +{remaining} more assistant messages\n")
                        break
            f.write("\n")

        f.write(f"{sep}\n")

    print(f"[debug] tool_call style statistics appended to: {output_path}")


def _scan_reasoning_content_worker(arrow_path: str, sub: str) -> Dict[str, Any]:
    """
    Worker function: scan a single arrow file and count trajectories
    with/without non-empty reasoning_content (correct only).
    Returns counts per source_dataset.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    # Per-source counts: {source_dataset: {"with_rc": N, "without_rc": N}}
    source_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"with_rc": 0, "without_rc": 0})

    f = pa.memory_map(arrow_path, "r")
    try:
        reader = ipc.open_stream(f)
        schema = reader.schema

        # Quick check: does messages schema have reasoning_content field?
        msg_type = schema.field("messages").type
        has_rc_field = "reasoning_content" in msg_type.value_type.names

        for batch in reader:
            names = batch.schema.names
            n = batch.num_rows
            if "correctness" in names:
                corr = batch.column("correctness").to_pylist()
            else:
                corr = ["correct"] * n

            src_list = (batch.column("source_dataset").to_pylist()
                        if "source_dataset" in names else [sub] * n)

            if not has_rc_field:
                # No reasoning_content field at all -> all are "without"
                for i in range(n):
                    if corr[i] != "correct":
                        continue
                    source_counts[src_list[i] or sub]["without_rc"] += 1
            else:
                # Need to check messages for non-empty reasoning_content
                msgs_list = batch.column("messages").to_pylist()
                for i in range(n):
                    if corr[i] != "correct":
                        continue
                    msgs = msgs_list[i]
                    if msgs is None:
                        source_counts[src_list[i] or sub]["without_rc"] += 1
                        continue
                    has_rc = False
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue
                        rc = m.get("reasoning_content")
                        if rc is not None and rc != "":
                            has_rc = True
                            break
                    src_key = src_list[i] or sub
                    if has_rc:
                        source_counts[src_key]["with_rc"] += 1
                    else:
                        source_counts[src_key]["without_rc"] += 1
    finally:
        f.close()

    return dict(source_counts)


def _append_reasoning_content_statistics(
    data_paths: List[str],
    output_path: str,
) -> None:
    """
    Efficiently scan all arrow shards in parallel to count how many
    correct trajectories have non-empty reasoning_content vs not.
    Outputs per-source statistics to the debug file.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\n[debug] Scanning arrow shards for reasoning_content statistics ...")

    # Collect all arrow file tasks
    arrow_tasks: List[Tuple[str, str]] = []  # (arrow_path, sub_dir_name)
    project_root = _project_root
    for data_path in data_paths:
        abs_path = data_path if os.path.isabs(data_path) else os.path.join(project_root, data_path)
        if not os.path.isdir(abs_path):
            continue
        for sub in sorted(os.listdir(abs_path)):
            sub_path = os.path.join(abs_path, sub)
            if not os.path.isdir(sub_path) or sub.startswith("."):
                continue
            for fname in sorted(os.listdir(sub_path)):
                if fname.endswith(".arrow"):
                    arrow_tasks.append((os.path.join(sub_path, fname), sub))

    if not arrow_tasks:
        print("[debug] No arrow files found for reasoning_content statistics.")
        return

    # Parallel scan
    # Per-source aggregated counts
    source_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"with_rc": 0, "without_rc": 0})

    num_workers = min(16, len(arrow_tasks), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_scan_reasoning_content_worker, path, sub): (path, sub)
            for path, sub in arrow_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Scanning reasoning_content", unit="file"):
            result = future.result()
            for src, counts in result.items():
                source_totals[src]["with_rc"] += counts["with_rc"]
                source_totals[src]["without_rc"] += counts["without_rc"]

    # Compute totals
    total_with_rc = sum(v["with_rc"] for v in source_totals.values())
    total_without_rc = sum(v["without_rc"] for v in source_totals.values())
    total_correct = total_with_rc + total_without_rc

    # ---- Write statistics to debug file ----
    with open(output_path, "a", encoding="utf-8") as f:
        sep = "=" * 120
        f.write(f"\n\n{sep}\n")
        f.write(f"  REASONING_CONTENT STATISTICS (correct trajectories only)\n")
        f.write(f"{sep}\n\n")

        f.write(f"  Total correct trajectories scanned: {total_correct}\n")
        f.write(f"  With non-empty reasoning_content:   {total_with_rc} "
                f"({100.0 * total_with_rc / total_correct:.1f}%)\n" if total_correct > 0 else
                f"  With non-empty reasoning_content:   {total_with_rc}\n")
        f.write(f"  Without reasoning_content:          {total_without_rc} "
                f"({100.0 * total_without_rc / total_correct:.1f}%)\n\n" if total_correct > 0 else
                f"  Without reasoning_content:          {total_without_rc}\n\n")

        # Per-source breakdown
        f.write(f"  Per-source breakdown:\n")
        f.write(f"  {'Source Dataset':<60s} | {'With RC':>10s} | {'Without RC':>10s} | {'Total':>10s} | {'% With RC':>10s}\n")
        f.write(f"  {'-'*60}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}\n")
        for src in sorted(source_totals.keys()):
            w = source_totals[src]["with_rc"]
            wo = source_totals[src]["without_rc"]
            t = w + wo
            pct = f"{100.0 * w / t:.1f}%" if t > 0 else "N/A"
            f.write(f"  {src:<60s} | {w:>10d} | {wo:>10d} | {t:>10d} | {pct:>10s}\n")

        f.write(f"\n{sep}\n")

    print(f"[debug] reasoning_content statistics appended to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/pretrain/trajectory_all_transfer.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Trajectory All Transfer dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Preprocess: tokenize and cache all correct trajectories")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/trajectory_all_transfer.yaml",
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
