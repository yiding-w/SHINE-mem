#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory All Transfer V2 Dataset Module for SHINE_V2

Loads trajectories from SHINE_SWE_DISTILLATION_v2 + SHINE_SWE_OPENSOURCE_v2,
tokenizes each trajectory using the chat template, caches the results,
and groups trajectories by repo for transfer learning.

Key differences from trajectory_all_transfer:
    - Uses v2 data paths (DISTILLATION_v2 + OPENSOURCE_v2)
    - Requires issue_content_hash field on every trajectory (raises error if missing)
    - Supports max_same_issue: limits trajectories with the same issue_content_hash
      by randomly sampling (seeded by seed.dataset) when count exceeds the limit
      (does NOT require re-preprocessing when changed)
    - Supports subset: filters to specific sub-datasets
      (does NOT require re-preprocessing when changed)
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

CACHE_FILE = "all_trajectories.npz"
REPO_CACHE_FILE = "repo_metadata.json"
MANIFEST_FILE = "manifest.json"
VERIFIED_FILE = "VERIFIED"
LOCK_FILE = ".cache_lock"

VALID_OPENSOURCE_SUBSETS = [
    "CoderForge-Preview",
    "Kwai-Klear-SWE-smith-mini-swe-agent-plus-trajectories-66k",
    "MEnvData-SWE-Trajectory",
    "Nemotron-SFT-SWE-v3",
    "Nemotron-SWE-v1",
    "Open-SWE-Traces",
    "OpenSWE-Trajectory",
    "Scale-SWE-Distilled-DeepSeek-v3.2",
    "Scale-SWE-Distilled-DeepSeek-v4-Pro-High-41k",
    "SWE-Factory-Kimi-K2-2.8K",
    "SWE-Factory-Kimi-K2-RS",
    "SWE-Gym-OpenHands-SFT-Trajectories",
    "SWE-Hero-openhands-trajectories",
    "SWE-Lego-Real-Data",
    "SWE-Lego-Synthetic-Data",
    "SWE-rebench-openhands-trajectories",
    "SWE-smith-trajectories",
    "SWE-Zero-openhands-trajectories",
]
VALID_SUBSET_NAMES = VALID_OPENSOURCE_SUBSETS + ["distillation"]


# ---------------------------------------------------------------------------
# Parallel tokenization worker (reused from trajectory_all_transfer)
# ---------------------------------------------------------------------------

def _init_worker(tokenizer_path: str, tokenizer_cfg_dict: dict):
    """Pool initializer: load tokenizer once per worker."""
    global _worker_tokenizer
    from omegaconf import OmegaConf
    _worker_tokenizer = create_tokenizer(
        tokenizer_path, tokenizer_cfg=OmegaConf.create(tokenizer_cfg_dict)
    )


def _fix_tools(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Fix tool definitions loaded from arrow files."""
    if not tools:
        return None
    fixed_tools = []
    for tool in tools:
        tool = dict(tool)
        if "function" in tool and tool["function"]:
            func = dict(tool["function"])
            if "parameters" in func and func["parameters"]:
                params = dict(func["parameters"])
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
    """Fix messages loaded from arrow files."""
    fixed_messages = []
    for msg in messages:
        msg = dict(msg)
        for key in ("name", "tool_call_id"):
            if key in msg and msg[key] == "":
                del msg[key]
        if "reasoning_content" in msg and msg["reasoning_content"] == "":
            del msg["reasoning_content"]
        if msg.get("role") == "assistant" and msg.get("content") == "" and msg.get("tool_calls"):
            del msg["content"]
        if "tool_calls" in msg:
            if not msg["tool_calls"]:
                del msg["tool_calls"]
            else:
                fixed_tc = []
                for tc in msg["tool_calls"]:
                    tc = dict(tc)
                    if "function" in tc and tc["function"]:
                        func = dict(tc["function"])
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
    """Tokenize a single trajectory (messages + tools) using the chat template."""
    if len(args) == 4:
        idx, messages, tools, tmp_dir = args
    else:
        idx, messages, tools = args
        tmp_dir = None

    try:
        global _worker_tokenizer
        fixed_messages = _fix_messages(messages)
        fixed_tools = _fix_tools(tools)

        token_ids = _worker_tokenizer.apply_chat_template(
            fixed_messages,
            tools=fixed_tools,
            add_generation_prompt=False,
            tokenize=True,
            preserve_thinking=True,
        )
        if hasattr(token_ids, 'input_ids'):
            token_ids = token_ids['input_ids']
        elif isinstance(token_ids, dict):
            token_ids = token_ids['input_ids']
        token_arr = np.array(token_ids, dtype=np.int32)

        if tmp_dir is not None:
            tmp_path = os.path.join(tmp_dir, f"{idx}.npy")
            np.save(tmp_path, token_arr)
            return (idx, len(token_arr))
        else:
            return (idx, token_arr)
    except Exception as e:
        logger.warning(f"[trajectory_all_transfer_v2] Failed to tokenize trajectory {idx}: {e}")
        return None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Tokenize all correct trajectories in parallel and cache the results.

    Every trajectory MUST have an issue_content_hash field. If any trajectory
    is missing this field, a RuntimeError is raised.
    """
    import io
    import shutil
    from omegaconf import OmegaConf

    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "trajectory_all_transfer_v2")
    data_paths = list(data_cfg.get("data_paths", [
        "data/SHINE_SWE_DISTILLATION_v2", "data/SHINE_SWE_OPENSOURCE_v2"
    ]))
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_v2_tokens")
    num_workers = data_cfg.get("preprocess_workers", 32)

    abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
    os.makedirs(abs_cache_dir, exist_ok=True)

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

    lock_path = os.path.join(abs_cache_dir, LOCK_FILE)
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        verified_path = os.path.join(abs_cache_dir, VERIFIED_FILE)
        repo_cache_path = os.path.join(abs_cache_dir, REPO_CACHE_FILE)

        if os.path.exists(verified_path) and os.path.exists(repo_cache_path):
            _print(f"  Cache already verified at: {abs_cache_dir}")
            _print(f"  Skipping preprocessing.")
            _print(sep)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(buf.getvalue())
            buf.close()
            return

        tokenizer_cfg_dict = OmegaConf.to_container(cfg.tokenizer, resolve=True)

        tmp_dir = os.path.join(abs_cache_dir, "_tmp_tokens")
        os.makedirs(tmp_dir, exist_ok=True)

        meta_path = os.path.join(tmp_dir, "_trajectory_meta.json")

        # Resume support
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
        trajectory_meta = None
        if os.path.exists(meta_path):
            _print(f"  Loading trajectory metadata from cache...")
            with open(meta_path, "r") as f:
                trajectory_meta = json.load(f)
            _print(f"  Loaded metadata for {len(trajectory_meta)} trajectories")

        if trajectory_meta is None:
            _print(f"  Building trajectory metadata (first run)...")
            import pyarrow as pa
            import pyarrow.ipc as ipc

            trajectory_meta = []
            shard_info_list = []
            num_missing_hash = 0

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
                        schema_names = set(table.schema.names)
                        num_rows = table.num_rows

                        shard_info_list.append((arrow_path, len(trajectory_meta), num_rows))

                        has_correctness = "correctness" in schema_names
                        has_repo = "repo" in schema_names
                        has_hash = "issue_content_hash" in schema_names

                        # Determine source_dataset name for subset filtering
                        is_distillation = "DISTILLATION" in data_path
                        if is_distillation:
                            source_dataset = "distillation"
                        else:
                            source_dataset = shard_dir.replace(".openai", "")

                        # Opensource data MUST have correctness field; distillation data may lack it (treated as correct)
                        if not has_correctness and not is_distillation:
                            f.close()
                            del table
                            raise RuntimeError(
                                f"[trajectory_all_transfer_v2] Opensource data file '{arrow_path}' "
                                f"is missing the 'correctness' field. All opensource data MUST have "
                                f"this field. Only distillation data is allowed to omit it."
                            )

                        repo_col = table.column("repo").to_pylist() if has_repo else ["unknown"] * num_rows
                        correctness_col = table.column("correctness").to_pylist() if has_correctness else [None] * num_rows
                        hash_col = table.column("issue_content_hash").to_pylist() if has_hash else [None] * num_rows

                        for i in range(num_rows):
                            issue_hash = hash_col[i]
                            if issue_hash is None or issue_hash == "":
                                num_missing_hash += 1
                            trajectory_meta.append({
                                "repo": repo_col[i],
                                "correctness": correctness_col[i],
                                "issue_content_hash": issue_hash,
                                "source_dataset": source_dataset,
                                "source": f"{data_path}/{shard_dir}",
                            })
                        f.close()
                        del table
                _print(f"    {data_path}: scanned, total so far: {len(trajectory_meta)}")

            # Validate: every trajectory must have issue_content_hash
            if num_missing_hash > 0:
                raise RuntimeError(
                    f"[trajectory_all_transfer_v2] {num_missing_hash} trajectories are missing "
                    f"'issue_content_hash' field. All trajectories MUST have this field."
                )

            with open(meta_path, "w") as f:
                json.dump(trajectory_meta, f)
            shard_info_path = os.path.join(tmp_dir, "_shard_info.json")
            with open(shard_info_path, "w") as f:
                json.dump(shard_info_list, f)
            _print(f"  Saved trajectory metadata ({len(trajectory_meta)} entries)")

        num_total = len(trajectory_meta)
        _print(f"  Total trajectories: {num_total}")

        # Determine correct indices
        correct_indices = []
        num_skipped_incorrect = 0
        num_skipped_unverified = 0
        num_skipped_no_repo = 0
        num_correctness_unknown = 0
        for global_idx, meta in enumerate(trajectory_meta):
            repo = meta.get("repo")
            correctness = meta.get("correctness")
            if repo is None:
                num_skipped_no_repo += 1
                continue
            if correctness == "correct":
                correct_indices.append(global_idx)
            elif correctness is None:
                correct_indices.append(global_idx)
                num_correctness_unknown += 1
            elif correctness == "unverified":
                num_skipped_unverified += 1
            else:
                num_skipped_incorrect += 1

        num_correct = len(correct_indices)
        _print(f"  Correct: {num_correct}, Skipped incorrect: {num_skipped_incorrect}, "
               f"unverified: {num_skipped_unverified}, no_repo: {num_skipped_no_repo}")

        global_to_pos = {gidx: pos for pos, gidx in enumerate(correct_indices)}

        all_correct_positions = set(range(num_correct))
        to_process_positions = sorted(all_correct_positions - already_done)
        num_to_process = len(to_process_positions)
        num_already_done = len(already_done & all_correct_positions)

        if num_already_done > 0:
            _print(f"  Resuming: {num_already_done} done, {num_to_process} remaining.")
        else:
            _print(f"  Tokenizing {num_correct} trajectories with {num_workers} workers...")
        t0 = time.time()

        # Length index
        length_index_path = os.path.join(tmp_dir, "_lengths.json")
        length_index = {}
        if os.path.exists(length_index_path):
            try:
                with open(length_index_path, "r") as f:
                    length_index = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                length_index = {}

        results = [None] * num_correct
        num_failed = 0

        for pos in range(num_correct):
            if pos in length_index:
                results[pos] = length_index[pos]

        # Reload lengths for items done but not in index
        need_reload = [pos for pos in (already_done & all_correct_positions) if pos not in length_index]
        if need_reload:
            _print(f"  Loading lengths for {len(need_reload)} files...")
            from concurrent.futures import ThreadPoolExecutor as _TPE

            def _get_length(pos):
                try:
                    arr = np.load(os.path.join(tmp_dir, f"{pos}.npy"))
                    return (pos, len(arr))
                except Exception:
                    return (pos, None)

            with _TPE(max_workers=min(32, num_workers)) as executor:
                for pos, length in tqdm(executor.map(_get_length, need_reload),
                                        total=len(need_reload), desc="  Loading lengths"):
                    if length is not None:
                        length_index[pos] = length
                        results[pos] = length
                    else:
                        if os.path.exists(os.path.join(tmp_dir, f"{pos}.npy")):
                            os.remove(os.path.join(tmp_dir, f"{pos}.npy"))
                        to_process_positions.append(pos)
                        num_to_process += 1

        # Streaming tokenization
        if num_to_process > 0:
            _print(f"  Streaming tokenization: {num_to_process} trajectories...")
            import pyarrow as pa
            import pyarrow.ipc as ipc
            from bisect import bisect_left

            to_process_global_sorted = sorted(correct_indices[pos] for pos in to_process_positions)

            shard_info_path = os.path.join(tmp_dir, "_shard_info.json")
            with open(shard_info_path, "r") as f:
                shard_info_list = [(s[0], s[1], s[2]) for s in json.load(f)]

            shards_with_work = []
            for arrow_path, shard_start, num_rows in shard_info_list:
                shard_end = shard_start + num_rows
                left_idx = bisect_left(to_process_global_sorted, shard_start)
                if left_idx < len(to_process_global_sorted) and to_process_global_sorted[left_idx] < shard_end:
                    shards_with_work.append((arrow_path, shard_start, num_rows))

            _print(f"  {len(shards_with_work)}/{len(shard_info_list)} shards with work")
            to_process_global_set = set(to_process_global_sorted)

            with mp.Pool(processes=num_workers, initializer=_init_worker,
                         initargs=(model_path, tokenizer_cfg_dict)) as pool:
                pbar = tqdm(total=num_to_process, desc="  Tokenizing", unit="traj")

                for arrow_path, shard_start, num_rows in shards_with_work:
                    shard_end = shard_start + num_rows
                    shard_needs_set = {gidx for gidx in range(shard_start, shard_end)
                                       if gidx in to_process_global_set}
                    if not shard_needs_set:
                        continue

                    f_a = pa.memory_map(arrow_path, "r")
                    reader = ipc.open_stream(f_a)
                    batch_offset = shard_start
                    shard_work = []

                    for batch in reader:
                        batch_size = batch.num_rows
                        batch_end = batch_offset + batch_size
                        needs_in_batch = [gidx for gidx in range(batch_offset, batch_end)
                                          if gidx in shard_needs_set]

                        if needs_in_batch:
                            col_names = batch.schema.names
                            has_tools = "tools" in col_names
                            messages_col = batch.column("messages").to_pylist()
                            tools_col = batch.column("tools").to_pylist() if has_tools else None

                            for gidx in needs_in_batch:
                                local_idx = gidx - batch_offset
                                pos = global_to_pos[gidx]
                                tool_val = tools_col[local_idx] if tools_col else None
                                shard_work.append((pos, messages_col[local_idx], tool_val, tmp_dir))
                            del messages_col, tools_col

                        batch_offset = batch_end

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
                    if shard_work:
                        for result in pool.imap_unordered(_tokenize_trajectory, shard_work, chunksize=8):
                            if result is not None:
                                pos, length = result
                                length_index[pos] = length
                                results[pos] = length
                            else:
                                num_failed += 1
                            pbar.update(1)

                pbar.close()

            with open(length_index_path, "w") as f:
                json.dump(length_index, f)

        # Assemble cache
        valid_positions = [pos for pos, r in enumerate(results) if r is not None]
        valid_lengths = [results[pos] for pos in valid_positions]
        _print(f"  Tokenized {len(valid_positions)} trajectories in {time.time() - t0:.1f}s")
        if num_failed > 0:
            _print(f"  WARNING: {num_failed} failed")

        if not valid_positions:
            raise RuntimeError("All trajectories failed tokenization.")

        lengths = np.array(valid_lengths, dtype=np.int32)
        offsets = np.zeros(len(valid_positions) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths.astype(np.int64))
        total_tokens = int(offsets[-1])
        _print(f"  Total tokens: {total_tokens:,}")

        tokens = np.empty(total_tokens, dtype=np.int32)

        def _load_and_place(args):
            _, correct_pos, start, end = args
            tokens[start:end] = np.load(os.path.join(tmp_dir, f"{correct_pos}.npy"))

        from concurrent.futures import ThreadPoolExecutor
        load_args = [(i, cp, int(offsets[i]), int(offsets[i+1])) for i, cp in enumerate(valid_positions)]
        with ThreadPoolExecutor(max_workers=min(32, num_workers)) as executor:
            list(tqdm(executor.map(_load_and_place, load_args), total=len(load_args), desc="  Assembling"))

        shutil.rmtree(tmp_dir, ignore_errors=True)

        cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
        np.savez(cache_path, tokens=tokens, offsets=offsets, lengths=lengths)

        # Build repo metadata
        repo_list = []
        issue_hash_list = []
        source_dataset_list = []
        source_list = []
        for correct_pos in valid_positions:
            global_idx = correct_indices[correct_pos]
            meta = trajectory_meta[global_idx]
            repo_list.append(meta["repo"])
            issue_hash_list.append(meta["issue_content_hash"])
            source_dataset_list.append(meta["source_dataset"])
            source_list.append(meta.get("source", "unknown"))

        repo_metadata = {
            "repos": repo_list,
            "issue_content_hashes": issue_hash_list,
            "source_datasets": source_dataset_list,
            "sources": source_list,
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

        manifest = {
            "num_trajectories": len(valid_positions),
            "total_tokens": total_tokens,
            "num_failed": num_failed,
            "data_paths": data_paths,
            "model_path": model_path,
        }
        with open(os.path.join(abs_cache_dir, MANIFEST_FILE), "w") as f:
            json.dump(manifest, f, indent=2)

        with open(verified_path, "w") as f:
            f.write(f"verified at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        _print(f"  Preprocessing complete.")
        _print(sep)

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryAllTransferV2Dataset(BaseDataset):
    """
    Dataset that loads pre-tokenized trajectories from cache, groups them
    by repo, and creates consecutive-pair samples for transfer learning.

    Supports max_same_issue and subset filtering at runtime (no re-preprocessing).
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        max_token_length: int,
        seed: int = 42,
        repo_names: Optional[List[str]] = None,
        continuous_repo_num: int = -1,
        max_same_issue: int = -1,
        subset: Optional[List[str]] = None,
        min_traj_per_repo: int = -1,
    ):
        super().__init__(model_path)
        self.max_token_length = max_token_length
        self._base_seed = seed
        self._continuous_repo_num = continuous_repo_num
        self._max_same_issue = max_same_issue
        self._subset = subset
        self._min_traj_per_repo = min_traj_per_repo

        # Load cache
        abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
        verified_path = os.path.join(abs_cache_dir, VERIFIED_FILE)
        if not os.path.exists(verified_path):
            raise RuntimeError(f"Cache not verified at {abs_cache_dir}. Run preprocessing first.")

        repo_cache_path = os.path.join(abs_cache_dir, REPO_CACHE_FILE)
        if not os.path.exists(repo_cache_path):
            raise RuntimeError(f"Repo metadata not found at {repo_cache_path}.")

        cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
        data = np.load(cache_path, mmap_mode='r')
        self._tokens = data["tokens"]
        self._offsets = data["offsets"]
        self._lengths = data["lengths"]

        with open(repo_cache_path, "r") as f:
            repo_metadata = json.load(f)
        repo_list = repo_metadata["repos"]
        issue_hash_list = repo_metadata["issue_content_hashes"]
        source_dataset_list = repo_metadata["source_datasets"]

        # Validate subset names
        if subset is not None:
            invalid_names = [s for s in subset if s not in VALID_SUBSET_NAMES]
            if invalid_names:
                raise RuntimeError(
                    f"[trajectory_all_transfer_v2] Invalid subset names: {invalid_names}. "
                    f"Valid names are: {VALID_SUBSET_NAMES}"
                )

        # Apply subset filtering
        if subset is not None:
            subset_set = set(subset)
            subset_mask = [source_dataset_list[i] in subset_set for i in range(len(repo_list))]
        else:
            subset_mask = [True] * len(repo_list)

        # Apply repo filter set (for train/val split)
        if repo_names is not None:
            repo_names_set = set(repo_names)
        else:
            repo_names_set = None

        # Group by repo with subset + max_token_length + repo_names filters applied
        all_repo_to_valid_indices: Dict[str, List[int]] = defaultdict(list)
        for i, repo in enumerate(repo_list):
            if not subset_mask[i]:
                continue
            if self._lengths[i] > max_token_length:
                continue
            if repo_names_set is not None and repo not in repo_names_set:
                continue
            all_repo_to_valid_indices[repo].append(i)

        # Apply max_same_issue deduplication AFTER all other filters (random sampling)
        if max_same_issue > 0:
            # Collect all valid indices and group by issue_content_hash
            hash_to_indices: Dict[str, List[int]] = defaultdict(list)
            for indices in all_repo_to_valid_indices.values():
                for i in indices:
                    h = issue_hash_list[i]
                    hash_to_indices[h].append(i)
            # For each hash, if count exceeds max_same_issue, randomly sample
            rng_dedup = random.Random(seed)
            kept_indices: set = set()
            for h, indices in hash_to_indices.items():
                if len(indices) <= max_same_issue:
                    kept_indices.update(indices)
                else:
                    sampled = rng_dedup.sample(indices, max_same_issue)
                    kept_indices.update(sampled)
            # Rebuild repo_to_valid_indices with only kept indices
            self._repo_to_valid_indices: Dict[str, List[int]] = {}
            for repo, indices in all_repo_to_valid_indices.items():
                filtered = [i for i in indices if i in kept_indices]
                if filtered:
                    self._repo_to_valid_indices[repo] = filtered
        else:
            self._repo_to_valid_indices = dict(all_repo_to_valid_indices)

        # Apply min_traj_per_repo filter: discard repos with too few trajectories
        if min_traj_per_repo > 0:
            self._repo_to_valid_indices = {
                repo: indices for repo, indices in self._repo_to_valid_indices.items()
                if len(indices) >= min_traj_per_repo
            }

        # Build samples
        self.samples: List[Dict[str, Any]] = []
        self._build_samples(seed)

        logger.info(
            f"[trajectory_all_transfer_v2] Created {len(self.samples)} samples "
            f"from {len(self._repo_to_valid_indices)} repos "
            f"(max_same_issue={max_same_issue}, subset={subset})"
        )

    def _build_samples(self, seed: int):
        """Build samples with Poisson-chunked repo ordering."""
        rng = random.Random(seed)
        self.samples = []

        repo_names = list(self._repo_to_valid_indices.keys())
        rng.shuffle(repo_names)

        all_chunks: List[List[Dict[str, Any]]] = []

        for repo_name in repo_names:
            valid_indices = list(self._repo_to_valid_indices[repo_name])

            if len(valid_indices) < 2:
                # Skip repos with only 1 trajectory (no valid pair can be formed)
                # (This should not happen if min_traj_per_repo >= 2, but kept as safety check)
                continue

            rng.shuffle(valid_indices)

            repo_samples = []
            for pos in range(len(valid_indices)):
                ctx_idx = valid_indices[pos]
                conv_idx = valid_indices[(pos + 1) % len(valid_indices)]
                repo_samples.append({
                    "context_idx": ctx_idx,
                    "conversation_idx": conv_idx,
                    "repo": repo_name,
                })

            target = self._continuous_repo_num
            if target > 0 and len(repo_samples) > target:
                np_rng = np.random.RandomState(rng.randint(0, 2**31))
                start = 0
                while start < len(repo_samples):
                    remaining = len(repo_samples) - start
                    if remaining <= target:
                        all_chunks.append(repo_samples[start:])
                        break
                    cs = max(1, int(np_rng.poisson(target)))
                    end = min(start + cs, len(repo_samples))
                    leftover = len(repo_samples) - end
                    if 0 < leftover < max(1, target // 2):
                        end = len(repo_samples)
                    all_chunks.append(repo_samples[start:end])
                    start = end
            else:
                all_chunks.append(repo_samples)

        rng.shuffle(all_chunks)
        for chunk in all_chunks:
            self.samples.extend(chunk)

    def set_epoch(self, epoch: int):
        """Re-shuffle for the given epoch."""
        self._build_samples(self._base_seed + epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_token_ids(self, traj_idx: int) -> torch.Tensor:
        """Load token ids for a trajectory on-demand from memory-mapped array."""
        start = int(self._offsets[traj_idx])
        end = int(self._offsets[traj_idx + 1])
        return torch.from_numpy(
            self._tokens[start:end].astype(np.int64).copy()
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "context_token_ids": self._load_token_ids(sample["context_idx"]),
            "conversation_token_ids": self._load_token_ids(sample["conversation_idx"]),
            "repo": sample["repo"],
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

_THINK_TOKEN_ID = 248068
_IM_END_TOKEN_ID = 248046


def _compute_masked_labels(conv_tokens: torch.Tensor) -> torch.Tensor:
    """Compute labels: mask everything except tokens after <think> up to <|im_end|>."""
    labels = torch.full_like(conv_tokens, -100)
    length = conv_tokens.size(0)
    in_valid_region = False
    i = 0
    while i < length:
        if not in_valid_region:
            if conv_tokens[i].item() == _THINK_TOKEN_ID:
                i += 1
                in_valid_region = True
            else:
                i += 1
        else:
            labels[i] = conv_tokens[i]
            if conv_tokens[i].item() == _IM_END_TOKEN_ID:
                in_valid_region = False
            i += 1
    return labels


class TrajectoryAllTransferV2Collator(BaseCollator):
    """Collator that pads trajectories and computes masked labels."""

    def __init__(self, model_path: str, max_token_length: int,
                 pad_token_id: int = 0, num_mem_token: int = 0):
        super().__init__(model_path)
        self.max_token_length = max_token_length
        self.pad_token_id = pad_token_id
        self.num_mem_token = num_mem_token

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch_size = len(samples)
        max_len = self.max_token_length
        context_total_len = max_len + self.num_mem_token

        context_ids = torch.full((batch_size, context_total_len), self.pad_token_id, dtype=torch.long)
        conversation_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        context_lengths = torch.zeros(batch_size, dtype=torch.long)
        extra_info_list = []

        for i, s in enumerate(samples):
            ctx_len = min(s["context_token_ids"].size(0), max_len)
            context_ids[i, :ctx_len] = s["context_token_ids"][:ctx_len]
            context_lengths[i] = ctx_len

            conv_tokens = s["conversation_token_ids"][:max_len]
            conv_len = conv_tokens.size(0)
            conversation_ids[i, :conv_len] = conv_tokens
            labels[i, :conv_len] = _compute_masked_labels(conv_tokens)

            extra_info_list.append({"repo": s["repo"]})

        return [{
            "context_ids": context_ids,
            "conversation_ids": conversation_ids,
            "labels": labels,
            "context_lengths": context_lengths,
            "extra_info": extra_info_list,
        }]


# ---------------------------------------------------------------------------
# Train/Val split helper
# ---------------------------------------------------------------------------

def _get_train_val_repo_split(
    cache_dir: str, max_token_length: int, validation_split_num: int,
    seed: int, max_same_issue: int = -1, subset: Optional[List[str]] = None,
    validation_max_traj_per_repo: int = -1, min_traj_per_repo: int = -1,
) -> Tuple[List[str], List[str]]:
    """Split repos into train and val sets."""
    abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)

    data = np.load(os.path.join(abs_cache_dir, CACHE_FILE))
    lengths = data["lengths"]

    with open(os.path.join(abs_cache_dir, REPO_CACHE_FILE), "r") as f:
        repo_metadata = json.load(f)
    repo_list = repo_metadata["repos"]
    issue_hash_list = repo_metadata["issue_content_hashes"]
    source_dataset_list = repo_metadata["source_datasets"]

    # Apply filters
    if subset is not None:
        subset_set = set(subset)
        subset_mask = [source_dataset_list[i] in subset_set for i in range(len(repo_list))]
    else:
        subset_mask = [True] * len(repo_list)

    # Group by repo with subset + max_token_length filters applied first
    repo_to_valid_indices: Dict[str, List[int]] = defaultdict(list)
    for i, repo in enumerate(repo_list):
        if not subset_mask[i]:
            continue
        if lengths[i] > max_token_length:
            continue
        repo_to_valid_indices[repo].append(i)

    # Apply max_same_issue deduplication AFTER all other filters
    if max_same_issue > 0:
        hash_to_indices: Dict[str, List[int]] = defaultdict(list)
        for indices in repo_to_valid_indices.values():
            for i in indices:
                h = issue_hash_list[i]
                hash_to_indices[h].append(i)
        rng_dedup = random.Random(seed)
        kept_indices: set = set()
        for h, indices in hash_to_indices.items():
            if len(indices) <= max_same_issue:
                kept_indices.update(indices)
            else:
                sampled = rng_dedup.sample(indices, max_same_issue)
                kept_indices.update(sampled)
    else:
        kept_indices = None  # No dedup, keep all

    repo_sample_counts: Dict[str, int] = defaultdict(int)
    for repo, indices in repo_to_valid_indices.items():
        for i in indices:
            if kept_indices is None or i in kept_indices:
                repo_sample_counts[repo] += 1

    # Apply min_traj_per_repo filter: discard repos with too few trajectories
    if min_traj_per_repo > 0:
        repo_sample_counts = {
            repo: count for repo, count in repo_sample_counts.items()
            if count >= min_traj_per_repo
        }

    rng = random.Random(seed)
    all_repos = list(repo_sample_counts.keys())
    rng.shuffle(all_repos)

    val_repos = []
    val_total = 0
    for repo in all_repos:
        if val_total >= validation_split_num:
            break
        count = repo_sample_counts[repo]
        if count > 0:
            # Skip repos that exceed validation_max_traj_per_repo
            if validation_max_traj_per_repo > 0 and count > validation_max_traj_per_repo:
                continue
            val_repos.append(repo)
            val_total += count

    val_repo_set = set(val_repos)
    train_repos = [r for r in all_repos if r not in val_repo_set]

    return train_repos, val_repos


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """Create dataset and collator."""
    data_cfg = cfg.data

    max_token_length = data_cfg.max_token_length
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_v2_tokens")
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)
    continuous_repo_num = data_cfg.get("continuous_repo_num", -1)
    max_same_issue = data_cfg.get("max_same_issue", -1)
    min_traj_per_repo = data_cfg.get("min_traj_per_repo", -1)
    subset = data_cfg.get("subset", None)
    if subset is not None:
        subset = list(subset)

    validation_max_traj_per_repo = data_cfg.get("validation_max_traj_per_repo", -1)

    train_repo_names = None
    if validation_split_num > 0:
        train_repos, _ = _get_train_val_repo_split(
            cache_dir=cache_dir, max_token_length=max_token_length,
            validation_split_num=validation_split_num, seed=seed,
            max_same_issue=max_same_issue, subset=subset,
            validation_max_traj_per_repo=validation_max_traj_per_repo,
            min_traj_per_repo=min_traj_per_repo,
        )
        train_repo_names = train_repos

    dataset = TrajectoryAllTransferV2Dataset(
        model_path=model_path, cache_dir=cache_dir,
        max_token_length=max_token_length, seed=seed,
        repo_names=train_repo_names, continuous_repo_num=continuous_repo_num,
        max_same_issue=max_same_issue, subset=subset,
        min_traj_per_repo=min_traj_per_repo,
    )

    collator = TrajectoryAllTransferV2Collator(
        model_path=model_path, max_token_length=max_token_length,
        pad_token_id=pad_token_id, num_mem_token=num_mem_token,
    )

    return dataset, collator


def create_val_dataset(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """Create validation dataset."""
    data_cfg = cfg.data
    max_token_length = data_cfg.max_token_length
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_v2_tokens")
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    validation_split_num = data_cfg.get("validation_split_num", -1)
    continuous_repo_num = data_cfg.get("continuous_repo_num", -1)
    max_same_issue = data_cfg.get("max_same_issue", -1)
    min_traj_per_repo = data_cfg.get("min_traj_per_repo", -1)
    subset = data_cfg.get("subset", None)
    if subset is not None:
        subset = list(subset)

    validation_max_traj_per_repo = data_cfg.get("validation_max_traj_per_repo", -1)

    if validation_split_num <= 0:
        return None

    _, val_repos = _get_train_val_repo_split(
        cache_dir=cache_dir, max_token_length=max_token_length,
        validation_split_num=validation_split_num, seed=seed,
        max_same_issue=max_same_issue, subset=subset,
        validation_max_traj_per_repo=validation_max_traj_per_repo,
        min_traj_per_repo=min_traj_per_repo,
    )

    if not val_repos:
        return None

    return TrajectoryAllTransferV2Dataset(
        model_path=model_path, cache_dir=cache_dir,
        max_token_length=max_token_length, seed=seed,
        repo_names=val_repos, continuous_repo_num=continuous_repo_num,
        max_same_issue=max_same_issue, subset=subset,
        min_traj_per_repo=min_traj_per_repo,
    )


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """Debug: inspect samples and output statistics."""
    from utils.mydata import resolve_pad_token_id, debug_dataset

    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    num_mem_token = 10

    dataset, collator = create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
    tokenizer = create_tokenizer(model_path, tokenizer_cfg=cfg.tokenizer)

    data_cfg = cfg.data
    max_same_issue = data_cfg.get("max_same_issue", -1)
    subset = data_cfg.get("subset", None)
    if subset is not None:
        subset = list(subset)

    metadata = {
        "max_token_length": data_cfg.max_token_length,
        "cache_dir": data_cfg.get("cache_dir", "cache/trajectory_all_transfer_v2_tokens"),
        "max_same_issue": max_same_issue,
        "subset": subset,
        "num_samples_in_dataset": len(dataset),
    }

    dataset_name = data_cfg.get("name", "trajectory_all_transfer_v2")

    # Select diverse samples for visualization
    TOOL_CALL_TOKEN_ID = 248058
    NUM_PER_STYLE = 5
    openai_indices = []
    xml_indices = []
    total_samples = len(dataset.samples)
    step = max(1, total_samples // 200)
    for idx in range(0, total_samples, step):
        ctx_tokens = dataset.samples[idx]["context_token_ids"]
        has_tool_call = (ctx_tokens == TOOL_CALL_TOKEN_ID).any().item()
        if has_tool_call and len(openai_indices) < NUM_PER_STYLE:
            openai_indices.append(idx)
        elif not has_tool_call and len(xml_indices) < NUM_PER_STYLE:
            xml_indices.append(idx)
        if len(openai_indices) >= NUM_PER_STYLE and len(xml_indices) >= NUM_PER_STYLE:
            break

    selected_indices = openai_indices + xml_indices
    if selected_indices:
        original_samples = dataset.samples
        reordered = [original_samples[i] for i in selected_indices]
        remaining = [s for idx, s in enumerate(original_samples) if idx not in set(selected_indices)]
        dataset.samples = reordered + remaining

    num_debug_samples = len(selected_indices) if selected_indices else 5

    debug_dataset(
        dataset=dataset, collator=collator, tokenizer=tokenizer,
        dataset_name=dataset_name, metadata=metadata,
        num_samples=num_debug_samples, num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )

    # Output statistics
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_all_transfer_v2_tokens")
    abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)

    with open(os.path.join(abs_cache_dir, REPO_CACHE_FILE), "r") as f:
        repo_metadata = json.load(f)
    source_dataset_list = repo_metadata["source_datasets"]
    issue_hash_list = repo_metadata["issue_content_hashes"]

    # Per-source stats
    source_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "selected": 0})
    data = np.load(os.path.join(abs_cache_dir, CACHE_FILE))
    lengths = data["lengths"]

    if subset is not None:
        subset_set = set(subset)
        s_mask = [source_dataset_list[i] in subset_set for i in range(len(source_dataset_list))]
    else:
        s_mask = [True] * len(source_dataset_list)

    # Apply subset + max_token_length filters first
    filtered_mask = [
        s_mask[i] and lengths[i] <= data_cfg.max_token_length
        for i in range(len(source_dataset_list))
    ]

    # Apply max_same_issue deduplication AFTER all other filters
    if max_same_issue > 0:
        # Group filtered indices by issue_content_hash
        hash_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i in range(len(source_dataset_list)):
            if not filtered_mask[i]:
                continue
            h = issue_hash_list[i]
            hash_to_indices[h].append(i)
        # For each hash, if count exceeds max_same_issue, randomly sample
        seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
        rng_dedup = random.Random(seed)
        selected_mask = [False] * len(source_dataset_list)
        for h, indices in hash_to_indices.items():
            if len(indices) <= max_same_issue:
                for i in indices:
                    selected_mask[i] = True
            else:
                sampled = rng_dedup.sample(indices, max_same_issue)
                for i in sampled:
                    selected_mask[i] = True
    else:
        selected_mask = filtered_mask

    # Apply min_traj_per_repo filter: discard repos with too few trajectories
    min_traj_per_repo = data_cfg.get("min_traj_per_repo", -1)
    if min_traj_per_repo > 0:
        repo_list_local = repo_metadata["repos"]
        repo_selected_counts: Dict[str, int] = defaultdict(int)
        for i in range(len(source_dataset_list)):
            if selected_mask[i]:
                repo_selected_counts[repo_list_local[i]] += 1
        # Zero out repos below threshold
        disqualified_repos = {r for r, c in repo_selected_counts.items() if c < min_traj_per_repo}
        if disqualified_repos:
            for i in range(len(source_dataset_list)):
                if selected_mask[i] and repo_list_local[i] in disqualified_repos:
                    selected_mask[i] = False

    for i in range(len(source_dataset_list)):
        src = source_dataset_list[i]
        source_stats[src]["total"] += 1
        if selected_mask[i]:
            source_stats[src]["selected"] += 1

    unique_hashes = len(set(h for i, h in enumerate(issue_hash_list) if selected_mask[i]))

    # Compute train/validation split by repo
    validation_split_num = data_cfg.get("validation_split_num", -1)
    seed = cfg.get("seed", {}).get("dataset", 42) if cfg.get("seed") else 42
    repo_list = repo_metadata["repos"]

    validation_max_traj_per_repo = data_cfg.get("validation_max_traj_per_repo", -1)

    if validation_split_num > 0:
        train_repos, val_repos = _get_train_val_repo_split(
            cache_dir=abs_cache_dir, max_token_length=data_cfg.max_token_length,
            validation_split_num=validation_split_num, seed=seed,
            max_same_issue=max_same_issue, subset=subset,
            validation_max_traj_per_repo=validation_max_traj_per_repo,
            min_traj_per_repo=min_traj_per_repo,
        )
        train_repo_set = set(train_repos)
        val_repo_set = set(val_repos)
    else:
        train_repo_set = set(repo_list)
        val_repo_set = set()

    # Build train/val masks
    train_selected_mask = [selected_mask[i] and repo_list[i] in train_repo_set for i in range(len(repo_list))]
    val_selected_mask = [selected_mask[i] and repo_list[i] in val_repo_set for i in range(len(repo_list))]

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{dataset_name}_debug.txt")

    # Extract trajectory counts from repo_metadata
    num_total_raw = repo_metadata.get("num_total_raw", "N/A")
    num_correct_traj = repo_metadata.get("num_correct", "N/A")
    num_skipped_incorrect = repo_metadata.get("num_skipped_incorrect", "N/A")
    num_skipped_unverified = repo_metadata.get("num_skipped_unverified", "N/A")
    num_skipped_no_repo = repo_metadata.get("num_skipped_no_repo", "N/A")
    num_failed_tokenization = repo_metadata.get("num_failed_tokenization", "N/A")
    num_trajectories_in_cache = repo_metadata.get("num_trajectories", len(lengths))

    with open(output_path, "a", encoding="utf-8") as f:
        sep = "=" * 120
        f.write(f"\n\n{sep}\n")
        f.write(f"  DATASET STATISTICS (trajectory_all_transfer_v2)\n")
        f.write(f"  max_token_length = {data_cfg.max_token_length}\n")
        f.write(f"  max_same_issue = {max_same_issue}\n")
        f.write(f"  min_traj_per_repo = {min_traj_per_repo}\n")
        f.write(f"  subset = {subset}\n")
        f.write(f"  validation_split_num = {validation_split_num}\n")
        f.write(f"  unique issue_content_hashes (after filtering) = {unique_hashes}\n")
        f.write(f"\n  --- Trajectory Counts ---\n")
        f.write(f"  Total raw trajectories: {num_total_raw}\n")
        f.write(f"  Correct trajectories (before tokenization): {num_correct_traj}\n")
        f.write(f"  Skipped incorrect: {num_skipped_incorrect}\n")
        f.write(f"  Skipped unverified: {num_skipped_unverified}\n")
        f.write(f"  Skipped no_repo: {num_skipped_no_repo}\n")
        f.write(f"  Failed tokenization: {num_failed_tokenization}\n")
        f.write(f"  Correct trajectories in cache (tokenized successfully): {num_trajectories_in_cache}\n")
        f.write(f"{sep}\n\n")

        f.write(f"  {'Source Dataset':<60s} | {'Total (cache)':>12s} | {'Selected':>15s}\n")
        f.write(f"  {'-'*60}-+-{'-'*12}-+-{'-'*15}\n")
        total_all = 0
        selected_all = 0
        for src in sorted(source_stats.keys()):
            s = source_stats[src]
            total_all += s["total"]
            selected_all += s["selected"]
            f.write(f"  {src:<60s} | {s['total']:>12d} | {s['selected']:>15d}\n")
        f.write(f"  {'-'*60}-+-{'-'*12}-+-{'-'*15}\n")
        f.write(f"  {'TOTAL':<60s} | {total_all:>12d} | {selected_all:>15d}\n\n")
        f.write(f"{sep}\n\n")

        # Per-repo statistics for TRAIN set
        repo_train_total: Dict[str, int] = defaultdict(int)
        repo_train_selected: Dict[str, int] = defaultdict(int)
        for i, repo in enumerate(repo_list):
            if repo in train_repo_set:
                repo_train_total[repo] += 1
                if train_selected_mask[i]:
                    repo_train_selected[repo] += 1

        sorted_train_repos = sorted(repo_train_total.keys(), key=lambda r: repo_train_total[r], reverse=True)

        f.write(f"\n{sep}\n")
        f.write(f"  PER-REPO STATISTICS — TRAIN SET (selected / total)\n")
        f.write(f"  Train repos: {len(sorted_train_repos)}, Train selected trajectories: {sum(1 for x in train_selected_mask if x)}\n")
        f.write(f"{sep}\n\n")
        f.write(f"  {'Repo':<80s} | {'Selected':>8s} | {'Total':>8s} | {'Ratio':>8s}\n")
        f.write(f"  {'-'*80}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
        for repo in sorted_train_repos:
            t = repo_train_total[repo]
            v = repo_train_selected.get(repo, 0)
            ratio = f"{v/t*100:.1f}%" if t > 0 else "N/A"
            f.write(f"  {repo:<80s} | {v:>8d} | {t:>8d} | {ratio:>8s}\n")
        f.write(f"  {'-'*80}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
        repos_with_selected_train = sum(1 for r in sorted_train_repos if repo_train_selected.get(r, 0) > 0)
        f.write(f"  Total train repos: {len(sorted_train_repos)}, repos with selected trajectories: {repos_with_selected_train}\n\n")
        f.write(f"{sep}\n\n")

        # Per-repo statistics for VALIDATION set
        if val_repo_set:
            repo_val_total: Dict[str, int] = defaultdict(int)
            repo_val_selected: Dict[str, int] = defaultdict(int)
            for i, repo in enumerate(repo_list):
                if repo in val_repo_set:
                    repo_val_total[repo] += 1
                    if val_selected_mask[i]:
                        repo_val_selected[repo] += 1

            sorted_val_repos = sorted(repo_val_total.keys(), key=lambda r: repo_val_total[r], reverse=True)

            f.write(f"\n{sep}\n")
            f.write(f"  PER-REPO STATISTICS — VALIDATION SET (selected / total)\n")
            f.write(f"  Validation repos: {len(sorted_val_repos)}, Validation selected trajectories: {sum(1 for x in val_selected_mask if x)}\n")
            f.write(f"{sep}\n\n")
            f.write(f"  {'Repo':<80s} | {'Selected':>8s} | {'Total':>8s} | {'Ratio':>8s}\n")
            f.write(f"  {'-'*80}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
            for repo in sorted_val_repos:
                t = repo_val_total[repo]
                v = repo_val_selected.get(repo, 0)
                ratio = f"{v/t*100:.1f}%" if t > 0 else "N/A"
                f.write(f"  {repo:<80s} | {v:>8d} | {t:>8d} | {ratio:>8s}\n")
            f.write(f"  {'-'*80}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
            repos_with_selected_val = sum(1 for r in sorted_val_repos if repo_val_selected.get(r, 0) > 0)
            f.write(f"  Total validation repos: {len(sorted_val_repos)}, repos with selected trajectories: {repos_with_selected_val}\n\n")
            f.write(f"{sep}\n")
        else:
            f.write(f"\n{sep}\n")
            f.write(f"  VALIDATION SET: disabled (validation_split_num <= 0)\n")
            f.write(f"{sep}\n")

    # --- Visualization: token length distribution histograms ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_lengths = lengths.astype(np.int64)
    max_token_length = data_cfg.max_token_length

    # Determine selected lengths (after all filters)
    selected_lengths_arr = np.array([
        lengths[i] for i in range(len(lengths)) if selected_mask[i]
    ], dtype=np.int64)

    # Train/validation selected lengths
    train_lengths_arr = np.array([
        lengths[i] for i in range(len(lengths)) if train_selected_mask[i]
    ], dtype=np.int64)
    val_lengths_arr = np.array([
        lengths[i] for i in range(len(lengths)) if val_selected_mask[i]
    ], dtype=np.int64)

    # Plot 1: Correct trajectory token length distribution
    num_correct_for_pct = len(all_lengths)  # all_lengths = correct trajectories in cache
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist(all_lengths, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(x=max_token_length, color="red", linestyle="--", linewidth=2,
               label=f"max_token_length={max_token_length}")
    ax.set_xlabel("Token Length", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Correct Trajectories Token Length Distribution (N={num_correct_for_pct:,})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    pct_of_correct = len(selected_lengths_arr) / num_correct_for_pct * 100 if num_correct_for_pct > 0 else 0
    stats_text = (f"min={all_lengths.min():,}, max={all_lengths.max():,}\n"
                  f"mean={all_lengths.mean():,.0f}, median={np.median(all_lengths):,.0f}\n"
                  f"≤{max_token_length}: {len(selected_lengths_arr):,} ({pct_of_correct:.1f}% of correct trajectories)")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    total_png_path = os.path.join(output_dir, f"{dataset_name}_total.png")
    plt.savefig(total_png_path, dpi=150)
    plt.close(fig)
    print(f"[debug] Token length distribution (all) saved to: {total_png_path}")

    # Plot 2: Train set selected trajectory token length distribution
    if len(train_lengths_arr) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.hist(train_lengths_arr, bins=100, color="forestgreen", edgecolor="black", alpha=0.7)
        ax.axvline(x=max_token_length, color="red", linestyle="--", linewidth=2,
                   label=f"max_token_length={max_token_length}")
        ax.set_xlabel("Token Length", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Train Set Selected Trajectories Token Length Distribution (N={len(train_lengths_arr):,})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        stats_text = (f"min={train_lengths_arr.min():,}, max={train_lengths_arr.max():,}\n"
                      f"mean={train_lengths_arr.mean():,.0f}, median={np.median(train_lengths_arr):,.0f}")
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        train_png_path = os.path.join(output_dir, f"{dataset_name}_train.png")
        plt.savefig(train_png_path, dpi=150)
        plt.close(fig)
        print(f"[debug] Token length distribution (train) saved to: {train_png_path}")
    else:
        print("[debug] WARNING: No train selected trajectories to plot.")

    # Plot 3: Validation set selected trajectory token length distribution
    if len(val_lengths_arr) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.hist(val_lengths_arr, bins=100, color="darkorange", edgecolor="black", alpha=0.7)
        ax.axvline(x=max_token_length, color="red", linestyle="--", linewidth=2,
                   label=f"max_token_length={max_token_length}")
        ax.set_xlabel("Token Length", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Validation Set Selected Trajectories Token Length Distribution (N={len(val_lengths_arr):,})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        stats_text = (f"min={val_lengths_arr.min():,}, max={val_lengths_arr.max():,}\n"
                      f"mean={val_lengths_arr.mean():,.0f}, median={np.median(val_lengths_arr):,.0f}")
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        val_png_path = os.path.join(output_dir, f"{dataset_name}_validation.png")
        plt.savefig(val_png_path, dpi=150)
        plt.close(fig)
        print(f"[debug] Token length distribution (validation) saved to: {val_png_path}")
    else:
        print("[debug] WARNING: No validation selected trajectories to plot (validation_split_num <= 0 or no data).")

    print(f"\n[debug] Statistics appended to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Trajectory All Transfer V2 dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true")
    group.add_argument("--preprocess", action="store_true")
    parser.add_argument("--config", type=str,
                        default="configs/data/pretrain/trajectory_all_transfer_v2.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    data_cfg = OmegaConf.load(args.config)
    _base_yaml = os.path.join("configs", "base.yaml")
    _base_cfg = OmegaConf.load(_base_yaml) if os.path.exists(_base_yaml) else OmegaConf.create({})
    _dataset_seed = _base_cfg.get("seed", {}).get("dataset", 42)

    _tokenizer_yaml = os.path.join("configs", "tokenizer", "origin.yaml")
    _tokenizer_cfg = OmegaConf.load(_tokenizer_yaml) if os.path.exists(_tokenizer_yaml) else OmegaConf.create({})

    cfg = OmegaConf.create({
        "data": data_cfg,
        "seed": {"dataset": _dataset_seed},
        "tokenizer": _tokenizer_cfg,
    })

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
            print("ERROR: --model_path is required for --preprocess mode.")
            sys.exit(1)
        preprocess(cfg, model_path)
    elif args.debug:
        if model_path is None:
            print("ERROR: --model_path is required for --debug mode.")
            sys.exit(1)
        debug(cfg, model_path)