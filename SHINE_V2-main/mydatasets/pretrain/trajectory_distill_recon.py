#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory Distill Recon Dataset Module for SHINE_V2

Loads trajectories from SHINE_SWE_DISTILLATION, tokenizes each trajectory
using the chat template, caches the results, and filters by max_token_length.

Each trajectory becomes one data sample where context, conversation, and labels
are all the same token sequence (with padding positions masked as -100 in labels).

Preprocessing:
    Tokenizes all trajectories in parallel using multiprocessing, applies
    the chat template via the tokenizer, and stores results in a cache
    directory as .npz files.

Dataset:
    Loads cached tokens and filters trajectories by max_token_length.
    Each trajectory that fits within max_token_length becomes one sample.

Unified factory interface:
    create_dataset_and_collator(cfg, model_path, pad_token_id, num_mem_token)
        -> (TrajectoryDistillReconDataset, TrajectoryDistillReconCollator)
"""

from __future__ import annotations

import os
import json
import random
import logging
import time
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any

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
MANIFEST_FILE = "manifest.json"
VERIFIED_FILE = "VERIFIED"


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
                        # The chat template iterates over arguments.items()
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
        # (apply_chat_template with tokenize=True returns BatchEncoding in this
        # version of transformers, not a plain list)
        text = _worker_tokenizer.apply_chat_template(
            fixed_messages,
            tools=fixed_tools,
            add_generation_prompt=False,
            tokenize=False,
        )
        token_ids = _worker_tokenizer.encode(text, add_special_tokens=False)
        return (idx, np.array(token_ids, dtype=np.int32))
    except Exception as e:
        logger.warning(f"[trajectory_distill_recon] Failed to tokenize trajectory {idx}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def _load_all_shards(data_path: str) -> List[Dict[str, Any]]:
    """
    Load all arrow shard directories under data_path and return a list of rows.
    Each row is a dict with 'messages' and 'tools' keys.
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
            # Convert to list of dicts
            for i in range(table.num_rows):
                row_slice = table.slice(i, 1).to_pydict()
                messages = row_slice["messages"][0]
                tools = row_slice["tools"][0] if "tools" in row_slice else None
                all_rows.append({
                    "messages": messages,
                    "tools": tools,
                })
            f.close()

    logger.info(f"[trajectory_distill_recon] Loaded {len(all_rows)} trajectories from {len(shard_dirs)} shards")
    return all_rows


# ---------------------------------------------------------------------------
# Preprocessing: tokenize all trajectories and cache
# ---------------------------------------------------------------------------

def preprocess(cfg, model_path: str):
    """
    Tokenize all trajectories in parallel and cache the results.

    Cache layout::
        {cache_dir}/
            manifest.json              # metadata (num_trajectories, etc.)
            all_trajectories.npz       # tokens and offsets for all trajectories
            VERIFIED                   # marker written after successful caching

    The .npz file stores:
        - tokens: 1-D int32 array (concatenation of all trajectory token sequences)
        - offsets: 1-D int64 array (start offset of each trajectory in tokens array)
        - lengths: 1-D int32 array (length of each trajectory)
    """
    import io
    from omegaconf import OmegaConf

    data_cfg = cfg.data
    dataset_name = data_cfg.get("name", "trajectory_distill_recon")
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

    # Check if already verified
    verified_path = os.path.join(abs_cache_dir, VERIFIED_FILE)
    if os.path.exists(verified_path):
        _print(f"  Cache already verified at: {abs_cache_dir}")
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
        raise RuntimeError("All trajectories failed tokenization. Check data format and tokenizer compatibility.")

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
    cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
    _print(f"  Saving cache to: {cache_path}")
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

    _print(f"  Cache verified and saved.")
    _print(sep)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    buf.close()

    logger.info(f"[trajectory_distill_recon] Preprocessing complete. Output: {output_path}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryDistillReconDataset(BaseDataset):
    """
    Dataset that loads pre-tokenized trajectories from cache and filters
    by max_token_length.

    Each sample is a dict with:
        - context_ids:      LongTensor (token_length,) -- the trajectory tokens
        - conversation_ids: LongTensor (token_length,) -- same as context_ids
        - labels:           LongTensor (token_length,) -- same as context_ids
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        max_token_length: int,
    ):
        super().__init__(model_path)
        self.max_token_length = max_token_length

        # Load cache
        abs_cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(_project_root, cache_dir)
        verified_path = os.path.join(abs_cache_dir, VERIFIED_FILE)
        if not os.path.exists(verified_path):
            raise RuntimeError(
                f"Cache not verified at {abs_cache_dir}. "
                f"Run preprocessing first: python mydatasets/pretrain/trajectory_distill_recon.py --preprocess"
            )

        cache_path = os.path.join(abs_cache_dir, CACHE_FILE)
        data = np.load(cache_path)
        tokens = data["tokens"]
        offsets = data["offsets"]
        lengths = data["lengths"]

        # Filter trajectories by max_token_length
        self.samples: List[torch.Tensor] = []
        num_total = len(lengths)
        for i in range(num_total):
            if lengths[i] <= max_token_length:
                start = int(offsets[i])
                end = int(offsets[i + 1])
                token_ids = torch.from_numpy(tokens[start:end].astype(np.int64))
                self.samples.append(token_ids)

        logger.info(
            f"[trajectory_distill_recon] Loaded {len(self.samples)} trajectories "
            f"(filtered from {num_total}, max_token_length={max_token_length})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token_ids = self.samples[idx]
        return {
            "context_ids": token_ids,
            "conversation_ids": token_ids.clone(),
            "labels": token_ids.clone(),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class TrajectoryDistillReconCollator(BaseCollator):
    """
    Collator that pads trajectories to fixed lengths.

    Each trajectory is right-padded to max_token_length.
    Labels use -100 for padding positions.

    Produced batch keys:
        context_ids:      (B, context_total_len)  where context_total_len = max_token_length + num_mem_token
        conversation_ids: (B, max_token_length)
        labels:           (B, max_token_length)
        context_lengths:  (B,)  actual number of valid tokens per sample
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

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
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

        for i, s in enumerate(samples):
            seq_len = min(s["context_ids"].size(0), max_len)
            # Layout: [valid_tokens | mem_placeholders | padding]
            context_ids[i, :seq_len] = s["context_ids"][:seq_len]
            conversation_ids[i, :seq_len] = s["conversation_ids"][:seq_len]
            labels[i, :seq_len] = s["labels"][:seq_len]
            context_lengths[i] = seq_len

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
# Factory function -- unified interface
# ---------------------------------------------------------------------------

def create_dataset_and_collator(cfg, model_path: str, pad_token_id: int, num_mem_token: int = 0):
    """
    Create a TrajectoryDistillReconDataset and its collator.

    Parameters from cfg.data (trajectory_distill_recon.yaml):
        - max_token_length:   maximum token length for filtering trajectories
        - cache_dir:          directory containing preprocessed cache

    Args:
        cfg: Full Hydra DictConfig (must have cfg.data).
        model_path: Absolute path to the model directory.
        pad_token_id: Token id used for padding.
        num_mem_token: Number of memory token placeholders.

    Returns:
        tuple: (TrajectoryDistillReconDataset, TrajectoryDistillReconCollator)
    """
    data_cfg = cfg.data

    max_token_length = data_cfg.max_token_length
    cache_dir = data_cfg.get("cache_dir", "cache/trajectory_distill_recon_tokens")

    logger.info(
        f"[trajectory_distill_recon] Creating dataset: "
        f"max_token_length={max_token_length}, cache_dir={cache_dir}"
    )

    dataset = TrajectoryDistillReconDataset(
        model_path=model_path,
        cache_dir=cache_dir,
        max_token_length=max_token_length,
    )

    collator = TrajectoryDistillReconCollator(
        model_path=model_path,
        max_token_length=max_token_length,
        pad_token_id=pad_token_id,
        num_mem_token=num_mem_token,
    )

    return dataset, collator


# ---------------------------------------------------------------------------
# Debug -- inspect first few samples
# ---------------------------------------------------------------------------

def debug(cfg, model_path: str):
    """
    Create the dataset + collator, then call the generic
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
        "max_token_length": data_cfg.max_token_length,
        "cache_dir": data_cfg.get("cache_dir", "cache/trajectory_distill_recon_tokens"),
        "num_samples_in_dataset": len(dataset),
    }

    debug_dataset(
        dataset=dataset,
        collator=collator,
        tokenizer=tokenizer,
        dataset_name=data_cfg.get("name", "trajectory_distill_recon"),
        metadata=metadata,
        num_samples=5,
        num_mem_token=num_mem_token,
        pad_token_id=pad_token_id,
    )


# ---------------------------------------------------------------------------
# CLI entry point:  python mydatasets/pretrain/trajectory_distill_recon.py --debug | --preprocess
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Trajectory Distill Recon dataset utilities")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", help="Debug: inspect first 5 samples")
    group.add_argument("--preprocess", action="store_true", help="Preprocess: tokenize and cache all trajectories")
    parser.add_argument("--config", type=str, default="configs/data/pretrain/trajectory_distill_recon.yaml",
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
