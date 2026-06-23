#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utility functions for the training pipeline.
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

from utils.myparallel import is_main_process, is_main_process_per_node


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to 'Xd Xh Xm X.Xs' human-readable string.

    Only includes non-zero components (except seconds which is always shown).
    Seconds are displayed with one decimal place.

    Examples:
        format_duration(0.5)       -> '0.5s'
        format_duration(65.3)      -> '1m 5.3s'
        format_duration(3661.12)   -> '1h 1m 1.1s'
        format_duration(90061.0)   -> '1d 1h 1m 1.0s'
    """
    if seconds < 0:
        seconds = 0.0
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    secs = seconds % 60
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs:.1f}s")
    return " ".join(parts)


# Categories that only produce log files on node 0 (global aggregation results).
NODE0_ONLY_CATEGORIES: Set[str] = {"per_node_loss", "evaluation"}


def setup_debug_loggers(
    log_dir: str,
    node_rank: int,
    session_id: str = "",
) -> Tuple[Dict[str, str], Dict[str, str], Set[str]]:
    """Set up dedicated debug loggers for various categories.

    Each debug category writes to its own file: logs/<category>_node<rank>_<session_id>.log
    Node-0-only categories (e.g. per_node_loss, evaluation) only create files on node 0.

    When session_id is provided, log filenames include it so that each training
    session (initial run + each resume) produces separate files. This ensures
    wandb preserves all sessions' logs without overwriting.

    Args:
        log_dir: Directory to store log files.
        node_rank: The rank of the current node.
        session_id: Unique identifier for this training session (e.g. timestamp).
                    If empty, no session suffix is added (backward compatible).

    Returns:
        A tuple of (debug_categories, debug_log_paths, node0_only_categories):
            - debug_categories: mapping from category name to logger name
            - debug_log_paths: mapping from category name to log file path
            - node0_only_categories: set of category names that only log on node 0
    """
    os.makedirs(log_dir, exist_ok=True)

    # Session suffix for unique filenames per training session
    _session_suffix = f"_{session_id}" if session_id else ""

    # Category -> logger name mapping
    debug_categories: Dict[str, str] = {
        "training_detail":               "debug.training_detail",
        "training_detail_ppl_threshold": "debug.training_detail_ppl_threshold",
        "gpu_memory":                    "debug.gpu_memory",
        "dp_consistency":                "debug.dp_consistency",
        "nograd_loradict":               "debug.nograd_loradict",
        "per_node_loss":                 "debug.per_node_loss",
        "evaluation":                    "debug.evaluation",
    }
    node0_only_categories = NODE0_ONLY_CATEGORIES
    debug_log_paths: Dict[str, str] = {}

    for cat_name, logger_name in debug_categories.items():
        cat_logger = logging.getLogger(logger_name)
        cat_logger.setLevel(logging.DEBUG)
        cat_logger.propagate = False  # Do NOT propagate to root logger / stdout

        # Remove any existing handlers from previous sessions (in case of re-init)
        cat_logger.handlers.clear()

        # per_node_loss / evaluation: only node 0 gets a file handler
        if cat_name in node0_only_categories:
            if is_main_process_per_node() and is_main_process():
                log_path = os.path.join(log_dir, f"{cat_name}{_session_suffix}.log")
                debug_log_paths[cat_name] = log_path
                fh = logging.FileHandler(log_path, mode="w")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                cat_logger.addHandler(fh)
                cat_logger.info(f"Debug log initialized: {log_path}")
        else:
            log_path = os.path.join(log_dir, f"{cat_name}_node{node_rank}{_session_suffix}.log")
            debug_log_paths[cat_name] = log_path
            if is_main_process_per_node():
                fh = logging.FileHandler(log_path, mode="w")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                cat_logger.addHandler(fh)
                cat_logger.info(f"Debug log initialized: {log_path}")

    return debug_categories, debug_log_paths, node0_only_categories


def flush_debug_loggers(debug_categories: Dict[str, str]) -> None:
    """Flush all debug logger handlers to ensure final writes are captured.

    Args:
        debug_categories: mapping from category name to logger name.
    """
    if is_main_process_per_node():
        for logger_name in debug_categories.values():
            for handler in logging.getLogger(logger_name).handlers:
                handler.flush()
