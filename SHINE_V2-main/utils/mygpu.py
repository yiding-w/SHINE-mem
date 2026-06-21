#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Memory Monitoring Utilities

This module provides utility functions for monitoring GPU memory usage
in distributed training scenarios, particularly useful for pipeline parallel setups.

Usage Examples:
-------------
1. Basic single GPU monitoring:
   >>> from utils.mygpu import gpu_stats
   >>> gpu_stats("After model loading")

2. Distributed GPU monitoring (all ranks):
   >>> from utils.mygpu import all_gpu_stats
   >>> all_gpu_stats("After forward pass")

3. Get memory statistics as dictionary:
   >>> stats = gpu_stats("Checkpoint", return_stats=True)
   >>> print(f"Allocated: {stats['alloc_gb']:.2f} GB")

Note: These functions require torch.distributed to be initialized for all_gpu_stats.
"""

import torch
import torch.distributed as dist
import logging
from utils.myparallel import is_main_process, is_main_process_per_node, get_local_rank

logger = logging.getLogger(__name__)


def gpu_stats(label: str = "", device: torch.device = None, return_stats: bool = False):
    """
    Print VRAM usage for the local GPU and optionally return statistics.
    
    This function provides detailed memory usage information for a single GPU,
    including allocated memory, reserved memory, total memory, and estimated free memory.
    
    Args:
        label (str): Descriptive label for the memory checkpoint (e.g., "After model load")
        device (torch.device): Specific GPU device to monitor. If None, uses current device.
        return_stats (bool): If True, returns memory statistics as dictionary instead of printing.
        
    Returns:
        dict or None: If return_stats=True, returns dictionary with keys:
            - alloc_gb: Allocated memory in GB
            - reserv_gb: Reserved memory in GB  
            - total_gb: Total device memory in GB
            - free_est_gb: Estimated free memory in GB
    
    Example:
        >>> gpu_stats("After loading model")
        [GPU 0] After loading model              | Alloc:    2.45 GB | Reserved:    3.20 GB | Total:   24.00 GB | Free(est):   21.55 GB
        
        >>> stats = gpu_stats("Checkpoint", return_stats=True)
        >>> print(stats['alloc_gb'])
        2.45
    """
    if not torch.cuda.is_available():
        if is_main_process_per_node():
            logger.warning("CUDA is not available - cannot monitor GPU memory")
        return None if not return_stats else {"alloc_gb": 0, "reserv_gb": 0, "total_gb": 0, "free_est_gb": 0}
    
    torch.cuda.synchronize()
    if device is None:
        device = torch.cuda.current_device()
    idx = device if isinstance(device, int) else device.index
    
    alloc = torch.cuda.memory_allocated(idx) / 1024**3
    reserv = torch.cuda.memory_reserved(idx) / 1024**3
    total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    free_est = total - alloc
    
    if not return_stats and is_main_process_per_node():
        msg = (
            f"[GPU {idx}] {label:30s} | "
            f"Alloc: {alloc:7.2f} GB | Reserved: {reserv:7.2f} GB | "
            f"Total: {total:7.2f} GB | Free(est): {free_est:7.2f} GB"
        )
        logger.info(msg)
    
    return {"alloc_gb": alloc, "reserv_gb": reserv, "total_gb": total, "free_est_gb": free_est}


def _report_peak_memory(label: str):
    """Report peak GPU memory usage for all ranks, then reset the peak stats.

    Uses ``dist.gather_object`` on the **node-local** process group (Gloo-backed
    under the hood for object collectives) instead of ``dist.all_gather`` on the
    default (NCCL) group.  This avoids deadlocks caused by mixing NCCL
    point-to-point pipeline ops with NCCL collectives on the same communicator.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    local = torch.cuda.current_device()
    peak_bytes = torch.cuda.max_memory_allocated(local)
    peak_gb = peak_bytes / 1024**3
    current_bytes = torch.cuda.memory_allocated(local)
    current_gb = current_bytes / 1024**3

    if dist.is_initialized():
        from utils.myparallel import get_pipeline_config, is_main_process_per_node
        from utils import myparallel as _myparallel_mod
        parallel_cfg = get_pipeline_config()
        node_group = parallel_cfg.get("node_process_group", None)
        total_gpus = parallel_cfg.get("total_gpus", 8)

        # If PP config has no node_group, try TP config (TP mode)
        _tp_cfg = _myparallel_mod._tp_config
        if node_group is None and _tp_cfg is not None:
            node_group = _tp_cfg.get("node_process_group", None)
            total_gpus = _tp_cfg.get("total_gpus", 8)

        if node_group is not None:
            # gather_object uses the Gloo backend internally for object
            # collectives, so it does NOT conflict with pending NCCL p2p ops.
            dst_global = dist.get_global_rank(node_group, 0)
            # Use tp_rank for TP mode, stage for PP mode
            _stage_or_tp = parallel_cfg.get("stage", 0)
            if _tp_cfg is not None:
                _stage_or_tp = _tp_cfg.get("tp_rank", 0)
            local_info = {"rank": dist.get_rank(), "stage": _stage_or_tp, "peak_gb": peak_gb, "current_gb": current_gb}
            if is_main_process_per_node():
                gathered = [None] * total_gpus
            else:
                gathered = None
            dist.gather_object(local_info, gathered, dst=dst_global, group=node_group)

            if is_main_process_per_node() and gathered is not None:
                # Sort by rank for deterministic output
                gathered = sorted([g for g in gathered if g is not None], key=lambda g: g["rank"])
                # Write peak memory info to gpu_memory debug logger
                _mem_logger = logging.getLogger("debug.gpu_memory")
                _mem_logger.info(f"{'='*70}")
                _mem_logger.info(f"  Peak GPU Memory — {label}")
                _mem_logger.info(f"{'='*70}")
                max_peak = 0.0
                _rank_label = "tp_rank" if _tp_cfg is not None else "stage"
                for g in gathered:
                    p = g["peak_gb"]
                    c = g["current_gb"]
                    max_peak = max(max_peak, p)
                    _mem_logger.info(f"  GPU {g['rank']} ({_rank_label} {g['stage']}): Peak = {p:7.2f} GB | Current = {c:7.2f} GB")
                _mem_logger.info(f"  MAX across node GPUs: {max_peak:.2f} GB")
                _mem_logger.info(f"{'='*70}")
        else:
            # Fallback: just log local info
            if is_main_process_per_node():
                logger.info(f"[GPU {local}] Peak Memory ({label}): {peak_gb:.2f} GB")
    else:
        if is_main_process_per_node():
            logger.info(f"[GPU {local}] Peak Memory ({label}): {peak_gb:.2f} GB")

    # Reset peak stats for next measurement
    torch.cuda.reset_peak_memory_stats(local)


def all_gpu_stats(label: str = ""):
    """
    Gather and print VRAM usage from all ranks in distributed training.
    
    This function collects memory statistics from all GPUs across all processes
    and prints a consolidated view on rank 0. Essential for monitoring memory
    usage in pipeline parallel and data parallel setups.
    
    Args:
        label (str): Descriptive label for the memory checkpoint
        
    Behavior:
        - If distributed is not initialized, falls back to single GPU monitoring
        - Rank 0 gathers and prints statistics from all ranks
        - Other ranks only contribute their local statistics
        
    Example output on rank 0:
        ======================================================================
          After forward pass
        ======================================================================
          GPU 0: Alloc=   12.45 GB | Reserved=   15.20 GB | Total=   24.00 GB | Free(est)=   11.55 GB
          GPU 1: Alloc=   10.23 GB | Reserved=   13.50 GB | Total=   24.00 GB | Free(est)=   13.77 GB
          TOTAL Alloc: 22.68 GB
        ======================================================================
    """
    if not dist.is_initialized():
        if is_main_process_per_node():
            logger.warning("Distributed not initialized - using single GPU monitoring")
        gpu_stats(label)
        return

    try:
        local = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(local) / 1024**3
        reserv = torch.cuda.memory_reserved(local) / 1024**3
        total = torch.cuda.get_device_properties(local).total_memory / 1024**3

        # Create tensor with memory info for gathering
        info = torch.tensor([alloc, reserv, total], dtype=torch.float64, device=local)
        gathered = [torch.zeros_like(info) for _ in range(dist.get_world_size())]
        
        # Use non-blocking all_gather with timeout to avoid hanging
        dist.all_gather(gathered, info)

        # Print on local rank 0 of every node
        if is_main_process_per_node():
            logger.info(f"\n{'='*70}")
            logger.info(f"  {label}")
            logger.info(f"{'='*70}")
            total_alloc = 0.0
            for i, g in enumerate(gathered):
                a, r, t = g[0].item(), g[1].item(), g[2].item()
                total_alloc += a
                logger.info(
                    f"  GPU {i}: Alloc={a:7.2f} GB | Reserved={r:7.2f} GB | "
                    f"Total={t:7.2f} GB | Free(est)={t - a:7.2f} GB"
                )
            logger.info(f"  TOTAL Alloc: {total_alloc:.2f} GB")
            logger.info(f"{'='*70}\n")
    
    except Exception as e:
        # If distributed communication fails, fall back to local monitoring
        if is_main_process_per_node():
            logger.warning(f"Distributed GPU stats failed: {e}. Falling back to local monitoring.")
        gpu_stats(f"{label} (local only)")


# =================== Cross-Node Bandwidth Test =============================

import time
import os


def _measure_pair_bandwidth(
    group: dist.ProcessGroup,
    device: torch.device,
    size_mb: int = 64,
    num_warmup: int = 3,
    num_iters: int = 10,
) -> float:
    """
    Measure bandwidth between two ranks using all_reduce on a 2-rank sub-group.

    The all_reduce on a 2-rank group effectively sends data in both directions.
    The measured bandwidth is: payload_bytes * num_iters / elapsed_time.

    Args:
        group: A torch.distributed ProcessGroup containing exactly 2 ranks.
        device: Local CUDA device.
        size_mb: Payload size in MB.
        num_warmup: Number of warmup iterations (not timed).
        num_iters: Number of timed iterations.

    Returns:
        Measured bandwidth in GB/s.
    """
    num_floats = size_mb * 1024 * 1024 // 4  # float32
    tensor = torch.randn(num_floats, device=device)

    # Warmup
    for _ in range(num_warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # all_reduce on 2 ranks: each rank sends size_mb to the other
    bandwidth_gbs = num_iters * size_mb / elapsed / 1024  # GB/s
    del tensor
    return bandwidth_gbs


def cross_node_bandwidth_test(
    size_mb: int = 64,
    num_warmup: int = 3,
    num_iters: int = 10,
):
    """
    Measure and print a bandwidth matrix across all global ranks.

    Must be called by **all** ranks (collective). The result matrix is
    printed on rank 0 with annotations showing which transfers are
    intra-node vs inter-node.

    Uses all_reduce on 2-rank sub-groups to measure pairwise bandwidth.
    All ranks participate in creating sub-groups (required by NCCL), but
    only the two ranks in each sub-group perform the actual data transfer.

    Args:
        size_mb: Payload size per transfer in MB.
        num_warmup: Warmup iterations (not timed).
        num_iters: Timed iterations.
    """
    if not dist.is_initialized():
        if is_main_process():
            logger.warning("Distributed not initialized — skipping cross-node bandwidth test")
        return

    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))

    if my_rank == 0:
        logger.info(f"\n{'='*70}")
        logger.info(f"  Cross-Node Bandwidth Test")
        logger.info(f"  world_size={world_size}  gpus_per_node={gpus_per_node}  "
                     f"payload={size_mb} MB  iters={num_iters}")
        logger.info(f"{'='*70}")

    # Collect node_id for each rank
    node_id = my_rank // gpus_per_node
    node_ids_tensor = torch.zeros(world_size, dtype=torch.long, device=device)
    node_ids_tensor[my_rank] = node_id
    dist.all_reduce(node_ids_tensor, op=dist.ReduceOp.SUM)
    node_ids = node_ids_tensor.cpu().tolist()

    # --- Decide which pairs to test ---
    # For small world_size (<=16), test all unique pairs.
    # For larger, test representative pairs.
    if world_size <= 16:
        pairs = [(i, j) for i in range(world_size) for j in range(i + 1, world_size)]
    else:
        pairs_set = set()
        num_nodes = world_size // gpus_per_node
        for n in range(num_nodes):
            base = n * gpus_per_node
            # Intra-node: first <-> second, first <-> last
            if gpus_per_node > 1:
                pairs_set.add((base, base + 1))
                pairs_set.add((base, base + gpus_per_node - 1))
            # Inter-node: node 0 rank 0 <-> this node rank 0
            if n > 0:
                pairs_set.add((0, base))
                pairs_set.add((gpus_per_node - 1, base))
        pairs = sorted(pairs_set)

    # --- Pre-create all sub-groups (NCCL requires all ranks to participate) ---
    if my_rank == 0:
        logger.info(f"  Creating {len(pairs)} sub-groups for pairwise tests...")

    pair_groups = {}
    for (r1, r2) in pairs:
        group = dist.new_group(ranks=[r1, r2])
        pair_groups[(r1, r2)] = group

    dist.barrier()

    # --- Run bandwidth tests ---
    results = {}  # (r1, r2) -> GB/s
    total_tests = len(pairs)
    for idx, (r1, r2) in enumerate(pairs):
        if my_rank == 0:
            n1, n2 = node_ids[r1], node_ids[r2]
            link_type = "intra-node" if n1 == n2 else "INTER-NODE"
            logger.info(f"  [{idx+1}/{total_tests}] Rank {r1}(N{n1}) <-> Rank {r2}(N{n2})  ({link_type})")

        group = pair_groups[(r1, r2)]
        if my_rank == r1 or my_rank == r2:
            bw = _measure_pair_bandwidth(group, device, size_mb, num_warmup, num_iters)
        else:
            bw = 0.0

        # Gather the result to rank 0
        bw_tensor = torch.tensor([bw], dtype=torch.float64, device=device)
        dist.broadcast(bw_tensor, src=r1)
        results[(r1, r2)] = bw_tensor.item()
        # Symmetric: all_reduce measures bidirectional
        results[(r2, r1)] = bw_tensor.item()

        dist.barrier()

    # --- Print results on rank 0 ---
    if my_rank == 0:
        num_nodes = world_size // gpus_per_node

        # 1. Print full matrix if small enough
        if world_size <= 16:
            col_w = 10
            header = "Src\\Dst".ljust(col_w) + "".join(
                f"R{d}".rjust(col_w) for d in range(world_size)
            )
            sep = "-" * len(header)
            logger.info(f"\n  === Bandwidth Matrix (GB/s, bidirectional all_reduce) ===")
            logger.info(f"  (R = global rank, N = node id)")
            logger.info(f"  {header}")
            logger.info(f"  {sep}")
            for src in range(world_size):
                row = f"R{src}(N{node_ids[src]})".ljust(col_w)
                for dst in range(world_size):
                    if src == dst:
                        row += "---".rjust(col_w)
                    elif (src, dst) in results:
                        row += f"{results[(src,dst)]:.2f}".rjust(col_w)
                    else:
                        row += "n/a".rjust(col_w)
                logger.info(f"  {row}")
            logger.info(f"  {sep}")

        # 2. Print summary: intra-node vs inter-node
        # Use only unique pairs to avoid double counting
        intra_bws = [results[(r1,r2)] for (r1,r2) in pairs if node_ids[r1] == node_ids[r2]]
        inter_bws = [results[(r1,r2)] for (r1,r2) in pairs if node_ids[r1] != node_ids[r2]]

        logger.info(f"\n  === Bandwidth Summary ===")
        if intra_bws:
            logger.info(f"  Intra-node (same node):  "
                         f"min={min(intra_bws):.2f}  max={max(intra_bws):.2f}  "
                         f"avg={sum(intra_bws)/len(intra_bws):.2f} GB/s  "
                         f"({len(intra_bws)} pairs tested)")
        if inter_bws:
            logger.info(f"  Inter-node (cross node): "
                         f"min={min(inter_bws):.2f}  max={max(inter_bws):.2f}  "
                         f"avg={sum(inter_bws)/len(inter_bws):.2f} GB/s  "
                         f"({len(inter_bws)} pairs tested)")
        if intra_bws and inter_bws:
            ratio = (sum(intra_bws)/len(intra_bws)) / (sum(inter_bws)/len(inter_bws))
            logger.info(f"  Intra/Inter ratio: {ratio:.1f}x")

        # 3. Print per-node-pair summary for multi-node
        if num_nodes > 1:
            logger.info(f"\n  === Per Node-Pair Bandwidth ===")
            for n1 in range(num_nodes):
                for n2 in range(n1 + 1, num_nodes):
                    pair_bws = [
                        results[(r1, r2)] for (r1, r2) in pairs
                        if (node_ids[r1] == n1 and node_ids[r2] == n2)
                        or (node_ids[r1] == n2 and node_ids[r2] == n1)
                    ]
                    if pair_bws:
                        logger.info(
                            f"  Node {n1} <-> Node {n2}: "
                            f"avg={sum(pair_bws)/len(pair_bws):.2f} GB/s  "
                            f"({len(pair_bws)} pairs)"
                        )

        logger.info(f"{'='*70}\n")

    # Cleanup sub-groups
    for group in pair_groups.values():
        dist.destroy_process_group(group)

    # Final barrier
    dist.barrier()


if __name__ == "__main__":
    # Simple test when run directly
    print("GPU Memory Monitoring Utilities")
    print("Available functions:")
    print("- gpu_stats(label, device, return_stats)")
    print("- all_gpu_stats(label)")
    print("- cross_node_bandwidth_test(size_mb, num_warmup, num_iters)")
    print("\nSee docstrings for detailed usage information.")