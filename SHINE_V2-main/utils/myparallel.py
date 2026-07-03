#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Parallel Utilities for Pipeline + Data Parallel Training

This module provides all utilities needed for distributed training with
pipeline parallelism (intra-node) and data parallelism (inter-node).

Architecture (example: 2 nodes × 8 GPUs, pipeline_parallel_size=4):
  Node 0:  GPU 0-3 = PP group 0 (DP rank 0),  GPU 4-7 = PP group 1 (DP rank 0)
  Node 1:  GPU 0-3 = PP group 0 (DP rank 1),  GPU 4-7 = PP group 1 (DP rank 1)

Key Concepts:
  - global_rank:  unique rank across all processes (set by torchrun)
  - local_rank:   rank within the current node (0..gpus_per_node-1)
  - pp_stage:     pipeline stage index (0..pipeline_parallel_size-1)
  - dp_rank:      data-parallel rank (processes with same pp_stage share a DP group)

Usage:
    from utils.myparallel import (
        init_distributed, cleanup_distributed,
        setup_pipeline_parallel, get_pipeline_config,
        get_rank, get_local_rank, get_world_size,
        is_main_process, is_first_stage, is_last_stage,
        is_node0,
        barrier, distributed_mean,
    )
"""

import os
import torch
import torch.distributed as dist
import datetime as dt
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state: populated by setup_pipeline_parallel()
# ---------------------------------------------------------------------------
_pipeline_config: Optional[Dict] = None
# Populated by setup_tensor_parallel() when running the TP path.
_tp_config: Optional[Dict] = None


# ========================== Distributed Basics =============================

def _dist_is_active() -> bool:
    """Check if torch.distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def _should_use_dist() -> bool:
    """Check if we are in a multi-process launch (torchrun sets WORLD_SIZE)."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    """Global rank across all processes."""
    return dist.get_rank() if _dist_is_active() else 0


def get_local_rank() -> int:
    """Rank within the current node (torchrun sets LOCAL_RANK)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """Total number of processes across all nodes."""
    return dist.get_world_size() if _dist_is_active() else 1


def is_main_process() -> bool:
    """True only on global rank 0 — use for logging / checkpointing."""
    return get_rank() == 0


def is_main_process_per_node() -> bool:
    """True on local rank 0 of every node — use for per-node logging."""
    return get_local_rank() == 0


def is_node0() -> bool:
    """True if the current process belongs to node 0.

    In a multi-node setup with ``total_gpus`` GPUs per node, node 0 owns
    global ranks ``0 .. total_gpus-1``.
    """
    parallel_cfg = get_pipeline_config()
    total_gpus = parallel_cfg.get("total_gpus", 1)
    return get_rank() < total_gpus


def barrier():
    """Block until all processes reach this point."""
    if _dist_is_active():
        dist.barrier()



# ======================== Init / Cleanup ===================================

def init_distributed(timeout_minutes: int = 10):
    """
    Initialize the default process group (NCCL / Gloo).

    Safe to call even in single-GPU mode — it will be a no-op.
    Also silences stdout on non-zero ranks to keep logs clean.
    """
    if _should_use_dist() and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=dt.timedelta(minutes=timeout_minutes),
        )
        # Silence non-local-rank-0 processes (keep first GPU per node verbose)
        if not is_main_process_per_node():
            import builtins
            builtins.print = lambda *a, **kw: None


def cleanup_distributed():
    """Destroy the process group (barrier first to avoid NCCL errors)."""
    if _dist_is_active():
        dist.barrier()
        dist.destroy_process_group()


# Keep old names as aliases for backward compatibility
ddp_init_if_needed = init_distributed
ddp_cleanup_if_needed = cleanup_distributed


# ====================== Pipeline Parallel Setup ============================

def setup_pipeline_parallel(
    total_gpus: int,
    pipeline_parallel_size: int,
) -> Dict:
    """
    Compute and store pipeline-parallel + data-parallel topology.

    Args:
        total_gpus: Number of GPUs **per node** (typically 8).
        pipeline_parallel_size: Number of pipeline stages that fit on one node.
            Must evenly divide *total_gpus*.

    Returns:
        Dict with keys:
            stage              – pipeline stage index (0-based)
            total_stages       – == pipeline_parallel_size
            device             – torch.device for this rank
            is_first           – True if stage == 0
            is_last            – True if stage == total_stages - 1
            data_parallel_rank – rank inside the data-parallel group
            data_parallel_size – number of data-parallel replicas
            dp_process_group   – torch ProcessGroup for data-parallel allreduce
                                 (None when DP size == 1)

    Raises:
        ValueError: if pipeline_parallel_size does not divide total_gpus.
    """
    global _pipeline_config

    if total_gpus % pipeline_parallel_size != 0:
        raise ValueError(
            f"pipeline_parallel_size ({pipeline_parallel_size}) must evenly "
            f"divide total_gpus per node ({total_gpus})"
        )

    local_rank = get_local_rank()
    global_rank = get_rank()
    world_size = get_world_size()

    # --- Pipeline stage is determined by local_rank within the node ---
    # e.g. with pp_size=4 and 8 GPUs/node:
    #   local_rank 0..3 → stage 0..3 (PP group 0)
    #   local_rank 4..7 → stage 0..3 (PP group 1)
    stage = local_rank % pipeline_parallel_size

    # Number of PP groups per node
    pp_groups_per_node = total_gpus // pipeline_parallel_size

    # PP group index within the node
    pp_group_in_node = local_rank // pipeline_parallel_size

    # Total number of PP groups across all nodes
    num_nodes = world_size // total_gpus if total_gpus > 0 else 1
    total_pp_groups = pp_groups_per_node * num_nodes

    # Data-parallel rank: which replica of the same stage this process is
    # Processes with the same stage across different PP groups form a DP group
    node_rank = global_rank // total_gpus
    data_parallel_rank = node_rank * pp_groups_per_node + pp_group_in_node
    data_parallel_size = total_pp_groups

    # Device is always the local GPU
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- Build data-parallel process groups ---
    # All ranks that share the same pipeline stage form one DP group.
    dp_process_group = None
    if _dist_is_active() and data_parallel_size > 1:
        # We need to create groups for every stage; all ranks must participate
        # in new_group() calls even if they don't belong to that group.
        for s in range(pipeline_parallel_size):
            # Collect global ranks that have this stage
            ranks_for_stage = []
            for n in range(num_nodes):
                for g in range(pp_groups_per_node):
                    r = n * total_gpus + g * pipeline_parallel_size + s
                    ranks_for_stage.append(r)
            group = dist.new_group(ranks=ranks_for_stage)
            if stage == s:
                dp_process_group = group

    # --- Build intra-node process group ---
    # All GPUs on the same node form one group.  Used for node-local
    # gather operations (e.g. debug anchor info).
    node_process_group = None
    if _dist_is_active():
        for n in range(num_nodes):
            node_ranks = list(range(n * total_gpus, (n + 1) * total_gpus))
            grp = dist.new_group(ranks=node_ranks)
            if node_rank == n:
                node_process_group = grp

    # Store pipeline environment variable for get_pipeline_config() fallback
    os.environ["PIPELINE_PARALLEL_SIZE"] = str(pipeline_parallel_size)

    _pipeline_config = {
        "stage": stage,
        "total_stages": pipeline_parallel_size,
        "device": device,
        "is_first": stage == 0,
        "is_last": stage == pipeline_parallel_size - 1,
        "data_parallel_rank": data_parallel_rank,
        "data_parallel_size": data_parallel_size,
        "dp_process_group": dp_process_group,
        "node_process_group": node_process_group,
        "total_gpus": total_gpus,
    }

    if is_main_process_per_node():
        logger.info(
            f"Pipeline topology: {pipeline_parallel_size} stages × "
            f"{data_parallel_size} DP replicas  "
            f"(world_size={world_size}, gpus_per_node={total_gpus})"
        )

    return _pipeline_config


def get_pipeline_config() -> Dict:
    """
    Return the pipeline config set by setup_pipeline_parallel().

    If setup has not been called yet, returns a sensible single-stage default.
    """
    if _pipeline_config is not None:
        return _pipeline_config

    # Fallback: no pipeline parallelism
    pp_size = int(os.environ.get("PIPELINE_PARALLEL_SIZE", "1"))
    local_rank = get_local_rank()
    stage = local_rank % pp_size

    return {
        "stage": stage,
        "total_stages": pp_size,
        "device": torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu"),
        "is_first": stage == 0,
        "is_last": stage == pp_size - 1,
        "data_parallel_rank": 0,
        "data_parallel_size": 1,
        "dp_process_group": None,
    }


# Keep old name as alias
get_pipeline_info = get_pipeline_config


# ====================== Tensor Parallel Setup ==============================
#
# PP and TP are mutually exclusive intra-node parallelism schemes; only one
# of setup_pipeline_parallel / setup_tensor_parallel is called per run.
# TP topology mirrors PP for symmetry: contiguous global ranks
# [base, base+tp_size) form one TP group; ranks at the same offset within
# different TP groups form one DP group.
# ---------------------------------------------------------------------------

def setup_tensor_parallel(
    total_gpus: int,
    tensor_parallel_size: int,
    sequence_parallel_size: int = 1,
) -> Dict:
    """
    Compute and store tensor-parallel + sequence-parallel + data-parallel topology.

    GPU layout (within a node):
        - The ``total_gpus`` GPUs are first divided into groups of size
          ``tp_size * sp_size``. Within each such group:
            * TP group: ``tp_size`` consecutive GPUs share the same SP rank.
            * SP group: ``sp_size`` GPUs with the same TP rank across TP groups.
        - Remaining groups across nodes form the DP dimension.

    Constraint: ``tp_size * sp_size`` must divide ``total_gpus``.
    Recommendation: ``tp_size * sp_size <= gpus_per_node`` (avoid cross-node P2P).

    Args:
        total_gpus: Number of GPUs per node.
        tensor_parallel_size: TP shards per group. Must divide total_gpus.
        sequence_parallel_size: SP shards per group (default 1 = no SP).
            Must satisfy ``tp_size * sp_size`` divides ``total_gpus``.

    Returns:
        Dict (also stored in module-global _tp_config) with keys:
            tp_rank, tensor_parallel_size, device,
            sp_rank, sequence_parallel_size, sp_process_group,
            data_parallel_rank, data_parallel_size,
            tp_process_group, dp_process_group, node_process_group,
            total_gpus
    """
    global _tp_config

    tp_size = tensor_parallel_size
    sp_size = sequence_parallel_size

    tp_sp_size = tp_size * sp_size
    if total_gpus % tp_sp_size != 0:
        raise ValueError(
            f"tp_size * sp_size ({tp_size} * {sp_size} = {tp_sp_size}) must evenly "
            f"divide total_gpus per node ({total_gpus})"
        )

    local_rank = get_local_rank()
    global_rank = get_rank()
    world_size = get_world_size()

    # Within a TP×SP group of tp_sp_size GPUs:
    #   rank_in_group = local_rank % tp_sp_size
    #   tp_rank = rank_in_group % tp_size
    #   sp_rank = rank_in_group // tp_size
    tp_rank = local_rank % tp_size
    sp_rank = (local_rank % tp_sp_size) // tp_size

    tp_sp_groups_per_node = total_gpus // tp_sp_size
    tp_sp_group_in_node = local_rank // tp_sp_size

    num_nodes = world_size // total_gpus if total_gpus > 0 else 1
    total_tp_sp_groups = tp_sp_groups_per_node * num_nodes

    node_rank = global_rank // total_gpus
    data_parallel_rank = node_rank * tp_sp_groups_per_node + tp_sp_group_in_node
    data_parallel_size = total_tp_sp_groups

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- TP process groups ---
    # Each TP group: tp_size consecutive ranks within a TP×SP group
    # that share the same sp_rank.
    tp_process_group = None
    if _dist_is_active() and tp_size > 1:
        # Total number of TP groups = total_tp_sp_groups * sp_size
        num_tp_groups = (world_size // tp_sp_size) * sp_size
        for g_idx in range(world_size // tp_size):
            # Enumerate all TP groups across the entire world
            # TP group g_idx contains ranks:
            #   base_of_tp_sp_group + sp_r * tp_size + [0..tp_size-1]
            tp_sp_group_idx = g_idx // sp_size
            sp_r = g_idx % sp_size
            base = tp_sp_group_idx * tp_sp_size + sp_r * tp_size
            ranks_for_tp = list(range(base, base + tp_size))
            group = dist.new_group(ranks=ranks_for_tp)
            if global_rank in ranks_for_tp:
                tp_process_group = group

    # --- SP process groups ---
    # Each SP group: sp_size ranks with the same tp_rank across sp positions
    # within the same TP×SP group.
    sp_process_group = None
    if _dist_is_active() and sp_size > 1:
        for g_idx in range(world_size // tp_sp_size):
            for tp_r in range(tp_size):
                base = g_idx * tp_sp_size
                ranks_for_sp = [base + sp_r * tp_size + tp_r for sp_r in range(sp_size)]
                group = dist.new_group(ranks=ranks_for_sp)
                if global_rank in ranks_for_sp:
                    sp_process_group = group

    # --- DP process groups ---
    # Each DP group: ranks at the same position within their TP×SP group,
    # across all TP×SP groups (across nodes).
    dp_process_group = None
    if _dist_is_active() and data_parallel_size > 1:
        for pos in range(tp_sp_size):
            ranks_for_dp = []
            for n in range(num_nodes):
                for g in range(tp_sp_groups_per_node):
                    r = n * total_gpus + g * tp_sp_size + pos
                    ranks_for_dp.append(r)
            group = dist.new_group(ranks=ranks_for_dp)
            if (local_rank % tp_sp_size) == pos:
                dp_process_group = group

    # --- DP+SP process groups (for ZeRO-1 sharding across SP+DP ranks) ---
    # Each DP+SP group: all ranks that share the same tp_rank within their
    # TP×SP group, across all TP×SP groups (nodes) AND across SP positions.
    # This allows ZeRO-1 to shard optimizer state across both SP and DP ranks
    # since they all hold identical parameters.
    dp_sp_process_group = None
    if _dist_is_active() and (data_parallel_size * sp_size) > 1:
        for tp_r in range(tp_size):
            ranks_for_dp_sp = []
            for n in range(num_nodes):
                for g in range(tp_sp_groups_per_node):
                    # For each TP×SP group, collect all SP ranks with this tp_rank
                    base = n * total_gpus + g * tp_sp_size
                    for sp_r in range(sp_size):
                        r = base + sp_r * tp_size + tp_r
                        ranks_for_dp_sp.append(r)
            group = dist.new_group(ranks=ranks_for_dp_sp)
            if tp_rank == tp_r:
                dp_sp_process_group = group

    node_process_group = None
    if _dist_is_active():
        for n in range(num_nodes):
            node_ranks = list(range(n * total_gpus, (n + 1) * total_gpus))
            grp = dist.new_group(ranks=node_ranks)
            if node_rank == n:
                node_process_group = grp

    os.environ["TENSOR_PARALLEL_SIZE"] = str(tp_size)
    os.environ["SEQUENCE_PARALLEL_SIZE"] = str(sp_size)

    _tp_config = {
        "tp_rank": tp_rank,
        "tensor_parallel_size": tp_size,
        "sp_rank": sp_rank,
        "sequence_parallel_size": sp_size,
        "device": device,
        "data_parallel_rank": data_parallel_rank,
        "data_parallel_size": data_parallel_size,
        "tp_process_group": tp_process_group,
        "sp_process_group": sp_process_group,
        "dp_process_group": dp_process_group,
        "dp_sp_process_group": dp_sp_process_group,
        "node_process_group": node_process_group,
        "total_gpus": total_gpus,
    }

    if is_main_process_per_node():
        logger.info(
            f"Tensor-parallel topology: TP={tp_size} x SP={sp_size} x "
            f"DP={data_parallel_size}  "
            f"(world_size={world_size}, gpus_per_node={total_gpus})"
        )

    return _tp_config


def get_tp_config() -> Dict:
    if _tp_config is None:
        raise RuntimeError(
            "setup_tensor_parallel() must be called before get_tp_config()."
        )
    return _tp_config


def get_tp_rank() -> int:
    return get_tp_config()["tp_rank"]


def get_tp_world_size() -> int:
    return get_tp_config()["tensor_parallel_size"]


def get_tp_process_group():
    return get_tp_config().get("tp_process_group")


# ===================== Pipeline Stage Queries ==============================

def is_first_stage() -> bool:
    """True if this process is pipeline stage 0."""
    return get_pipeline_config()["is_first"]


def is_last_stage() -> bool:
    """True if this process is the last pipeline stage."""
    return get_pipeline_config()["is_last"]


def get_pipeline_stage() -> int:
    """Current pipeline stage index (0-based)."""
    return get_pipeline_config()["stage"]


def get_total_pipeline_stages() -> int:
    """Total number of pipeline stages."""
    return get_pipeline_config()["total_stages"]


# ===================== Device Mapping ======================================

def setup_device_mapping(pipeline_parallel_size: int) -> List[torch.device]:
    """
    Return a list of devices for each pipeline stage on the current node.

    Args:
        pipeline_parallel_size: Number of pipeline stages per group.

    Returns:
        List of torch.device objects.
    """
    base = get_local_rank() - (get_local_rank() % pipeline_parallel_size)
    return [torch.device(f"cuda:{base + s}") for s in range(pipeline_parallel_size)]


# =================== Micro-batch Scheduling ================================

def get_micro_batch_schedule(
    total_micro_batches: int,
    pipeline_stages: int,
) -> List[Dict]:
    """
    Generate a 1F1B-style micro-batch schedule for pipeline parallelism.

    Args:
        total_micro_batches: Number of micro-batches per global batch.
        pipeline_stages: Number of pipeline stages.

    Returns:
        List of dicts mapping stage → micro_batch_index for each time step.
    """
    schedule = []
    for t in range(total_micro_batches + pipeline_stages - 1):
        step = {}
        for s in range(pipeline_stages):
            mb = t - s
            if 0 <= mb < total_micro_batches:
                step[s] = mb
        schedule.append(step)
    return schedule


# =================== Collective Helpers ====================================

@torch.no_grad()
def distributed_mean(
    value: float,
    device: torch.device,
    group: Optional[dist.ProcessGroup] = None,
) -> float:
    """
    Average a scalar across processes.

    Args:
        value: Local scalar value.
        device: Device to create the tensor on.
        group: Process group to reduce over.  Defaults to the data-parallel
               group if pipeline parallelism is active, otherwise WORLD.

    Returns:
        The mean value across the group.
    """
    if not _dist_is_active():
        return value

    # Default to DP group so that pipeline stages don't pollute the mean
    if group is None:
        cfg = get_pipeline_config()
        group = cfg.get("dp_process_group")

    t = torch.tensor([value], dtype=torch.float32, device=device)
    if group is not None:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        t /= dist.get_world_size(group)
    else:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    return float(t.item())


@torch.no_grad()
def sync_gradients_across_dp(
    model: torch.nn.Module,
    device: torch.device,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Average **in-place** the ``.grad`` of every trainable parameter on
    *device* across the data-parallel group.

    This must be called **after** backward and **before** optimizer.step()
    so that all DP replicas apply the same averaged gradient.

    Uses ``ReduceOp.AVG`` (requires NCCL ≥ 2.10).

    Args:
        model:  The module whose parameters should be synchronised.
                Only parameters satisfying ``requires_grad``,
                ``grad is not None``, and ``param.device == device``
                are touched.
        device: The current pipeline-stage device.  Parameters on other
                devices (e.g. ``meta``) are skipped.
        group:  The data-parallel process group.  If ``None``, defaults
                to the DP group stored in the pipeline config.  When the
                DP size is 1 (single-node) the function is a no-op.
    """
    if not _dist_is_active():
        return

    if group is None:
        cfg = get_pipeline_config()
        group = cfg.get("dp_process_group")

    if group is None:
        return  # DP size == 1, nothing to sync

    dp_size = dist.get_world_size(group)
    if dp_size <= 1:
        return

    for p in model.parameters():
        if p.requires_grad and p.grad is not None and p.device == device:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=group)


# =================== Pipeline Communication ================================
#
# This section provides pipeline-parallel communication primitives in four
# flavours, covering every combination of two axes:
#
#   Axis 1 – Blocking behaviour
#     • Blocking  (pipeline_send / pipeline_recv)
#     • Async     (pipeline_isend / pipeline_irecv)  → returns Work handle
#
#   Axis 2 – Autograd support
#     • No-grad   (pipeline_send / pipeline_recv / pipeline_isend / pipeline_irecv)
#     • With-grad (PipelineSend / PipelineRecv autograd.Functions)
#
# Summary table:
#
#   |              | Blocking          | Async (returns Work)  |
#   |--------------|-------------------|-----------------------|
#   | No gradient  | pipeline_send     | pipeline_isend        |
#   |              | pipeline_recv     | pipeline_irecv        |
#   | With gradient| PipelineSend.apply| PipelineAsyncSend.apply|
#   |              | PipelineRecv.apply| PipelineAsyncRecv.apply|
#
# =========================================================================

# --------------- Internal helper ------------------------------------------

def _get_pp_peer_rank(peer_stage: int) -> int:
    """
    Return the **global rank** of the process that holds *peer_stage* in the
    same pipeline-parallel group as the current process.

    Within one PP group the global ranks are contiguous:
        base = (global_rank // pp_size) * pp_size
        peer  = base + peer_stage
    """
    cfg = get_pipeline_config()
    pp_size = cfg["total_stages"]
    global_rank = get_rank()
    base = (global_rank // pp_size) * pp_size
    return base + peer_stage


# =========================================================================
# 1. No-grad primitives (blocking)
# =========================================================================

def pipeline_send(tensor: torch.Tensor, dst_stage: int, tag: int = 0) -> None:
    """
    Blocking send — no autograd.

    Sends *tensor* to *dst_stage* in the same PP group.
    Returns only after the data transfer is complete.

    Args:
        tensor: Tensor to send (must be on the current CUDA device).
        dst_stage: Destination pipeline stage index (0-based).
        tag: Message tag for disambiguation.
    """
    dst_rank = _get_pp_peer_rank(dst_stage)
    dist.send(tensor.contiguous(), dst=dst_rank, tag=tag)


def pipeline_recv(tensor: torch.Tensor, src_stage: int, tag: int = 0) -> torch.Tensor:
    """
    Blocking recv — no autograd.

    Receives into a pre-allocated *tensor* from *src_stage*.
    Returns only after the data has been fully written.

    Args:
        tensor: Pre-allocated buffer (correct shape / dtype / device).
        src_stage: Source pipeline stage index (0-based).
        tag: Message tag for disambiguation.

    Returns:
        The same *tensor*, now filled with received data.
    """
    src_rank = _get_pp_peer_rank(src_stage)
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor


# =========================================================================
# 2. No-grad primitives (async / non-blocking)
# =========================================================================

def pipeline_isend(tensor: torch.Tensor, dst_stage: int, tag: int = 0):
    """
    Non-blocking send — no autograd.

    Returns a ``Work`` handle immediately.  The caller **must** call
    ``handle.wait()`` before modifying or freeing *tensor*.

    Args:
        tensor: Tensor to send.
        dst_stage: Destination pipeline stage index.
        tag: Message tag.

    Returns:
        ``torch.distributed.Work`` handle.
    """
    dst_rank = _get_pp_peer_rank(dst_stage)
    return dist.isend(tensor.contiguous(), dst=dst_rank, tag=tag)


def pipeline_irecv(tensor: torch.Tensor, src_stage: int, tag: int = 0):
    """
    Non-blocking recv — no autograd.

    Returns a ``Work`` handle immediately.  The caller **must** call
    ``handle.wait()`` before reading *tensor*.

    Args:
        tensor: Pre-allocated buffer.
        src_stage: Source pipeline stage index.
        tag: Message tag.

    Returns:
        ``torch.distributed.Work`` handle.
    """
    src_rank = _get_pp_peer_rank(src_stage)
    return dist.irecv(tensor, src=src_rank, tag=tag)


# =========================================================================
# 3. With-grad primitives (blocking)
#
#    These are ``torch.autograd.Function`` subclasses.  In the forward pass
#    they perform the same send/recv as above; in the backward pass they
#    do the **reverse** communication so that gradients flow back through
#    the pipeline.
#
#    Forward:  stage i  ──send──>  stage i+1
#    Backward: stage i  <──recv──  stage i+1   (gradient flows back)
# =========================================================================

class PipelineSend(torch.autograd.Function):
    """
    Blocking send **with autograd support**.

    Forward:  sends *input* to *dst_stage* (blocks until complete).
    Backward: receives grad from *dst_stage* (the reverse direction).

    Usage::

        output = PipelineSend.apply(hidden_states, dst_stage, tag)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, dst_stage: int, tag: int = 0) -> torch.Tensor:
        ctx.dst_stage = dst_stage
        ctx.tag = tag
        pipeline_send(input, dst_stage, tag)
        # Return an empty tensor (same shape/dtype) instead of input to avoid
        # creating a view relationship (output._base == input) that prevents
        # the input tensor from being freed after backward.
        output = torch.empty(
            input.shape, dtype=input.dtype, device=input.device,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # In backward the *dst_stage* sends us the gradient.
        grad_input = torch.empty(
            grad_output.shape, dtype=grad_output.dtype, device=grad_output.device,
        )
        pipeline_recv(grad_input, src_stage=ctx.dst_stage, tag=ctx.tag)
        # No gradient for dst_stage and tag (int args)
        return grad_input, None, None


class PipelineRecv(torch.autograd.Function):
    """
    Blocking recv **with autograd support**.

    Forward:  receives a tensor from *src_stage* into a new buffer.
    Backward: sends grad back to *src_stage* (the reverse direction).

    Usage::

        # On the receiving stage — returns a fresh tensor with grad_fn
        hidden_states = PipelineRecv.apply(
            placeholder,   # any small tensor on the correct device (for autograd)
            src_stage, shape, dtype, device, tag,
        )

    Note:
        *placeholder* is a dummy tensor whose only purpose is to give
        autograd a node to attach to.  It is **not** used for data.
        You can pass ``torch.empty(0, device=device, requires_grad=True)``.
    """

    @staticmethod
    def forward(
        ctx,
        placeholder: torch.Tensor,
        src_stage: int,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        tag: int = 0,
    ) -> torch.Tensor:
        ctx.src_stage = src_stage
        ctx.tag = tag
        buf = torch.empty(shape, dtype=dtype, device=device)
        pipeline_recv(buf, src_stage, tag)
        return buf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # In backward we send the gradient back to the source stage.
        # Ensure contiguous to avoid NCCL non-contiguous tensor warnings.
        pipeline_send(grad_output.contiguous(), dst_stage=ctx.src_stage, tag=ctx.tag)
        # No gradient for placeholder, src_stage, shape, dtype, device, tag
        return None, None, None, None, None, None


# =========================================================================
# 4. With-grad primitives (async / non-blocking)
#
#    Same autograd semantics as above, but the forward communication is
#    non-blocking.  The returned tensor carries a ``Work`` handle in
#    ``tensor._pp_work`` so you can call ``tensor._pp_work.wait()``
#    before using the data.
#
#    ⚠️  The backward pass is always **blocking** (simplicity & safety).
# =========================================================================

class PipelineAsyncSend(torch.autograd.Function):
    """
    Non-blocking send **with autograd support**.

    Forward:  issues an async send and returns the input immediately.
              The ``Work`` handle is stored as ``output._pp_work``.
    Backward: blocking recv of gradient from *dst_stage*.

    .. note::

        ``autograd.Function.apply()`` may return a new wrapper tensor that
        does **not** carry custom attributes set in ``forward()``.  Use the
        wrapper function :func:`pipeline_async_send_with_grad` which handles
        this transparently, or retrieve the handle via
        ``PipelineAsyncSend.get_last_work()``.
    """

    _last_work = None  # class-level fallback for the most recent Work handle

    @staticmethod
    def forward(ctx, input: torch.Tensor, dst_stage: int, tag: int = 0) -> torch.Tensor:
        ctx.dst_stage = dst_stage
        ctx.tag = tag
        work = pipeline_isend(input, dst_stage, tag)
        PipelineAsyncSend._last_work = work
        # Save a reference to input so it stays alive until the async send
        # completes (the caller will call work.wait() before dropping the
        # output tensor).  We store it on the output tensor itself.
        # Return an empty tensor (same shape/dtype) instead of input to avoid
        # creating a view relationship (output._base == input) that prevents
        # the input tensor from being freed after backward.
        output = torch.empty(
            input.shape, dtype=input.dtype, device=input.device,
        )
        output._pp_work = work
        output._pp_send_buf = input  # prevent GC of input until send completes
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Allocate a contiguous recv buffer.  torch.empty_like preserves
        # strides, so if grad_output is non-contiguous the buffer would
        # also be non-contiguous, triggering NCCL warnings.
        grad_input = torch.empty(
            grad_output.shape, dtype=grad_output.dtype, device=grad_output.device,
        )
        pipeline_recv(grad_input, src_stage=ctx.dst_stage, tag=ctx.tag)
        return grad_input, None, None

    @classmethod
    def get_last_work(cls):
        """Return the Work handle saved during the most recent forward()."""
        return cls._last_work


class PipelineAsyncRecv(torch.autograd.Function):
    """
    Non-blocking recv **with autograd support**.

    Forward:  issues an async recv and returns the buffer immediately.
              The ``Work`` handle is stored as ``output._pp_work``.
    Backward: blocking send of gradient back to *src_stage*.

    .. note::

        ``autograd.Function.apply()`` may return a new wrapper tensor that
        does **not** carry custom attributes set in ``forward()``.  Use the
        wrapper function :func:`pipeline_async_recv_with_grad` which handles
        this transparently, or retrieve the handle via
        ``PipelineAsyncRecv.get_last_work()``.
    """

    _last_work = None  # class-level fallback for the most recent Work handle

    @staticmethod
    def forward(
        ctx,
        placeholder: torch.Tensor,
        src_stage: int,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        tag: int = 0,
    ) -> torch.Tensor:
        ctx.src_stage = src_stage
        ctx.tag = tag
        buf = torch.empty(shape, dtype=dtype, device=device)
        work = pipeline_irecv(buf, src_stage, tag)
        PipelineAsyncRecv._last_work = work
        buf._pp_work = work
        return buf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Ensure contiguous to avoid NCCL non-contiguous tensor warnings.
        pipeline_send(grad_output.contiguous(), dst_stage=ctx.src_stage, tag=ctx.tag)
        return None, None, None, None, None, None

    @classmethod
    def get_last_work(cls):
        """Return the Work handle saved during the most recent forward()."""
        return cls._last_work


# =========================================================================
# 5. Convenience wrappers (recommended API)
#
#    These functions encapsulate the _pp_work fallback logic so that
#    callers never need to worry about autograd stripping custom attrs.
# =========================================================================

def pipeline_send_with_grad(
    tensor: torch.Tensor, dst_stage: int, tag: int = 0,
) -> torch.Tensor:
    """
    Blocking send with autograd.

    Wraps :class:`PipelineSend` for tensors connected to the autograd graph.
    For tensors without gradient, falls back to a plain blocking send.

    When the input tensor has no gradient (``requires_grad=False`` and no
    ``grad_fn``), the send is performed without autograd so that no
    backward anchor is needed.

    Args:
        tensor: Tensor to send.
        dst_stage: Destination pipeline stage index.
        tag: Message tag.

    Returns:
        A tensor with ``grad_fn`` (for backward recv) if the input had
        gradient; otherwise the original tensor (no grad_fn).
    """
    # If the tensor is not connected to any autograd graph, fall back to
    # a plain blocking send — no autograd Function overhead, no backward recv.
    if not tensor.requires_grad and tensor.grad_fn is None:
        pipeline_send(tensor, dst_stage, tag)
        return tensor

    out = PipelineSend.apply(tensor, dst_stage, tag)
    return out


def pipeline_async_send_with_grad(
    tensor: torch.Tensor, dst_stage: int, tag: int = 0,
) -> torch.Tensor:
    """
    Async send with autograd — always returns a tensor with ``_pp_work``.

    This is the **recommended** way to do an async send that participates
    in the autograd graph.  It wraps :class:`PipelineAsyncSend` and
    guarantees that the returned tensor has a valid ``_pp_work`` handle,
    even when ``autograd.Function.apply()`` strips custom attributes.

    When the input tensor has no gradient (``requires_grad=False`` and no
    ``grad_fn``), the send is performed without autograd (plain ``isend``)
    so that no backward anchor is needed.  The returned tensor will also
    have ``requires_grad=False``, signalling callers that this anchor
    should not be used for backward.

    Args:
        tensor: Tensor to send.
        dst_stage: Destination pipeline stage index.
        tag: Message tag.

    Returns:
        The (possibly wrapped) tensor with ``_pp_work`` attribute set.
    """
    # If the tensor is not connected to any autograd graph, fall back to
    # a plain async send — no autograd Function overhead, no backward recv.
    if not tensor.requires_grad and tensor.grad_fn is None:
        work = pipeline_isend(tensor, dst_stage, tag)
        # Return the original tensor with _pp_work attached, but no grad_fn.
        tensor._pp_work = work
        return tensor

    out = PipelineAsyncSend.apply(tensor, dst_stage, tag)
    if not hasattr(out, "_pp_work"):
        out._pp_work = PipelineAsyncSend._last_work
    return out


def pipeline_async_recv_with_grad(
    device: torch.device,
    src_stage: int,
    shape: tuple,
    dtype: torch.dtype,
    tag: int = 0,
) -> torch.Tensor:
    """
    Async recv with autograd — always returns a tensor with ``_pp_work``.

    This is the **recommended** way to do an async recv that participates
    in the autograd graph.  It wraps :class:`PipelineAsyncRecv` and
    guarantees that the returned tensor has a valid ``_pp_work`` handle,
    even when ``autograd.Function.apply()`` strips custom attributes.

    Args:
        device: Device for the receive buffer and placeholder.
        src_stage: Source pipeline stage index.
        shape: Shape of the tensor to receive.
        dtype: Data type of the tensor to receive.
        tag: Message tag.

    Returns:
        A new tensor with received data (async — call ``_pp_work.wait()``
        before reading) and ``_pp_work`` attribute set.
    """
    placeholder = torch.empty(0, device=device, requires_grad=True)
    out = PipelineAsyncRecv.apply(placeholder, src_stage, shape, dtype, device, tag)
    if not hasattr(out, "_pp_work"):
        out._pp_work = PipelineAsyncRecv._last_work
    return out
