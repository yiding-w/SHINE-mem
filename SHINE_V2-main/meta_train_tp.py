#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meta training entry point — TP (Tensor Parallel) path.

This file replaced the original PP version. The PP scaffolding (mem_gather,
lora_scatter, pipeline_forward_train_multi_mb / pipeline_backward_multi_mb,
the three-phase forward, reverse-order backward) is gone. Under TP every
collective lives inside the per-linear forward / backward; the outer
training loop is plain forward → loss → backward → step.

Layout: TP=N intra-node × DP=(world_size / N).

  * The hypernetwork (m2p_transformer + 2D pos embeddings) is **replicated**
    on every TP rank. After backward, the per-rank gradients are partial
    (each rank only routed loss through its slice of the loradict) — we
    sum across the TP group, then average across DP.
  * The LLM is TP-sharded on its full_attention layers (Colwise on q/k/v
    /gate/up, Rowwise on o/down) and replicated on its linear_attention
    layers (GatedDeltaNet doesn't decompose cleanly along the feature dim).

The training loop is intentionally simple: this is the place to add
features later (eval, checkpoint, detail logging, etc.).

Usage::

    torchrun --nproc_per_node=<n> --nnodes=<num_nodes> --node_rank=<rank> \
        --master_addr=<master_ip> --master_port=<port> \
        meta_train.py parallel.tensor_parallel_size=4
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
import time
import warnings
from contextlib import nullcontext
from typing import Optional
import json

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from hypernetwork.tp_model_hypernetwork import TPModelHypernetwork
from utils.mydata import (
    PipelineDataLoader,
    resolve_pad_token_id,
    create_dataset_from_config,
)
from utils.myparallel import (
    init_distributed,
    cleanup_distributed,
    setup_tensor_parallel,
    get_tp_config,
    get_rank,
    get_world_size,
    is_main_process,
    is_main_process_per_node,
    barrier,
)
from utils.mygpu import all_gpu_stats, _report_peak_memory

from utils.mysaveload import (
    get_checkpoint_dir,
    get_step_checkpoint_dir,
    list_checkpoints,
    get_latest_checkpoint,
    resolve_forever_save_steps,
    build_checkpoint_run_name,
    get_pretrain_final_checkpoint,
    get_pretrain_annealing_final_checkpoint,
)
from utils.mylog import format_duration, setup_debug_loggers, flush_debug_loggers
from utils.myprofiler import TrainingProfiler
from utils.mytraining_debug import (
    DebugSchedule,
    NanInfTracker,
    compute_grad_norms,
    compute_post_clip_grad_norm,
    compute_param_norms,
    compute_generated_lora_norms,
    compute_loss_spike_metrics,
    check_dp_param_consistency,
    check_tp_param_consistency,
    check_sp_param_consistency,
    log_training_detail,
)
from utils.myloradict import collect_loradict_tensors

# wandb is optional — disabled in CI / no-creds boxes via WANDB_MODE
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # noqa: N816

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradient sync helpers (TP-aware)
# ---------------------------------------------------------------------------

def _tp_sum_grads(module: nn.Module, tp_group, tp_world: int) -> None:
    """All-reduce-SUM gradients of every trainable param across the TP group.

    The hypernetwork is replicated on every TP rank but each rank only
    routed loss through its slice of the loradict, so per-rank grads are
    partial; summing reconstructs the full gradient that a non-sharded
    reference run would produce (validated by tests/test_tp_vs_ref_grad.py).

    Launches each per-param collective as async then waits at the end so
    NCCL can stream them through one CUDA stream instead of paying full
    latency per call.
    """
    if tp_group is None or tp_world <= 1:
        return
    handles = []
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            handles.append(dist.all_reduce(
                p.grad, op=dist.ReduceOp.SUM, group=tp_group, async_op=True,
            ))
    for h in handles:
        h.wait()


def _dp_avg_grads(module: nn.Module, dp_group) -> None:
    """All-reduce-AVG gradients across DP replicas (async-launched, same
    rationale as _tp_sum_grads)."""
    if dp_group is None:
        return
    if dist.get_world_size(dp_group) <= 1:
        return
    handles = []
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            handles.append(dist.all_reduce(
                p.grad, op=dist.ReduceOp.AVG, group=dp_group, async_op=True,
            ))
    for h in handles:
        h.wait()

def _tp_sum_grads_tensors(metalora_dict, tp_group, tp_world: int) -> None:
    """All-reduce-SUM gradients of metalora tensors across the TP group."""
    if tp_group is None or tp_world <= 1:
        return
    from utils.myloradict import collect_loradict_tensors
    handles = []
    for t in collect_loradict_tensors(metalora_dict):
        if t.requires_grad and t.grad is not None:
            handles.append(dist.all_reduce(
                t.grad, op=dist.ReduceOp.SUM, group=tp_group, async_op=True,
            ))
    for h in handles:
        h.wait()


def _dp_avg_grads_tensors(metalora_dict, dp_group) -> None:
    """All-reduce-AVG gradients of metalora tensors across DP replicas."""
    if dp_group is None:
        return
    if dist.get_world_size(dp_group) <= 1:
        return
    from utils.myloradict import collect_loradict_tensors
    handles = []
    for t in collect_loradict_tensors(metalora_dict):
        if t.requires_grad and t.grad is not None:
            handles.append(dist.all_reduce(
                t.grad, op=dist.ReduceOp.AVG, group=dp_group, async_op=True,
            ))
    for h in handles:
        h.wait()


def _broadcast_trainable_from_dp_rank0(module: nn.Module, tp_cfg) -> None:
    """Make every DP replica start with bit-identical trainable params.

    The source is the rank whose dp_rank=0 within this TP slot. Other DP
    replicas overwrite their trainable params with the source values.
    """
    dp_group = tp_cfg.get("dp_process_group")
    if dp_group is None or dist.get_world_size(dp_group) <= 1:
        return
    # In our topology rank-r-in-DP-group-s lives at global rank
    # ``s + r * tp_world``: the DP group's rank-0 is the one with
    # data_parallel_rank=0 for this TP slot.
    src_global = dist.get_global_rank(dp_group, 0)
    for p in module.parameters():
        if p.requires_grad:
            dist.broadcast(p.data, src=src_global, group=dp_group)


def _broadcast_metalora_from_dp_rank0(metalora_dict, tp_cfg) -> None:
    """Make every DP replica start with bit-identical metalora tensors.

    Same logic as _broadcast_trainable_from_dp_rank0 but operates on the
    metalora dict structure (not an nn.Module).
    """
    dp_group = tp_cfg.get("dp_process_group")
    if dp_group is None or dist.get_world_size(dp_group) <= 1:
        return
    from utils.myloradict import collect_loradict_tensors
    src_global = dist.get_global_rank(dp_group, 0)
    for t in collect_loradict_tensors(metalora_dict):
        if t.requires_grad:
            dist.broadcast(t.data, src=src_global, group=dp_group)


def _broadcast_trainable_from_tp_rank0(module: nn.Module, tp_cfg) -> None:
    """Make every TP rank start with bit-identical trainable params.

    The hypernetwork is replicated across TP ranks, but since there is no
    explicit seed synchronisation before construction, each rank may have
    different random initial weights.  This broadcast from tp_rank=0
    ensures all TP ranks hold identical parameters.
    """
    tp_group = tp_cfg.get("tp_process_group")
    if tp_group is None or dist.get_world_size(tp_group) <= 1:
        return
    src_global = dist.get_global_rank(tp_group, 0)
    for p in module.parameters():
        if p.requires_grad:
            dist.broadcast(p.data, src=src_global, group=tp_group)


def _broadcast_metalora_from_tp_rank0(metalora_dict, tp_cfg) -> None:
    """Make every TP rank start with bit-identical metalora tensors.

    Same as _broadcast_trainable_from_tp_rank0 but operates on the
    metalora dict structure (not an nn.Module).
    """
    tp_group = tp_cfg.get("tp_process_group")
    if tp_group is None or dist.get_world_size(tp_group) <= 1:
        return
    from utils.myloradict import collect_loradict_tensors
    src_global = dist.get_global_rank(tp_group, 0)
    for t in collect_loradict_tensors(metalora_dict):
        if t.requires_grad:
            dist.broadcast(t.data, src=src_global, group=tp_group)


def _broadcast_trainable_from_sp_rank0(module: nn.Module, tp_cfg) -> None:
    """Make every SP rank start with bit-identical trainable params.

    When SP > 1, different SP ranks within the same TP×SP group may have
    different random initial weights (no seed sync).  This broadcast from
    sp_rank=0 ensures all SP ranks hold identical parameters.
    """
    sp_group = tp_cfg.get("sp_process_group")
    if sp_group is None or dist.get_world_size(sp_group) <= 1:
        return
    src_global = dist.get_global_rank(sp_group, 0)
    for p in module.parameters():
        if p.requires_grad:
            dist.broadcast(p.data, src=src_global, group=sp_group)


def _broadcast_metalora_from_sp_rank0(metalora_dict, tp_cfg) -> None:
    """Make every SP rank start with bit-identical metalora tensors.

    Same as _broadcast_trainable_from_sp_rank0 but operates on the
    metalora dict structure (not an nn.Module).
    """
    sp_group = tp_cfg.get("sp_process_group")
    if sp_group is None or dist.get_world_size(sp_group) <= 1:
        return
    from utils.myloradict import collect_loradict_tensors
    src_global = dist.get_global_rank(sp_group, 0)
    for t in collect_loradict_tensors(metalora_dict):
        if t.requires_grad:
            dist.broadcast(t.data, src=src_global, group=sp_group)


def _broadcast_mem_tokens(model: 'TPModelHypernetwork', tp_cfg) -> None:
    """Broadcast mem_tokens from TP rank 0, SP rank 0, and DP rank 0.

    mem_tokens lives on model.llm.model and is zero-initialised, so all
    ranks are naturally identical at construction time.  However, after
    resume from checkpoint the loaded values may differ across ranks if
    the checkpoint was saved by a single rank.  This function ensures
    bit-identical mem_tokens across TP, SP, and DP groups.
    """
    mem = getattr(model.llm.model, "mem_tokens", None)
    if mem is None:
        return

    # TP-group broadcast
    tp_group = tp_cfg.get("tp_process_group")
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        src_global = dist.get_global_rank(tp_group, 0)
        dist.broadcast(mem.data, src=src_global, group=tp_group)

    # SP-group broadcast
    sp_group = tp_cfg.get("sp_process_group")
    if sp_group is not None and dist.get_world_size(sp_group) > 1:
        src_global = dist.get_global_rank(sp_group, 0)
        dist.broadcast(mem.data, src=src_global, group=sp_group)

    # DP-group broadcast
    dp_group = tp_cfg.get("dp_process_group")
    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        src_global = dist.get_global_rank(dp_group, 0)
        dist.broadcast(mem.data, src=src_global, group=dp_group)


# ---------------------------------------------------------------------------
# Optimizer + scheduler (TP-aware, minimal)
# ---------------------------------------------------------------------------

def _create_optimizer_scheduler(
    model: TPModelHypernetwork,
    num_training_steps: int,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    warmup_steps: int,
    min_learning_rate: float = 0.0,
    dp_group=None,
    use_zero1: bool = True,
):
    from torch.optim import AdamW
    from utils.myloradict import collect_loradict_tensors

    decay = []
    no_decay = []
    for name, p in model.hypernetwork.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "layernorm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    # Collect trainable metalora tensors (same as PP's create_optimizer_and_scheduler)
    metalora_tensors = []
    if hasattr(model, 'metalora') and model.metalora is not None:
        all_metalora = collect_loradict_tensors(model.metalora)
        for t in all_metalora:
            if t.requires_grad:
                metalora_tensors.append(t)

    # Collect trainable w_transform parameters
    w_transform_params = []
    for wt_name in ['w_transform_context', 'w_transform_conversation']:
        wt_module = getattr(model, wt_name, None)
        if wt_module is not None:
            for p in wt_module.parameters():
                if p.requires_grad:
                    w_transform_params.append(p)

    n_trainable = (
        sum(p.numel() for p in decay)
        + sum(p.numel() for p in no_decay)
        + sum(t.numel() for t in metalora_tensors)
        + sum(p.numel() for p in w_transform_params)
    )

    if use_zero1 and dp_group is not None and dist.get_world_size(dp_group) > 1:
        # ZeRO-1: shard Adam state across DP replicas. With DP=2 this
        # halves per-rank Adam state on a 1.5 B-param hypernetwork (~12 GB
        # → ~6 GB), unlocking larger mb.
        from torch.distributed.optim import ZeroRedundancyOptimizer
        # ZeRO requires a single param list (all of the same dtype) but
        # supports per-param-group config via a custom defaults dict
        # only awkwardly. Easiest: build one ZeRO optimizer per group.
        all_params = decay + no_decay + metalora_tensors + w_transform_params
        optimizer = ZeroRedundancyOptimizer(
            all_params,
            optimizer_class=AdamW,
            process_group=dp_group,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            fused=True,
        )
        if is_main_process_per_node():
            logger.info(
                f"[Optimizer] ZeRO-1 over node-local {dist.get_world_size(dp_group)} ranks: "
                f"{len(decay)} decay + {len(no_decay)} no_decay + "
                f"{len(metalora_tensors)} metalora + "
                f"{len(w_transform_params)} w_transform = "
                f"{n_trainable:,} elements"
            )
    else:
        grouped = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": metalora_tensors, "weight_decay": 0.0},  # metalora: no weight decay
            {"params": w_transform_params, "weight_decay": 0.0},  # w_transform: no weight decay
        ]
        if is_main_process_per_node():
            logger.info(
                f"[Optimizer] trainable params: "
                f"{len(decay)} decay + {len(no_decay)} no_decay + "
                f"{len(metalora_tensors)} metalora + "
                f"{len(w_transform_params)} w_transform = "
                f"{n_trainable:,} elements"
            )
        optimizer = AdamW(grouped, lr=learning_rate, betas=(beta1, beta2), eps=eps, fused=True)

    # Use the same LR schedule as PP: linear warmup + linear decay to min_lr
    min_lr_ratio = min_learning_rate / learning_rate if learning_rate > 0 else 0.0
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    scheduler = LambdaLR(optimizer, lr_lambda)

    if is_main_process_per_node():
        logger.info(
            f"[Scheduler] Linear warmup ({warmup_steps} steps) + "
            f"linear decay to min_lr={min_learning_rate} (total {num_training_steps} steps)"
        )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Checkpoint save / load helpers (PP-compatible format)
# ---------------------------------------------------------------------------

def _tp_save_checkpoint(
    model: TPModelHypernetwork,
    optimizer,
    lr_scheduler,
    global_step: int,
    epoch: int,
    micro_step: int,
    run_name: str,
    forever_save_steps,
    save_total_limit: int,
    running_loss: float = 0.0,
    ema_time_per_step: float = 0.0,
    wandb_run_id: Optional[str] = None,
    t_start: float = 0.0,
    max_steps: int = 0,
    train_loader=None,
    **kwargs,
) -> float:
    """Save a checkpoint in PP-compatible format.

    Only the main process (global rank 0) writes files. The checkpoint
    layout matches the PP path exactly::

        step_dir/model/model_stage0.safetensors
        step_dir/training_state/optimizer_stage0.pt
        step_dir/training_state/scheduler.pt
        step_dir/training_state/metadata.pt

    This ensures that:
      - TP checkpoints can be loaded by PP (via PP's load_model which
        reads model_stage*.safetensors and matches by key name).
      - PP checkpoints can be loaded by TP (via TP's load_model which
        reads model_stage*.safetensors and loads only hypernet.* keys).

    Returns wall-clock seconds spent saving (0.0 on non-main processes).
    """
    if not is_main_process():
        return 0.0

    import shutil
    save_t0 = time.time()

    step_dir = get_step_checkpoint_dir(run_name, global_step)
    model_dir = os.path.join(step_dir, "model")
    training_state_dir = os.path.join(step_dir, "training_state")

    # Save model (PP-compatible safetensors format)
    model.save_model(model_dir)

    # Save training state
    os.makedirs(training_state_dir, exist_ok=True)

    # Save scheduler state
    torch.save(lr_scheduler.state_dict(),
               os.path.join(training_state_dir, "scheduler.pt"))

    # Save training metadata (same fields as PP for cross-compatibility)
    elapsed_time = time.time() - t_start if t_start > 0 else 0.0
    metadata = {
        "global_step": global_step,
        "epoch": epoch,
        "micro_step": micro_step,
        "running_loss": running_loss,
        "ema_time_per_step": ema_time_per_step,
        "total_context_tokens": kwargs.get("total_context_tokens", 0),
        "total_conv_total_tokens": kwargs.get("total_conv_total_tokens", 0),
        "total_conv_valid_tokens": kwargs.get("total_conv_valid_tokens", 0),
        "wandb_run_id": wandb_run_id,
        "elapsed_time": elapsed_time,
        "parallel_mode": "tp",
        "config_selections": kwargs.get("config_selections", None),
        "launch_cmd": kwargs.get("launch_cmd", ""),
        "prev_repo": kwargs.get("prev_repo", None),
        # Parallel topology for resume validation
        "parallel_topology": {
            "tp_size": kwargs.get("tp_size", None),
            "sp_size": kwargs.get("sp_size", None),
            "gpus_per_node": kwargs.get("gpus_per_node", None),
            "num_nodes": kwargs.get("num_nodes", None),
        },
    }
    if train_loader is not None and hasattr(train_loader, "state_dict"):
        metadata["dataloader_state"] = train_loader.state_dict()
    torch.save(metadata, os.path.join(training_state_dir, "metadata.pt"))

    save_duration = time.time() - save_t0

    # Rotate checkpoints (delete oldest non-forever)
    if global_step not in forever_save_steps:
        _tp_rotate_checkpoints(run_name, forever_save_steps, save_total_limit)

    return save_duration


def _tp_rotate_checkpoints(run_name: str, forever_save_steps, save_total_limit: int):
    """Delete oldest non-forever checkpoints if we exceed save_total_limit."""
    import shutil
    all_steps = list_checkpoints(run_name)
    non_forever_steps = [s for s in all_steps if s not in forever_save_steps]
    while len(non_forever_steps) > save_total_limit:
        oldest_step = non_forever_steps.pop(0)
        oldest_dir = get_step_checkpoint_dir(run_name, oldest_step)
        try:
            shutil.rmtree(oldest_dir)
            logger.info(f"  [Checkpoint] Deleted old checkpoint: step_{oldest_step}")
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"  [Checkpoint] Failed to delete step_{oldest_step}: {e}")


def _tp_save_optimizer_state_sharded(
    model: TPModelHypernetwork, optimizer, training_state_dir: str, local_rank: int
):
    """Save per-rank optimizer shard (ZeRO-1 sharded save — no consolidation needed).

    Each rank saves its own local optimizer state shard to a file keyed by
    local_rank. On resume, each rank loads its own shard. This avoids the
    expensive consolidate_state_dict() all-gather.

    Only ranks on node 0 need to save (all nodes have identical shards due to
    replicated parameters + identical node-local ZeRO-1 partitioning).
    """
    from safetensors.torch import save_file
    from torch.distributed.optim import ZeroRedundancyOptimizer

    # For ZeRO optimizer, get the local optimizer's state directly
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        # Access the underlying local optimizer state (only this rank's shard)
        local_opt = optimizer.optim
        opt_sd = local_opt.state_dict()
    else:
        opt_sd = optimizer.state_dict()

    tensors_dict = {}
    meta_dict = {}

    for str_idx, state in opt_sd["state"].items():
        idx = int(str_idx) if isinstance(str_idx, str) else str_idx
        param_meta = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                tensors_dict[f"param{idx}__{k}"] = v.cpu()
            else:
                param_meta[k] = v
        if param_meta:
            meta_dict[str(idx)] = param_meta

    # Save param_groups metadata (lr, betas, etc.)
    groups_meta = []
    for g in opt_sd["param_groups"]:
        meta = {k: v for k, v in g.items() if k != "params"}
        # Also save param indices for reconstruction
        meta["_param_indices"] = g["params"]
        groups_meta.append(meta)

    # Write per-rank files
    if tensors_dict:
        save_file(tensors_dict, os.path.join(
            training_state_dir, f"optimizer_tensors_rank{local_rank}.safetensors"))

    payload = {
        "non_tensor_state": meta_dict,
        "param_groups_meta": groups_meta,
    }
    torch.save(payload, os.path.join(
        training_state_dir, f"optimizer_meta_rank{local_rank}.pt"))


def _tp_load_checkpoint(
    model: TPModelHypernetwork,
    optimizer,
    lr_scheduler,
    checkpoint_dir: str,
    my_device: torch.device,
    local_rank: int = 0,
    tp_cfg: dict = None,
) -> dict:
    """Load a full checkpoint (model + optimizer + scheduler + metadata).

    Uses sharded optimizer loading — each rank loads its own shard by local_rank.
    Validates that parallel topology matches the checkpoint.

    Returns training metadata dict.
    """
    model_dir = os.path.join(checkpoint_dir, "model")
    training_state_dir = os.path.join(checkpoint_dir, "training_state")

    # Load and validate metadata first
    metadata_path = os.path.join(training_state_dir, "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location="cpu")
    else:
        metadata = {}

    # --- Validate parallel topology ---
    saved_topo = metadata.get("parallel_topology")
    if saved_topo is not None and tp_cfg is not None:
        current_topo = {
            "tp_size": tp_cfg.get("tensor_parallel_size"),
            "sp_size": tp_cfg.get("sequence_parallel_size"),
            "gpus_per_node": tp_cfg.get("total_gpus"),
        }
        mismatches = []
        for key in ["tp_size", "sp_size", "gpus_per_node"]:
            saved_val = saved_topo.get(key)
            current_val = current_topo.get(key)
            if saved_val is not None and current_val is not None and saved_val != current_val:
                mismatches.append(f"  {key}: checkpoint={saved_val}, current={current_val}")
        if mismatches:
            raise RuntimeError(
                f"[_tp_load_checkpoint] RESUME FAILED — parallel topology mismatch!\n"
                f"The checkpoint was saved with a different parallel configuration.\n"
                + "\n".join(mismatches) + "\n"
                f"Sharded optimizer state requires identical topology to resume.\n"
                f"Checkpoint: {checkpoint_dir}"
            )

    # Load model parameters (PP-compatible)
    model.load_model(model_dir)

    # Load optimizer state (sharded per-rank)
    _tp_load_optimizer_state_sharded(optimizer, training_state_dir, local_rank, my_device)

    # Load scheduler state
    sched_path = os.path.join(training_state_dir, "scheduler.pt")
    if os.path.exists(sched_path):
        sched_state = torch.load(sched_path, map_location="cpu")
        lr_scheduler.load_state_dict(sched_state)
    else:
        logger.warning("No scheduler checkpoint found at %s", sched_path)

    return metadata


def _tp_load_model_only(
    model: TPModelHypernetwork,
    checkpoint_dir: str,
) -> dict:
    """Load only model weights from a checkpoint (no optimizer/scheduler).

    Used for loading a pretrain final checkpoint as starting point for
    annealing/SFT, or for cross-loading between TP and PP.

    Returns training metadata dict (if available).
    """
    model_dir = os.path.join(checkpoint_dir, "model")
    model.load_model(model_dir)

    metadata_path = os.path.join(checkpoint_dir, "training_state", "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location="cpu")
    else:
        metadata = {}
    return metadata


def _tp_load_optimizer_state_sharded(
    optimizer, training_state_dir: str, local_rank: int, my_device: torch.device
):
    """Load per-rank optimizer shard (ZeRO-1 sharded load).

    Each rank loads its own shard file keyed by local_rank. This is the
    counterpart to _tp_save_optimizer_state_sharded().
    """
    from safetensors import safe_open
    from torch.distributed.optim import ZeroRedundancyOptimizer

    st_file = os.path.join(training_state_dir, f"optimizer_tensors_rank{local_rank}.safetensors")
    meta_file = os.path.join(training_state_dir, f"optimizer_meta_rank{local_rank}.pt")

    if not os.path.exists(st_file):
        raise FileNotFoundError(
            f"[_tp_load_optimizer_state_sharded] STRICT LOAD FAILED — "
            f"No optimizer shard file found for rank {local_rank}: {st_file}\n"
            f"Cannot resume training without optimizer state."
        )

    # Load tensors
    device_str = str(my_device)
    state_dict_state = {}
    with safe_open(st_file, framework="pt", device=device_str) as sf:
        for flat_key in sf.keys():
            sep_idx = flat_key.rfind("__")
            if sep_idx == -1:
                continue
            param_key = flat_key[:sep_idx]  # "param<idx>"
            state_name = flat_key[sep_idx + 2:]
            idx = int(param_key.replace("param", ""))
            if idx not in state_dict_state:
                state_dict_state[idx] = {}
            state_dict_state[idx][state_name] = sf.get_tensor(flat_key)

    # Load non-tensor metadata
    if os.path.exists(meta_file):
        payload = torch.load(meta_file, map_location="cpu")
        non_tensor_state = payload.get("non_tensor_state", {})
        groups_meta = payload.get("param_groups_meta", [])
    else:
        non_tensor_state = {}
        groups_meta = []

    # Merge non-tensor state (e.g. step count as int)
    for str_idx, meta in non_tensor_state.items():
        idx = int(str_idx)
        if idx not in state_dict_state:
            state_dict_state[idx] = {}
        state_dict_state[idx].update(meta)

    # Reconstruct param_groups with saved hyperparameters
    param_groups = []
    for gm in groups_meta:
        group = {k: v for k, v in gm.items() if k != "_param_indices"}
        group["params"] = gm.get("_param_indices", [])
        param_groups.append(group)

    load_sd = {
        "state": state_dict_state,
        "param_groups": param_groups,
    }

    # Load into the optimizer (for ZeRO, load into the local optimizer)
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer.optim.load_state_dict(load_sd)
    else:
        optimizer.load_state_dict(load_sd)

    # Fix device mismatch for fused Adam: after load_state_dict, 'step' tensors
    # may remain on CPU. Fused Adam requires all state tensors on CUDA.
    target_device = my_device
    opt_state = optimizer.state
    if hasattr(optimizer, 'optim') and hasattr(optimizer.optim, 'state'):
        opt_state = optimizer.optim.state
    for param_state in opt_state.values():
        if isinstance(param_state, dict) and "step" in param_state:
            step_val = param_state["step"]
            if isinstance(step_val, torch.Tensor) and step_val.device != target_device:
                param_state["step"] = step_val.to(target_device)


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def _train_step(
    model: TPModelHypernetwork,
    batch: dict,
    optimizer,
    lr_scheduler,
    tp_cfg,
    grad_accum_steps: int,
    gradient_clipping: float,
    micro_step: int,
    accum_loss: float,
    sdpa_ctx_factory,
    nan_inf_tracker: Optional[NanInfTracker] = None,
    monitor_nan_inf: bool = False,
    grad_norm_sched: Optional[DebugSchedule] = None,
    global_step: int = 0,
    distill_batch: Optional[dict] = None,
    distill_loss_fn=None,
    return_per_token_loss: bool = False,
    return_acc: bool = False,
) -> tuple:
    """Run one micro-batch forward+backward; step optimizer when
    ``(micro_step + 1) % grad_accum_steps == 0``.

    Returns (accum_loss, did_step, skipped, grad_norm_metrics, per_token_loss,
    distill_loss_item).  per_token_loss is a (B, S-1) float32 CPU tensor or
    None; distill_loss_item is a float or None.
    """
    context_ids = batch["context_ids"]
    conv_ids = batch["conversation_ids"]
    labels = batch["labels"]
    ctx_lengths = batch["context_lengths"]
    _per_token_loss = None  # populated when return_per_token_loss=True
    _distill_loss_val = None  # populated when distillation is active
    _regu_sq_norm = 0.0  # populated from model.forward() return

    with sdpa_ctx_factory():
        # Prepare distillation args (None if no distillation — zero extra cost)
        _distill_conv_ids = None
        _distill_labels = None
        if distill_loss_fn is not None and distill_batch is not None:
            _distill_conv_ids = distill_batch["conversation_ids"]
            _distill_labels = distill_batch["labels"]

        _has_distill = _distill_conv_ids is not None

        if return_per_token_loss and _has_distill:
            # Returns ((total_loss, per_token_loss, distill_loss_detached), regu_sq_norm, regu_loss)
            _result, _regu_sq_norm, _regu_loss = model(
                context_ids=context_ids,
                context_lengths=ctx_lengths,
                conversation_ids=conv_ids,
                labels=labels,
                return_per_token_loss=True,
                distill_loss_fn=distill_loss_fn,
                distill_conversation_ids=_distill_conv_ids,
                distill_labels=_distill_labels,
                grad_accum_steps=grad_accum_steps,
                return_acc=return_acc,
            )
            loss, _per_token_loss, _distill_loss_val = _result
        elif return_per_token_loss and not _has_distill:
            # Returns ((ce_loss, per_token_loss), regu_sq_norm, regu_loss)
            _result, _regu_sq_norm, _regu_loss = model(
                context_ids=context_ids,
                context_lengths=ctx_lengths,
                conversation_ids=conv_ids,
                labels=labels,
                return_per_token_loss=True,
                grad_accum_steps=grad_accum_steps,
                return_acc=return_acc,
            )
            loss, _per_token_loss = _result
        elif not return_per_token_loss and _has_distill:
            # Returns ((total_loss, distill_loss_detached), regu_sq_norm, regu_loss)
            _result, _regu_sq_norm, _regu_loss = model(
                context_ids=context_ids,
                context_lengths=ctx_lengths,
                conversation_ids=conv_ids,
                labels=labels,
                distill_loss_fn=distill_loss_fn,
                distill_conversation_ids=_distill_conv_ids,
                distill_labels=_distill_labels,
                grad_accum_steps=grad_accum_steps,
                return_acc=return_acc,
            )
            loss, _distill_loss_val = _result
        else:
            # Returns (scalar ce_loss, regu_sq_norm, regu_loss)
            _result, _regu_sq_norm, _regu_loss = model(
                context_ids=context_ids,
                context_lengths=ctx_lengths,
                conversation_ids=conv_ids,
                labels=labels,
                grad_accum_steps=grad_accum_steps,
                return_acc=return_acc,
            )
            loss = _result

        # For backward: add regu_loss to loss (so regu gradients flow).
        # But loss_val for logging uses the original loss (CE + distill only).
        backward_loss = loss
        if _regu_loss is not None:
            backward_loss = loss + _regu_loss
        scaled = backward_loss / grad_accum_steps
        torch.cuda.nvtx.range_push("Backward")
        scaled.backward()
        torch.cuda.nvtx.range_pop()  # Backward

    loss_val = loss.detach().float().item()
    accum_loss += loss_val

    # NaN/Inf detection (check local loss)
    _local_nan_inf = False
    if monitor_nan_inf and nan_inf_tracker is not None:
        _local_nan_inf = nan_inf_tracker.check_and_record(loss_val, global_step + 1)

    do_step = ((micro_step + 1) % grad_accum_steps == 0)
    _skipped = False
    _grad_norm_metrics = {}

    if do_step:
        # TP-sum on hypernetwork grads (each rank had partial grad),
        # then SP-sum (each SP rank saw partial sequence),
        # then DP-avg across replicas.
        torch.cuda.nvtx.range_push("GradSync")
        _tp_sum_grads(model.hypernetwork, tp_cfg.get("tp_process_group"), tp_cfg["tensor_parallel_size"])
        # SP-sum: each SP rank only saw a chunk of the sequence, so grads are partial
        _sp_group = tp_cfg.get("sp_process_group")
        _sp_world = tp_cfg.get("sequence_parallel_size", 1)
        if _sp_group is not None and _sp_world > 1:
            _tp_sum_grads(model.hypernetwork, _sp_group, _sp_world)
        _dp_avg_grads(model.hypernetwork, tp_cfg.get("dp_process_group"))

        # Metalora gradient sync: TP-sum + SP-sum + DP-avg
        for layer_idx, layer_lora in model.metalora.items():
            _tp_sum_grads_tensors(layer_lora, tp_cfg.get("tp_process_group"), tp_cfg["tensor_parallel_size"])
            if _sp_group is not None and _sp_world > 1:
                _tp_sum_grads_tensors(layer_lora, _sp_group, _sp_world)
            _dp_avg_grads_tensors(layer_lora, tp_cfg.get("dp_process_group"))

        # W-Transform gradient sync: TP-sum + SP-sum + DP-avg
        # (L and R have partial grads on each TP rank due to sharded W)
        for wt_name in ('w_transform_context', 'w_transform_conversation'):
            wt_module = getattr(model, wt_name, None)
            if wt_module is not None:
                # Access underlying module if torch.compile wrapped it
                _wt = wt_module._orig_mod if hasattr(wt_module, '_orig_mod') else wt_module
                _tp_sum_grads(_wt, tp_cfg.get("tp_process_group"), tp_cfg["tensor_parallel_size"])
                if _sp_group is not None and _sp_world > 1:
                    _tp_sum_grads(_wt, _sp_group, _sp_world)
                _dp_avg_grads(_wt, tp_cfg.get("dp_process_group"))
        torch.cuda.nvtx.range_pop()  # GradSync

        # NaN/Inf: broadcast skip decision across SP + DP replicas
        # All ranks that share the same parameters must agree on skip/no-skip,
        # otherwise gradients diverge and parameters become inconsistent.
        _skip_step = False
        if monitor_nan_inf:
            my_device = tp_cfg["device"]
            _skip_tensor = torch.tensor([1.0 if _local_nan_inf else 0.0],
                                         dtype=torch.float32, device=my_device)
            # SP group: different SP ranks see different sequence chunks and may
            # independently detect NaN; sync so all SP ranks make same decision.
            _sp_skip_group = tp_cfg.get("sp_process_group")
            if _sp_skip_group is not None and tp_cfg.get("sequence_parallel_size", 1) > 1:
                dist.all_reduce(_skip_tensor, op=dist.ReduceOp.MAX, group=_sp_skip_group)
            dp_group = tp_cfg.get("dp_process_group")
            if dp_group is not None:
                dist.all_reduce(_skip_tensor, op=dist.ReduceOp.MAX, group=dp_group)
            _skip_step = _skip_tensor.item() > 0.5

        if _skip_step:
            optimizer.zero_grad(set_to_none=True)
            _skipped = True
        else:
            # Grad norm monitoring (pre-clip)
            _upcoming_step = global_step + 1
            if grad_norm_sched is not None and grad_norm_sched.should_run(_upcoming_step):
                _grad_norm_metrics = compute_grad_norms(model, tp_cfg["device"])

            if gradient_clipping > 0:
                # Collect both hypernetwork params and metalora tensors for clipping
                from utils.myloradict import collect_loradict_tensors
                all_graded = [p for p in model.hypernetwork.parameters() if p.requires_grad and p.grad is not None]
                metalora_graded = [t for t in collect_loradict_tensors(model.metalora) if t.requires_grad and t.grad is not None]
                all_graded.extend(metalora_graded)
                # Include w_transform parameters in gradient clipping
                for wt_name in ('w_transform_context', 'w_transform_conversation'):
                    wt_module = getattr(model, wt_name, None)
                    if wt_module is not None:
                        _wt = wt_module._orig_mod if hasattr(wt_module, '_orig_mod') else wt_module
                        wt_graded = [p for p in _wt.parameters() if p.requires_grad and p.grad is not None]
                        all_graded.extend(wt_graded)
                torch.nn.utils.clip_grad_norm_(all_graded, gradient_clipping)

            # Post-clip grad norm
            if _grad_norm_metrics:
                _post_clip_total = compute_post_clip_grad_norm(model, tp_cfg["device"])
                _grad_norm_metrics["grad_norm/post_clip_total"] = _post_clip_total
                # Compute pre-clip total from the metrics
                _pre_clip_total = _grad_norm_metrics.get("grad_norm/hyper_avg", 0.0)
                if gradient_clipping > 0:
                    _grad_norm_metrics["grad_norm/clip_ratio"] = (
                        _post_clip_total / gradient_clipping if gradient_clipping > 0 else 0.0
                    )

            torch.cuda.nvtx.range_push("OptimizerStep")
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.nvtx.range_pop()  # OptimizerStep

    return accum_loss, do_step, _skipped, _grad_norm_metrics, (_per_token_loss.cpu() if _per_token_loss is not None else None), (_distill_loss_val.item() if _distill_loss_val is not None else None), _regu_sq_norm


# ---------------------------------------------------------------------------
# Evaluation (TP-specific, much simpler than PP since no pipeline)
# ---------------------------------------------------------------------------

def _tp_run_evaluation(
    model: TPModelHypernetwork,
    val_loader,
    tp_cfg: dict,
    sdpa_ctx_factory,
    global_step: int,
    use_wandb: bool = False,
    t_start: float = 0.0,
    max_steps: int = 0,
    ema_time_per_step: float = 0.0,
    distill_loss_fn=None,
    baseline_mode: bool = False,
) -> tuple[float, float]:
    """Run evaluation on the validation set.

    Much simpler than PP's run_evaluation because TP has no pipeline stages.
    All TP ranks have the full model, so we just do forward-only passes.

    This is a collective operation: ALL DP ranks must call it simultaneously
    (for the DP loss aggregation).

    Args:
        baseline_mode: If True, run base LLM only (no hypernetwork, no LoRA,
            no detach_state). Used for computing a baseline loss/ppl to
            compare against the trained model.

    Returns (avg_loss, avg_ppl) on main process, (0.0, 0.0) on others.
    """
    if val_loader is None:
        return 0.0, 0.0

    _eval_start_time = time.time()
    model.eval()
    my_device = tp_cfg["device"]
    dp_group = tp_cfg.get("dp_process_group")

    # Use eval_context so evaluation has its own fresh zero-initialized wdict
    # (independent of training state, same as PP)
    # In baseline_mode, skip eval_context entirely (no detach_state involvement)
    if not baseline_mode:
        _eval_ctx = model.detach_state.eval_context() if model.detach_state is not None else nullcontext()
    else:
        _eval_ctx = nullcontext()
    _eval_ctx.__enter__()

    total_loss = 0.0
    total_distill_loss = 0.0
    num_batches = 0
    do_eval_distill = (distill_loss_fn is not None) and not baseline_mode
    _eval_prev_repo = None  # Track repo for repo-change reset during eval
    _eval_running_reset_ratio = 0.0
    _eval_running_mean_update_step = 0.0
    _eval_running_regu_sq_norm = 0.0
    _eval_repo_reset_count = 0

    val_loader.set_epoch(0)  # Fixed epoch for reproducibility
    val_iter = iter(val_loader)

    with torch.no_grad():
        while True:
            try:
                batch_meta = next(val_iter)
            except StopIteration:
                break

            mbs = batch_meta.get("micro_batches", [])
            for mb in mbs:
                mb_dev = {
                    "conversation_ids": mb["conversation_ids"].to(my_device, non_blocking=True),
                    "labels": mb["labels"].to(my_device, non_blocking=True),
                }
                if not baseline_mode:
                    mb_dev["context_ids"] = mb["context_ids"].to(my_device, non_blocking=True)
                    mb_dev["context_lengths"] = mb["context_lengths"].to(my_device, non_blocking=True)

                if baseline_mode:
                    # Base LLM forward: no loradict, no nograd, no detach_state
                    with sdpa_ctx_factory():
                        loss = model.compute_loss(
                            input_ids=mb_dev["conversation_ids"],
                            labels=mb_dev["labels"],
                            loradict=None,
                            nograd_loradict=None,
                            nograd_wdict=None,
                        )
                    total_loss += loss.detach().float().item()
                    num_batches += 1
                    continue

                # --- Normal evaluation path (with hypernetwork) ---
                # Repo-change reset during eval (same logic as training)
                _eval_extra_info = mb.get("extra_info", None)
                if _eval_extra_info is not None and model.detach_state is not None:
                    _eval_cur_repo = _eval_extra_info[0].get("repo") if isinstance(_eval_extra_info, list) and len(_eval_extra_info) > 0 else None
                    if _eval_cur_repo is not None and _eval_prev_repo is not None and _eval_cur_repo != _eval_prev_repo:
                        model.detach_state.reset()
                        model.detach_state.init_steps()
                        _eval_repo_reset_count += 1
                    if _eval_cur_repo is not None:
                        _eval_prev_repo = _eval_cur_repo

                # Prepare distillation data (if available)
                _distill_conv_ids = None
                _distill_labels = None
                if do_eval_distill and mb.get("distill") is not None:
                    _distill_conv_ids = mb["distill"]["conversation_ids"].to(my_device, non_blocking=True)
                    _distill_labels = mb["distill"]["labels"].to(my_device, non_blocking=True)

                with sdpa_ctx_factory():
                    if _distill_conv_ids is not None:
                        # Forward with distillation: returns ((total_loss, distill_loss_detached), regu_sq_norm, regu_loss)
                        _result, _eval_regu_sq, _ = model(
                            context_ids=mb_dev["context_ids"],
                            context_lengths=mb_dev["context_lengths"],
                            conversation_ids=mb_dev["conversation_ids"],
                            labels=mb_dev["labels"],
                            distill_loss_fn=distill_loss_fn,
                            distill_conversation_ids=_distill_conv_ids,
                            distill_labels=_distill_labels,
                            grad_accum_steps=1,
                        )
                        # _result is (total_loss, distill_loss_detached)
                        # total_loss = ce_loss + coeff * distill_loss (no regu_loss)
                        _total_loss_tensor, _distill_loss_tensor = _result
                        _coeff = distill_loss_fn.coefficient if hasattr(distill_loss_fn, 'coefficient') else 1.0
                        _distill_loss_item = _distill_loss_tensor.float().item()
                        _total_loss_item = _total_loss_tensor.detach().float().item()
                        # CE loss = total_loss - coeff * distill_loss
                        _ce_loss = _total_loss_item - _coeff * _distill_loss_item
                        total_loss += _ce_loss
                        total_distill_loss += _distill_loss_item
                    else:
                        # Forward without distillation: returns (ce_loss, regu_sq_norm, regu_loss)
                        loss, _eval_regu_sq, _ = model(
                            context_ids=mb_dev["context_ids"],
                            context_lengths=mb_dev["context_lengths"],
                            conversation_ids=mb_dev["conversation_ids"],
                            labels=mb_dev["labels"],
                            grad_accum_steps=1,
                        )
                        total_loss += loss.detach().float().item()

                # Write detach_state for eval accumulation (no backward needed in eval)
                model.post_backward_detach_state(grad_accum_steps=1)

                # Accumulate regu_sq_norm
                _eval_running_regu_sq_norm += _eval_regu_sq if _eval_regu_sq else 0.0

                # set_last_sq_norms → update_steps → get_reset_stats → maybe_reset_slice
                # (same order as training loop)
                if model.detach_state is not None and hasattr(model.detach_state, "_local_batch_size"):
                    _ds_batch_size = model.detach_state._local_batch_size
                    if _eval_regu_sq and _eval_regu_sq > 0:
                        # All-reduce across TP for full sq_norm before threshold check
                        _sq_t = torch.tensor([_eval_regu_sq], dtype=torch.float64, device=my_device)
                        tp_group = tp_cfg.get("tp_process_group")
                        if tp_group is not None and tp_cfg["tensor_parallel_size"] > 1:
                            dist.all_reduce(_sq_t, op=dist.ReduceOp.SUM, group=tp_group)
                        _eval_sq_norms_per_sample = [_sq_t.item()] * _ds_batch_size
                        model.detach_state.set_last_sq_norms(_eval_sq_norms_per_sample)
                    for _si in range(_ds_batch_size):
                        model.detach_state.update_steps(_si)
                    _eval_reset_ratio, _eval_mean_upd = model.detach_state.get_reset_stats()
                    _eval_running_reset_ratio += _eval_reset_ratio
                    _eval_running_mean_update_step += _eval_mean_upd
                    for _si in range(_ds_batch_size):
                        model.detach_state.maybe_reset_slice(_si)

                num_batches += 1

    # Exit eval_context (restores training wdict)
    _eval_ctx.__exit__(None, None, None)
    model.train()

    if num_batches == 0:
        if is_main_process_per_node():
            logger.warning(f"  [Eval Step {global_step}] No validation batches processed")
        return 0.0, 0.0

    avg_loss = total_loss / num_batches
    avg_distill_loss = total_distill_loss / num_batches

    # DP aggregation: average loss across all DP replicas
    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        loss_tensor = torch.tensor([avg_loss, avg_distill_loss], device=my_device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG, group=dp_group)
        avg_loss = loss_tensor[0].item()
        avg_distill_loss = loss_tensor[1].item()

    # DP aggregation: average detach_state metrics across all DP replicas
    if model.detach_state is not None and num_batches > 0:
        _local_eval_regu = _eval_running_regu_sq_norm / num_batches
        _local_eval_reset = _eval_running_reset_ratio / num_batches
        _local_eval_update = _eval_running_mean_update_step / num_batches
        _local_eval_repo_reset = _eval_repo_reset_count / num_batches
        if dp_group is not None and dist.get_world_size(dp_group) > 1:
            _ds_eval_tensor = torch.tensor(
                [_local_eval_regu, _local_eval_reset, _local_eval_update, _local_eval_repo_reset],
                device=my_device,
            )
            dist.all_reduce(_ds_eval_tensor, op=dist.ReduceOp.AVG, group=dp_group)
            _global_eval_regu = _ds_eval_tensor[0].item()
            _global_eval_reset = _ds_eval_tensor[1].item()
            _global_eval_update = _ds_eval_tensor[2].item()
            _global_eval_repo_reset = _ds_eval_tensor[3].item()
        else:
            _global_eval_regu = _local_eval_regu
            _global_eval_reset = _local_eval_reset
            _global_eval_update = _local_eval_update
            _global_eval_repo_reset = _local_eval_repo_reset
    else:
        _global_eval_regu = 0.0
        _global_eval_reset = 0.0
        _global_eval_update = 0.0
        _global_eval_repo_reset = 0.0

    avg_ppl = math.exp(min(avg_loss, 20.0))
    # Total loss = CE + coefficient * distill (coefficient applied in distill_loss_fn)
    _distill_coeff = distill_loss_fn.coefficient if distill_loss_fn is not None else 1.0
    total_loss_val = avg_loss + _distill_coeff * avg_distill_loss
    total_ppl = math.exp(min(total_loss_val, 20.0))

    if is_main_process_per_node():
        eval_elapsed = time.time() - t_start if t_start > 0 else 0.0
        eval_duration = time.time() - _eval_start_time
        eval_steps_remaining = max_steps - global_step
        eval_eta = ema_time_per_step * eval_steps_remaining if ema_time_per_step > 0 else 0.0
        _eval_distill_suffix = ""
        if avg_distill_loss > 0:
            _eval_distill_suffix = (
                f", val_distill_loss={avg_distill_loss:.4f}, "
                f"val_total_loss={total_loss_val:.4f}, "
                f"val_total_ppl={total_ppl:.2f}"
            )
        logger.info(
            f"  [Eval Step {global_step}] "
            f"val_loss={avg_loss:.4f}, val_ppl={avg_ppl:.2f}"
            f"{_eval_distill_suffix}, "
            f"val_batches={num_batches}, "
            f"eval_time={format_duration(eval_duration)}, "
            f"elapsed={format_duration(eval_elapsed)}, eta={format_duration(eval_eta)}"
        )
        # Log to dedicated evaluation.log (same as PP)
        _eval_logger = logging.getLogger("debug.evaluation")
        _eval_logger.info(
            f"[Eval Step {global_step}] "
            f"val_loss={avg_loss:.6f}, val_ppl={avg_ppl:.4f}, "
            f"val_distill_loss={avg_distill_loss:.6f}, "
            f"val_total_loss={total_loss_val:.6f}, val_total_ppl={total_ppl:.4f}, "
            f"val_batches_per_node={num_batches}, "
            f"eval_time={format_duration(eval_duration)}, "
            f"elapsed={format_duration(eval_elapsed)}, eta={format_duration(eval_eta)}"
        )
    if is_main_process() and use_wandb and wandb is not None:
        _eval_wandb_metrics = {
            "wall_time": time.time() - t_start,
            "eval/loss": avg_loss,
            "eval/ppl": avg_ppl,
            "eval/distill_loss": avg_distill_loss,
            "eval/total_loss": total_loss_val,
            "eval/total_ppl": total_ppl,
        }
        # DetachState metrics for eval
        if model.detach_state is not None and num_batches > 0:
            _eval_wandb_metrics["eval/regu_sq_norm"] = _global_eval_regu
            _eval_wandb_metrics["eval/reset_ratio"] = _global_eval_reset
            _eval_wandb_metrics["eval/mean_update_step"] = _global_eval_update
            _eval_wandb_metrics["eval/repo_reset_ratio"] = _global_eval_repo_reset
        wandb.log(_eval_wandb_metrics, step=global_step)

    return avg_loss, avg_ppl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def tp_main(cfg: DictConfig):
    """TP training entry point.

    Called from ``meta_train.main()`` when ``cfg.parallel.mode == 'tp'``.
    Self-contained: runs its own training loop (forward -> loss -> backward
    -> step), independent of the PP pipeline scheduler. The PP code path in
    meta_train.py is left bit-for-bit unchanged.
    """
    # Allow TF32 for fp32 matmuls (softmax denominator, RMSNorm fp32 path,
    # gradient accumulation when wrapped in fp32). A800 has TF32 tensor
    # cores; default 'highest' precision uses fp32-only multiply, ~2-3×
    # slower on these ops. bf16 forward/backward is unaffected — this
    # only governs the fp32 GEMM precision.
    torch.set_float32_matmul_precision("high")
    init_distributed()

    if is_main_process_per_node():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
            force=True,
        )
        logger.info("Starting TP meta training …")

    # ------------------------------------------------------------------
    # 1. Parallel setup — TP only on the new path
    # ------------------------------------------------------------------
    tensor_parallel_size = int(cfg.parallel.get("tensor_parallel_size", cfg.parallel.total_gpus))
    sequence_parallel_size = int(cfg.parallel.get("sequence_parallel_size", 1))
    if cfg.parallel.get("pipeline_parallel_size", 1) > 1 and is_main_process_per_node():
        logger.warning(
            f"[main] parallel.pipeline_parallel_size={cfg.parallel.pipeline_parallel_size} "
            f"is ignored on the TP path — using tensor_parallel_size={tensor_parallel_size}."
        )
    tp_cfg = setup_tensor_parallel(
        total_gpus=cfg.parallel.total_gpus,
        tensor_parallel_size=tensor_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
    )
    my_device = tp_cfg["device"]
    if is_main_process_per_node():
        logger.info(
            f"TP={tp_cfg['tensor_parallel_size']} SP={tp_cfg['sequence_parallel_size']} "
            f"DP={tp_cfg['data_parallel_size']} "
            f"tp_rank={tp_cfg['tp_rank']} sp_rank={tp_cfg.get('sp_rank', 0)} "
            f"dp_rank={tp_cfg['data_parallel_rank']} device={my_device}"
        )
    all_gpu_stats("After TP setup")

    # ------------------------------------------------------------------
    # 2. Build TP model + hypernetwork
    # ------------------------------------------------------------------
    # Inject detach_state config into model config (detach_state is a
    # top-level Hydra config group, but TPModelHypernetwork reads it from
    # model_cfg.detach_state).
    if cfg.get("detach_state"):
        from omegaconf import OmegaConf, open_dict
        with open_dict(cfg.model):
            cfg.model.detach_state = cfg.detach_state

    if is_main_process_per_node():
        logger.info("Building TPModelHypernetwork …")
    model = TPModelHypernetwork(
        model_cfg=cfg.model,
        m2p_transformer_cfg=cfg.m2p_transformer,
        tp_rank=tp_cfg["tp_rank"],
        tp_world=tp_cfg["tensor_parallel_size"],
        tp_process_group=tp_cfg.get("tp_process_group"),
        sp_group=tp_cfg.get("sp_process_group"),
        sp_world=tp_cfg.get("sequence_parallel_size", 1),
        dtype=torch.bfloat16,
        activation_checkpointing_llm=cfg.training.get("tp_knobs", {}).get("activation_checkpointing_llm", True),
        activation_checkpointing_m2p=cfg.training.get("tp_knobs", {}).get("activation_checkpointing_m2p", True),
        ckpt_skip_stride_llm=cfg.training.get("tp_knobs", {}).get("ckpt_skip_stride_llm", 0),
        ckpt_skip_stride_m2p=cfg.training.get("tp_knobs", {}).get("ckpt_skip_stride_m2p", 0),
        cpu_offload=cfg.training.get("tp_knobs", {}).get("cpu_offload", False),
        compile_hypernetwork=cfg.training.get("tp_knobs", {}).get("compile_hypernetwork", True),
    )
    all_gpu_stats("After model load")

    # Optional: disable SDPA Flash for A/B comparison via debug.no_flash.
    no_flash = cfg.get("debug", {}).get("no_flash", False)
    def make_sdpa_ctx():
        if no_flash:
            return sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        return nullcontext()

    # ------------------------------------------------------------------
    # 3. Dataset + DataLoader
    # ------------------------------------------------------------------
    from hydra.utils import get_original_cwd
    model_path = str(cfg.model.path)
    if not os.path.isabs(model_path):
        model_path = os.path.join(get_original_cwd(), model_path)
    pad_token_id = resolve_pad_token_id(model_path, tokenizer_cfg=cfg.tokenizer)
    vocab_size = model._vocab_size
    num_mem_token = model._num_mem_token

    train_dataset, val_dataset, collator = create_dataset_from_config(
        cfg, model_path, pad_token_id, num_mem_token,
    )

    local_batch_size = cfg.training.tp_batchsize.batch_size
    local_micro_batch_size = cfg.training.tp_batchsize.batch_size  # TP mode: micro_batch = batch_size (no pipeline splitting)
    if local_batch_size % local_micro_batch_size != 0:
        raise ValueError(
            f"local_batch_size ({local_batch_size}) must be a multiple of "
            f"local_micro_batch_size ({local_micro_batch_size})."
        )
    global_batch = local_batch_size * tp_cfg["data_parallel_size"]

    dp_size = tp_cfg["data_parallel_size"]
    dp_rank = tp_cfg["data_parallel_rank"]
    _samples_per_replica = math.ceil(len(train_dataset) / dp_size)
    _estimated_batches_per_epoch = _samples_per_replica // local_batch_size

    # Every rank loads data (pipeline_stage=0 / total_pipeline_stages=1 makes
    # the loader's is_first_stage=True). Inside a TP group all ranks have the
    # same dp_rank → DistributedSampler hands them identical indices.
    train_loader = PipelineDataLoader(
        dataset=train_dataset,
        batch_size=global_batch,
        micro_batch_size=local_micro_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        pipeline_stage=0,
        total_pipeline_stages=1,
        num_workers=cfg.data.get("num_workers", 4),
        shuffle=cfg.data.shuffle,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        batches_per_epoch=_estimated_batches_per_epoch,
        seed=cfg.seed.dataset,
    )

    if is_main_process_per_node():
        logger.info(
            f"Dataset: {len(train_dataset)} train samples (name={cfg.data.get('name')}), "
            f"global_batch={global_batch}, local_batch={local_batch_size}, "
            f"micro_batch={local_micro_batch_size}, num_mem_token={num_mem_token}"
        )

    # Create validation loader (if val_dataset exists)
    val_loader_for_eval = None
    if val_dataset is not None and len(val_dataset) > 0:
        # Parse evaluation batch size config (TP mode)
        _eval_tp_cfg = cfg.training.get("evaluation_tp_batchsize", {})
        _eval_tp_batch_size = int(_eval_tp_cfg.get("batch_size", local_batch_size)) if _eval_tp_cfg else local_batch_size
        _eval_tp_global_batch = _eval_tp_batch_size * dp_size

        val_loader_for_eval = PipelineDataLoader(
            dataset=val_dataset,
            batch_size=_eval_tp_global_batch,
            micro_batch_size=_eval_tp_batch_size,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            pipeline_stage=0,
            total_pipeline_stages=1,
            num_workers=cfg.data.get("num_workers", 4),
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collator,
            seed=cfg.seed.dataset,
        )
        if is_main_process_per_node():
            logger.info(f"  Validation: {len(val_dataset)} samples")

    # ------------------------------------------------------------------
    # 4. Optimizer + scheduler
    # ------------------------------------------------------------------
    grad_accum_steps = cfg.training.tp_batchsize.gradient_accumulation_steps

    # Initialize DetachState with runtime parameters now that batch config is known
    model.init_detach_state(
        local_batch_size=local_batch_size,
        micro_batch_size=local_micro_batch_size,
        tp_rank=tp_cfg["tp_rank"],
        tp_world=tp_cfg["tensor_parallel_size"],
        tp_process_group=tp_cfg.get("tp_process_group"),
        data_parallel_size=dp_size,
        grad_accum_steps=grad_accum_steps,
    )

    if train_loader.data_loader is not None:
        batches_per_epoch = len(train_loader.data_loader)
    else:
        batches_per_epoch = _estimated_batches_per_epoch
    steps_per_epoch = max(batches_per_epoch // grad_accum_steps, 1)

    max_steps_raw = cfg.training.max_steps
    _epoch_pat = re.compile(r"^([\d.]+)-epoch$", re.IGNORECASE)
    if isinstance(max_steps_raw, str):
        m = _epoch_pat.match(max_steps_raw.strip())
        if not m:
            raise ValueError(f"max_steps='{max_steps_raw}' not int or 'n-epoch' format")
        max_steps = max(int(float(m.group(1)) * steps_per_epoch), 1)
    else:
        max_steps = int(max_steps_raw)
    num_epochs = math.ceil(max_steps / steps_per_epoch)

    # Resolve warmup_steps: supports integer (exact steps) or float in (0, 1) (proportion of total steps)
    _raw_warmup = cfg.training.warmup_steps
    if isinstance(_raw_warmup, float) and 0 < _raw_warmup < 1:
        warmup_steps = int(_raw_warmup * max_steps)
    else:
        warmup_steps = int(_raw_warmup)

    if is_main_process_per_node():
        logger.info(
            f"Training plan: {num_epochs} epochs × {steps_per_epoch} steps/epoch "
            f"= up to {max_steps} optimizer steps (grad_accum={grad_accum_steps}, "
            f"warmup_steps={warmup_steps})"
        )

    optimizer, lr_scheduler = _create_optimizer_scheduler(
        model=model,
        num_training_steps=max_steps,
        learning_rate=cfg.optimizer.learning_rate,
        min_learning_rate=cfg.optimizer.get("min_learning_rate", 0.0),
        weight_decay=cfg.optimizer.weight_decay,
        beta1=cfg.optimizer.beta1,
        beta2=cfg.optimizer.beta2,
        eps=cfg.optimizer.eps,
        warmup_steps=warmup_steps,
        dp_group=tp_cfg.get("node_process_group"),
        use_zero1=cfg.training.get("tp_knobs", {}).get("zero1", True),
    )

    # ------------------------------------------------------------------
    # 5. Sync initial trainable params (TP-group + DP-group)
    # ------------------------------------------------------------------
    if is_main_process_per_node():
        logger.info("  [INIT] Broadcasting hypernetwork + metalora params from TP rank 0 …")
    # First: ensure all TP ranks within a group hold identical params
    # (no global seed is set before model construction, so random init differs)
    _broadcast_trainable_from_tp_rank0(model.hypernetwork, tp_cfg)
    _broadcast_metalora_from_tp_rank0(model.metalora, tp_cfg)
    # W-transform modules are replicated across TP ranks (all params must be identical)
    for _wt_name in ('w_transform_context', 'w_transform_conversation'):
        _wt_module = getattr(model, _wt_name, None)
        if _wt_module is not None:
            _broadcast_trainable_from_tp_rank0(_wt_module, tp_cfg)
    # SP broadcast: ensure all SP ranks within a group hold identical params
    if tp_cfg.get("sequence_parallel_size", 1) > 1:
        if is_main_process_per_node():
            logger.info("  [INIT] Broadcasting hypernetwork + metalora params from SP rank 0 …")
        _broadcast_trainable_from_sp_rank0(model.hypernetwork, tp_cfg)
        _broadcast_metalora_from_sp_rank0(model.metalora, tp_cfg)
        for _wt_name in ('w_transform_context', 'w_transform_conversation'):
            _wt_module = getattr(model, _wt_name, None)
            if _wt_module is not None:
                _broadcast_trainable_from_sp_rank0(_wt_module, tp_cfg)
    if is_main_process_per_node():
        logger.info("  [INIT] Broadcasting hypernetwork + metalora params from DP rank 0 …")
    # Then: ensure all DP replicas hold identical params
    _broadcast_trainable_from_dp_rank0(model.hypernetwork, tp_cfg)
    _broadcast_metalora_from_dp_rank0(model.metalora, tp_cfg)
    for _wt_name in ('w_transform_context', 'w_transform_conversation'):
        _wt_module = getattr(model, _wt_name, None)
        if _wt_module is not None:
            _broadcast_trainable_from_dp_rank0(_wt_module, tp_cfg)
    # Also broadcast mem_tokens (zero-init is naturally consistent, but
    # broadcast anyway for safety and resume correctness)
    _broadcast_mem_tokens(model, tp_cfg)

    # Initialize variables that may be set by resume
    t_start = 0
    global_step = 0
    start_epoch = 0
    running_loss = 0.0
    ema_time_per_step = 0.0
    total_context_tokens = 0
    total_conv_total_tokens = 0
    total_conv_valid_tokens = 0

    # ------------------------------------------------------------------
    # 5.5 Resume from checkpoint (PP-compatible format)
    # ------------------------------------------------------------------
    from hydra.utils import get_original_cwd
    # Use cfg.mode first (same as PP), fall back to env var
    training_mode = cfg.get("mode", os.environ.get("TRAINING_MODE", "pretrain"))
    exp_name = os.environ.get("EXP_NAME", cfg.get("name", "default"))
    annealing_name = os.environ.get("ANNEALING_NAME", "")
    sft_name = os.environ.get("SFT_NAME", "")
    run_name = build_checkpoint_run_name(exp_name, training_mode, annealing_name, sft_name)

    save_total_limit = int(cfg.training.get("save_total_limit", 3))
    forever_save_steps = resolve_forever_save_steps(cfg.training.get("forever_save_steps", -1))

    # Determine resume checkpoint
    resume_checkpoint_dir = None
    # Inference modes never have/need optimizer state — load weights only.
    load_model_only_flag = (
        os.environ.get("MEMORY_QA_GEN", "") == "1"
        or os.environ.get("SQUAD_QA_GEN", "") == "1"
    )
    # warm_start_from is model-only initialization for a new training run.
    # resume_from remains strict resume and requires optimizer/scheduler state.
    _warm_start_raw = cfg.training.get("warm_start_from", None)
    _resume_from_raw = cfg.training.get("resume_from", "latest")
    if _warm_start_raw is not None and str(_warm_start_raw).lower() not in ("null", "none", ""):
        if _resume_from_raw is not None and str(_resume_from_raw).lower() not in ("null", "none", ""):
            raise ValueError(
                "training.warm_start_from is model-only initialization and cannot be used "
                "together with training.resume_from. Set training.resume_from=null when "
                "using warm_start_from."
            )
        resume_checkpoint_dir = str(_warm_start_raw)
        if not os.path.isabs(resume_checkpoint_dir):
            resume_checkpoint_dir = os.path.join(get_original_cwd(), resume_checkpoint_dir)
        load_model_only_flag = True
        if is_main_process_per_node():
            logger.info(f"  [WarmStart] Loading model-only from: {resume_checkpoint_dir}")

    # Read resume_from config (same key as PP: supports "latest", path, or null/None)
    if _resume_from_raw is not None and str(_resume_from_raw).lower() not in ("null", "none", ""):
        if str(_resume_from_raw).lower() == "latest":
            # Auto-detect latest checkpoint for this run
            latest = get_latest_checkpoint(run_name)
            if latest is not None:
                resume_checkpoint_dir = latest
                if is_main_process_per_node():
                    logger.info(f"  [Resume] Found latest checkpoint: {resume_checkpoint_dir}")
        else:
            # Explicit path
            resume_checkpoint_dir = str(_resume_from_raw)
            if not os.path.isabs(resume_checkpoint_dir):
                resume_checkpoint_dir = os.path.join(get_original_cwd(), resume_checkpoint_dir)

        # If no checkpoint found yet, check for prior-stage final checkpoint
        if resume_checkpoint_dir is None:
            if is_main_process_per_node():
                logger.info(f"  [Resume] No existing checkpoint found for run '{run_name}'.")
            # Check for prior-stage final checkpoint (for annealing/SFT)
            if training_mode == "pretrain_annealing":
                prior = get_pretrain_final_checkpoint(exp_name)
                if prior is not None:
                    resume_checkpoint_dir = prior
                    load_model_only_flag = True
                    if is_main_process_per_node():
                        logger.info(f"  [Resume] pretrain_annealing mode: loading pretrain final checkpoint: {prior}")
            elif training_mode == "sft":
                if annealing_name == "null":
                    # annealing_name=null: load pretrain final directly, skip annealing
                    _pretrain_final = get_pretrain_final_checkpoint(exp_name)
                    if _pretrain_final is not None:
                        resume_checkpoint_dir = _pretrain_final
                        load_model_only_flag = True
                        if is_main_process_per_node():
                            logger.info(f"  [Resume] SFT mode (annealing_name=null): loading pretrain final checkpoint: {_pretrain_final}")
                    else:
                        raise RuntimeError(
                            f"SFT mode with annealing_name=null: no pretrain final checkpoint found for '{exp_name}'. "
                            f"Expected at: checkpoint/{exp_name}/pretrain/final/"
                        )
                else:
                    # Default: load pretrain_annealing final checkpoint
                    prior = get_pretrain_annealing_final_checkpoint(exp_name, annealing_name)
                    if prior is not None:
                        resume_checkpoint_dir = prior
                        load_model_only_flag = True
                        if is_main_process_per_node():
                            logger.info(f"  [Resume] SFT mode: loading pretrain_annealing final checkpoint: {prior}")
                    else:
                        raise RuntimeError(
                            f"SFT mode: no pretrain_annealing final checkpoint found for '{exp_name}' "
                            f"with annealing_name='{annealing_name}'. "
                            f"Expected at: checkpoint/{exp_name}/pretrain_annealing/{annealing_name}/final/. "
                            f"Use --annealing_name null to load pretrain final checkpoint directly."
                        )

    _resume_metadata = {}  # Store metadata for wandb resume
    # In evaluation modes, we only need model weights — skip optimizer/scheduler loading
    _eval_baseline_env = os.environ.get("EVALUATION_BASELINE", "") == "1"
    _eval_export_env = os.environ.get("EVALUATION_EXPORT_LORA", "") == "1"
    if (_eval_baseline_env or _eval_export_env) and not load_model_only_flag:
        load_model_only_flag = True
        if is_main_process_per_node():
            logger.info("  [Resume] Evaluation mode detected — loading model-only (skipping optimizer/scheduler).")
    if resume_checkpoint_dir is not None:
        if load_model_only_flag:
            if is_main_process_per_node():
                logger.info(f"  [Resume] Loading model-only from: {resume_checkpoint_dir}")
            _resume_metadata = _tp_load_model_only(model, resume_checkpoint_dir)
            barrier()
            if is_main_process_per_node():
                _pt_step = _resume_metadata.get("global_step", "?")
                logger.info(f"  [Resume] Loaded model from step {_pt_step}. "
                            f"Optimizer/scheduler start fresh.")
        else:
            if is_main_process_per_node():
                logger.info(f"  [Resume] Loading checkpoint from: {resume_checkpoint_dir}")
            _resume_metadata = _tp_load_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=resume_checkpoint_dir,
                my_device=my_device,
                local_rank=dist.get_rank() % tp_cfg["total_gpus"],
                tp_cfg=tp_cfg,
            )
            global_step = _resume_metadata.get("global_step", 0)
            start_epoch = _resume_metadata.get("epoch", 0)
            running_loss = _resume_metadata.get("running_loss", 0.0)
            ema_time_per_step = _resume_metadata.get("ema_time_per_step", 0.0)
            # Restore token stats (same as PP)
            total_context_tokens = _resume_metadata.get("total_context_tokens", 0)
            total_conv_total_tokens = _resume_metadata.get("total_conv_total_tokens", 0)
            total_conv_valid_tokens = _resume_metadata.get("total_conv_valid_tokens", 0)
            # Adjust t_start for seamless wall-clock time (same as PP)
            _saved_elapsed = _resume_metadata.get("elapsed_time", 0.0)
            if _saved_elapsed > 0:
                t_start = time.time() - _saved_elapsed
            # Restore dataloader state for reproducibility (same as PP)
            _dl_state = _resume_metadata.get("dataloader_state", None)
            if _dl_state is not None and hasattr(train_loader, "load_state_dict"):
                train_loader.load_state_dict(_dl_state)
            barrier()
            if is_main_process_per_node():
                logger.info(
                    f"  [Resume] Restored from step {global_step}, epoch {start_epoch}, "
                    f"elapsed={_saved_elapsed:.1f}s"
                )
        # Re-broadcast params after loading (ensure all TP + SP + DP replicas are in sync)
        _broadcast_trainable_from_tp_rank0(model.hypernetwork, tp_cfg)
        _broadcast_metalora_from_tp_rank0(model.metalora, tp_cfg)
        for _wt_name in ('w_transform_context', 'w_transform_conversation'):
            _wt_module = getattr(model, _wt_name, None)
            if _wt_module is not None:
                _broadcast_trainable_from_tp_rank0(_wt_module, tp_cfg)
        if tp_cfg.get("sequence_parallel_size", 1) > 1:
            _broadcast_trainable_from_sp_rank0(model.hypernetwork, tp_cfg)
            _broadcast_metalora_from_sp_rank0(model.metalora, tp_cfg)
            for _wt_name in ('w_transform_context', 'w_transform_conversation'):
                _wt_module = getattr(model, _wt_name, None)
                if _wt_module is not None:
                    _broadcast_trainable_from_sp_rank0(_wt_module, tp_cfg)
        _broadcast_trainable_from_dp_rank0(model.hypernetwork, tp_cfg)
        _broadcast_metalora_from_dp_rank0(model.metalora, tp_cfg)
        for _wt_name in ('w_transform_context', 'w_transform_conversation'):
            _wt_module = getattr(model, _wt_name, None)
            if _wt_module is not None:
                _broadcast_trainable_from_dp_rank0(_wt_module, tp_cfg)
        _broadcast_mem_tokens(model, tp_cfg)

        # Load detach_state from checkpoint (each rank loads its own shard)
        if model.detach_state is not None and not load_model_only_flag:
            ds_path = os.path.join(
                resume_checkpoint_dir, "detach_state",
                f"dp{tp_cfg['data_parallel_rank']}_tp{tp_cfg['tp_rank']}.pt"
            )
            if os.path.exists(ds_path):
                ds_state = torch.load(ds_path, map_location=my_device)
                model.detach_state.load_state_dict(ds_state)
                if is_main_process_per_node():
                    logger.info(f"  [DetachState] Loaded from {ds_path}")
            else:
                if is_main_process_per_node():
                    logger.info(
                        f"  [DetachState] No checkpoint found at {ds_path} — "
                        f"starting fresh (zero wdict)."
                    )

    # ==================================================================
    # Memory-QA free-form generation (inference). Branch HERE — right after the
    # model is loaded — so we skip all training-only setup (optimizer/profiler/
    # peak-memory) which is irrelevant and can trip on inference runs.
    # ==================================================================
    if os.environ.get("MEMORY_QA_GEN", "") == "1":
        if os.environ.get("MEMORY_QA_ICL", "") == "1":
            from eval_memory_gen import run_memory_qa_icl
            run_memory_qa_icl(model, cfg, tp_cfg, my_device)
        else:
            from eval_memory_gen import run_memory_qa_gen
            run_memory_qa_gen(model, cfg, tp_cfg, my_device)
        if is_main_process_per_node():
            logger.info("[MEMORY_QA_GEN] done. Exiting.")
        cleanup_distributed()
        os._exit(0)

    if os.environ.get("SQUAD_QA_GEN", "") == "1":
        from eval_squad_gen import run_squad_qa_gen
        run_squad_qa_gen(model, cfg, tp_cfg, my_device)
        if is_main_process_per_node():
            logger.info("[SQUAD_QA_GEN] done. Exiting.")
        cleanup_distributed()
        os._exit(0)
    # ------------------------------------------------------------------
    # 5.6 Resolve config selections (needed for wandb tags + consistency check)
    # ------------------------------------------------------------------
    _config_env_keys = {
        "model": "MODEL_CONFIG",
        "m2p_transformer": "M2P_TRANSFORMER_CONFIG",
        "training": "TRAINING_CONFIG",
        "optimizer": "OPTIMIZER_CONFIG",
        "data": "DATA_CONFIG",
        "debug": "DEBUG_CONFIG",
        "tokenizer": "TOKENIZER_CONFIG",
        "detach_state": "DETACH_STATE_CONFIG",
    }
    _resolved_configs = {}
    for _key, _env_var in _config_env_keys.items():
        _val = os.environ.get(_env_var, "")
        if not _val:
            _val = ""  # TP doesn't parse yaml defaults; env vars are authoritative
        _resolved_configs[_key] = _val

    # ------------------------------------------------------------------
    # 6. wandb (optional)
    # ------------------------------------------------------------------
    use_wandb = False
    wandb_run_id = None
    _wandb_disabled = (
        os.environ.get("WANDB_MODE", "online") in ("disabled", "offline")
        or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
    )
    if is_main_process() and wandb is not None and not _wandb_disabled:
        wandb_cfg = cfg.get("wandb", {})
        wandb_project = os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project", "SHINE_V2")
        wandb_run_name = os.environ.get("WANDB_NAME") or wandb_cfg.get("run_name")
        wandb_entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")

        # Build wandb tags for filtering (aligned with PP format)
        # Note: wandb tags have a 64-character limit
        wandb_tags = [training_mode, f"parallel=tp"]
        if exp_name:
            _tag_name = exp_name[:64] if len(exp_name) > 64 else exp_name
            wandb_tags.append(_tag_name)
        if annealing_name and annealing_name != "null":
            _ann_tag = annealing_name[:64] if len(annealing_name) > 64 else annealing_name
            wandb_tags.append(_ann_tag)
        if annealing_name == "null":
            wandb_tags.append("no_annealing")
        # Add config module selections as tags (same as PP)
        for _cfg_key, _cfg_val in _resolved_configs.items():
            _cfg_tag = f"{_cfg_key}={_cfg_val}"
            if len(_cfg_tag) > 64:
                _cfg_tag = _cfg_tag[:64]
            wandb_tags.append(_cfg_tag)
        # TP-specific tags for parallel topology
        wandb_tags.append(f"tp={tp_cfg['tensor_parallel_size']}")
        wandb_tags.append(f"dp={tp_cfg['data_parallel_size']}")

        # Build wandb config
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config["experiment_name"] = exp_name
        wandb_config["training_mode"] = training_mode
        wandb_config["annealing_name"] = annealing_name if annealing_name else None
        wandb_config["sft_name"] = sft_name if training_mode == "sft" else None
        wandb_config["launch_cmd"] = os.environ.get("LAUNCH_CMD", "")
        # Store resolved config selections prominently in wandb config
        wandb_config["config_selections"] = _resolved_configs
        # Parallel mode fields for unified filtering across PP/TP runs
        wandb_config["parallel_mode"] = "tp"
        wandb_config["tp_size"] = tp_cfg["tensor_parallel_size"]
        wandb_config["dp_size"] = tp_cfg["data_parallel_size"]
        wandb_config["total_gpus"] = cfg.parallel.total_gpus
        wandb_config["world_size"] = dist.get_world_size() if dist.is_initialized() else 1
        wandb_config["num_nodes"] = wandb_config["world_size"] // cfg.parallel.total_gpus

        # Build wandb notes with config summary for quick identification in UI
        _notes_lines = [f"mode={training_mode}"]
        for _cfg_key, _cfg_val in _resolved_configs.items():
            _notes_lines.append(f"{_cfg_key}={_cfg_val}")
        _notes_lines.append(f"tp={tp_cfg['tensor_parallel_size']} dp={tp_cfg['data_parallel_size']}")
        wandb_notes = " | ".join(_notes_lines)

        try:
            # If resuming, use the saved wandb run_id to continue the same run
            _saved_wandb_id = _resume_metadata.get("wandb_run_id", None)
            if _saved_wandb_id is not None and resume_checkpoint_dir is not None and not load_model_only_flag:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    entity=wandb_entity,
                    id=_saved_wandb_id,
                    resume="must",
                    tags=wandb_tags,
                    config=wandb_config,
                    notes=wandb_notes,
                    reinit=True,
                )
                logger.info(f"wandb RESUMED: project={wandb_project}, run_id={_saved_wandb_id}")
            else:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    entity=wandb_entity,
                    config=wandb_config,
                    tags=wandb_tags,
                    notes=wandb_notes,
                    reinit=True,
                )
            use_wandb = True
            wandb_run_id = wandb.run.id
            # Define wall_time as custom x-axis for seamless resume
            wandb.define_metric("wall_time")
            wandb.define_metric("train/*", step_metric="wall_time")
            wandb.define_metric("eval/*", step_metric="wall_time")
            logger.info(f"wandb logging enabled: project={wandb_project} run={wandb.run.name} id={wandb_run_id}")
        except Exception as e:
            # wandb.init can fail on hosts where sentry_sdk is missing or the
            # network is unreachable — neither is a reason to abort training.
            logger.warning(f"wandb.init failed: {e!r}; continuing without wandb")
            use_wandb = False

    # ------------------------------------------------------------------
    # 7. Monitoring setup
    # ------------------------------------------------------------------
    # Debug loggers (same as PP: per-category log files)
    _log_subdir = os.environ.get("LOG_SUBDIR", "")
    _log_dir = os.path.join(get_original_cwd(), "logs", _log_subdir) if _log_subdir else os.path.join(get_original_cwd(), "logs")
    _node_rank_for_log = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0")))
    from datetime import datetime
    _session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _debug_categories, _debug_log_paths, _node0_only_categories = setup_debug_loggers(
        log_dir=_log_dir,
        node_rank=_node_rank_for_log,
        session_id=_session_id,
    )
    if is_main_process_per_node():
        logger.info(f"Debug logs directory: {_log_dir} (categories: {list(_debug_categories.keys())})")

    _report_peak_memory("Before training")

    # NaN/Inf detection
    _monitor_cfg = cfg.get("debug", {}).get("monitor", {})
    monitor_nan_inf = bool(_monitor_cfg.get("nan_inf", True))
    monitor_loss_spike = bool(_monitor_cfg.get("loss_spike", False))
    nan_inf_tracker = NanInfTracker() if monitor_nan_inf else None
    nan_inf_stop_steps = int(_monitor_cfg.get("nan_inf_stop_steps", -1))

    # Grad norm schedule (read from debug.monitor, same as PP)
    grad_norm_sched = DebugSchedule(
        _monitor_cfg.get("grad_norm_steps", -1), "grad_norm_steps")
    # Param norm schedule
    param_norm_sched = DebugSchedule(
        _monitor_cfg.get("param_norm_steps", -1), "param_norm_steps")
    # Generated LoRA norm schedule
    gen_lora_norm_sched = DebugSchedule(
        _monitor_cfg.get("gen_lora_norm_steps", -1), "gen_lora_norm_steps")

    # Eval schedule
    eval_steps_raw = cfg.training.get("eval_steps", -1)
    eval_sched = DebugSchedule(eval_steps_raw, "eval_steps")

    # Generation-eval schedule: per-type free-decode accuracy during training.
    # Off by default; set MEMORY_QA_GEN_EVERY=500 to run every 500 steps.
    # MEMORY_QA_GEN_RECALL=1 additionally runs the deferred-recall probe (answer
    # each segment's QA from the FULL accumulated W -> isolates detach_state).
    gen_eval_sched = DebugSchedule(int(os.environ.get("MEMORY_QA_GEN_EVERY", "-1")), "gen_eval")
    _gen_eval_recall = os.environ.get("MEMORY_QA_GEN_RECALL", "") == "1"
    _gen_eval_nhist = int(os.environ.get("MEMORY_QA_GEN_NUM", "8"))
    if (int(os.environ.get("MEMORY_QA_GEN_EVERY", "-1")) > 0
            and tp_cfg["tensor_parallel_size"] > 1 and is_main_process_per_node()):
        logger.warning(
            "[gen_eval] MEMORY_QA_GEN_EVERY set but tensor_parallel_size>1 — "
            "in-loop generation eval is main-rank-only and would deadlock under "
            "TP collectives, so it will be SKIPPED. Use pure DP (tp_size=1) to enable it.")
    # Debug schedules (same as PP)
    log_peak_memory_sched = DebugSchedule(
        cfg.get("debug", {}).get("log_peak_memory_steps", -1), "log_peak_memory_steps")
    log_train_detail_sched = DebugSchedule(
        cfg.get("debug", {}).get("log_train_detail_steps", -1), "log_train_detail_steps")
    check_consistency_sched = DebugSchedule(
        cfg.get("debug", {}).get("check_consistency_across_nodes", -1), "check_consistency_across_nodes")

    if is_main_process_per_node():
        logger.info(
            f"  Monitoring: nan_inf={monitor_nan_inf}, loss_spike={monitor_loss_spike}, "
            f"nan_inf_stop_steps={nan_inf_stop_steps}, "
            f"grad_norm_sched={grad_norm_sched}, param_norm_sched={param_norm_sched}, "
            f"gen_lora_norm_sched={gen_lora_norm_sched}, eval_sched={eval_sched}"
        )
        logger.info(f"  Debug schedules: {log_peak_memory_sched}, {log_train_detail_sched}, {check_consistency_sched}")

    # Token counters are initialized in section 8 (training loop init) with resume-awareness

    # Loss running average (for loss spike detection)
    loss_running_avg = None
    _loss_avg_alpha = 0.01

    # Detail log accumulator (same as PP: records per-step token details for debugging)
    detail_log_accumulator = []
    # PPL-threshold-triggered detail log accumulator
    detail_log_accumulator_ppl = []
    log_train_detail_ppl_threshold = float(cfg.get("debug", {}).get("log_train_detail_ppl_threshold", -1))
    # Tokenizer for detail logging (lazy-loaded only when needed)
    _detail_tokenizer = None

    # ------------------------------------------------------------------
    # 7b. Distillation setup
    # ------------------------------------------------------------------
    from utils.mydistill import create_distill_loss_fn
    distill_cfg = cfg.training.get("distill", None)
    distill_loss_fn = create_distill_loss_fn(distill_cfg)
    if distill_loss_fn is not None:
        # Set SP group for SP-aware distillation loss computation
        _sp_group = tp_cfg.get("sp_process_group")
        if _sp_group is not None and tp_cfg.get("sequence_parallel_size", 1) > 1:
            distill_loss_fn.set_sp_group(_sp_group)
        if is_main_process_per_node():
            logger.info(
                f"  Distillation enabled: mode={distill_cfg.mode}, "
                f"loss_type={distill_cfg.loss_type}, "
                f"coefficient={distill_cfg.coefficient}, "
                f"temperature={distill_cfg.get('temperature', 'N/A')}"
            )
    else:
        if is_main_process_per_node():
            logger.info("  Distillation: disabled")

    # ------------------------------------------------------------------
    # 7c. Config consistency check on resume (same as PP)
    # ------------------------------------------------------------------
    # _resolved_configs was already computed in section 5.6
    _force_overwrite = os.environ.get("FORCE_OVERWRITE", "") == "1"
    _config_changes = {}
    if _resume_metadata and not load_model_only_flag:
        _saved_configs = _resume_metadata.get("config_selections", None)
        if _saved_configs is not None:
            for _key in set(list(_saved_configs.keys()) + list(_resolved_configs.keys())):
                _old_val = _saved_configs.get(_key, "<not set>")
                _new_val = _resolved_configs.get(_key, "<not set>")
                if _old_val != _new_val:
                    _config_changes[_key] = (_old_val, _new_val)
            if _config_changes:
                _diff_msg = "\n".join(
                    f"    {k}: '{old}' -> '{new}'" for k, (old, new) in sorted(_config_changes.items())
                )
                _diff_msg += f"\n    [current launch_cmd]: {os.environ.get('LAUNCH_CMD', '')}"
                if not _force_overwrite:
                    raise RuntimeError(
                        f"\n{'='*80}\n"
                        f"[FATAL] Config mismatch on resume!\n"
                        f"The following config selections differ from the checkpoint:\n"
                        f"{_diff_msg}\n\n"
                        f"To force resume with changed configs, add --force_overwrite to launch_cluster.sh.\n"
                        f"{'='*80}"
                    )
                else:
                    if is_main_process_per_node():
                        logger.warning(
                            f"  [Resume] Config mismatch detected (--force_overwrite enabled):\n{_diff_msg}"
                        )

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    gradient_clipping = cfg.training.gradient_clipping
    logging_steps = int(cfg.training.logging_steps)
    save_sched = DebugSchedule(cfg.training.get("save_steps", -1), "save_steps")
    global_step = 0 if resume_checkpoint_dir is None or load_model_only_flag else global_step
    micro_step = 0
    running_loss = 0.0 if resume_checkpoint_dir is None or load_model_only_flag else running_loss
    running_distill_loss = 0.0
    # Train answer-token accuracy window (computed only on to-be-logged steps to
    # keep overhead negligible; SUM-reduced over DP in the logging block).
    _train_acc_on = os.environ.get("MEMORY_QA_TRAIN_ACC", "1") != "0"
    _train_acc_corr = 0
    _train_acc_tot = 0
    _train_ans_corr = 0
    _train_ans_tot = 0
    running_regu_sq_norm = 0.0
    running_reset_ratio = 0.0
    running_mean_update_step = 0.0
    running_repo_reset_count = 0  # Count of repo-change resets
    # Restore _prev_repo from checkpoint metadata (for repo-change reset continuity)
    _prev_repo = _resume_metadata.get("prev_repo", None) if (resume_checkpoint_dir is not None and not load_model_only_flag) else None
    accum_loss = 0.0
    # t_start: if resuming with elapsed_time, it was already adjusted in section 5.5
    # Only reset to now if not resuming or loading model-only
    if resume_checkpoint_dir is None or load_model_only_flag or t_start == 0:
        t_start = time.time()
    step_start_time = 0.0  # Will be set after first step (same as PP)
    ema_time_per_step = 0.0 if resume_checkpoint_dir is None or load_model_only_flag else ema_time_per_step
    ema_alpha = 0.1
    start_epoch = 0 if resume_checkpoint_dir is None or load_model_only_flag else start_epoch
    # Epoch-level tracking (same as PP)
    epoch_loss_sum = 0.0
    epoch_steps = 0
    # Per-window token counters (reset after each logging window, same as PP)
    step_context_tokens = 0
    step_conv_total_tokens = 0
    step_conv_valid_tokens = 0
    # Token stats: restore from resume if available
    if resume_checkpoint_dir is not None and not load_model_only_flag:
        # Already restored in section 5.5
        pass
    else:
        total_context_tokens = 0
        total_conv_total_tokens = 0
        total_conv_valid_tokens = 0
    # wandb_run_id is set in the wandb init block above; keep it if already set
    if not use_wandb:
        wandb_run_id = None

    model.train()
    barrier()

    # --- [DEBUG] Initial parameter consistency check (after broadcast) ---
    if check_consistency_sched.should_run(0):
        if is_main_process_per_node():
            logger.info("  [DEBUG] Running initial DP parameter consistency check (step 0)...")
        _compat_cfg = {
            "dp_process_group": tp_cfg.get("dp_process_group"),
            "data_parallel_rank": tp_cfg["data_parallel_rank"],
            "stage": 0,
            "total_stages": 1,
            "node_process_group": None,
        }
        check_dp_param_consistency(
            model=model,
            my_device=my_device,
            parallel_cfg=_compat_cfg,
            global_step=0,
        )
        # Also check TP-group consistency for replicated params (w_transform, hypernetwork)
        check_tp_param_consistency(
            model=model,
            my_device=my_device,
            tp_cfg=tp_cfg,
            global_step=0,
        )
        # Also check SP-group consistency for replicated params
        check_sp_param_consistency(
            model=model,
            my_device=my_device,
            tp_cfg=tp_cfg,
            global_step=0,
        )

    # --- [INIT] Profiler (zero overhead when disabled) ---
    _profiler = TrainingProfiler(
        profiler_cfg=cfg.debug.get("profiler", {}),
        log_dir=_log_dir,
    )
    _profiler_ctx = _profiler.context()
    _profiler_ctx.__enter__()

    # ==================================================================
    # 8a. TP Forward Comparison — TP Load mode
    # ==================================================================
    _tp_fwd_compare_cfg = cfg.get("debug", {}).get("tp_forward_compare", None)
    _tp_fwd_compare_mode = _tp_fwd_compare_cfg.get("mode", None) if _tp_fwd_compare_cfg else None
    if _tp_fwd_compare_mode == "tp_load":
        from utils.tp_forward_compare import run_tp_load_mode
        run_tp_load_mode(
            model=model,
            train_loader=train_loader,
            cfg=cfg,
            tp_cfg=tp_cfg,
            my_device=my_device,
            make_sdpa_ctx=make_sdpa_ctx,
        )
        if is_main_process_per_node():
            logger.info("TP Load mode complete. Exiting.")
        _profiler_ctx.__exit__(None, None, None)
        cleanup_distributed()
        # Use os._exit to avoid segfault during Python object destruction
        # (CUDA/NCCL cleanup race condition). The results are already saved
        # and fsync'd at this point.
        os._exit(0)

    # ==================================================================
    # 8b. Evaluation Baseline Mode — base LLM only, no hypernetwork
    # ==================================================================
    _evaluation_baseline = os.environ.get("EVALUATION_BASELINE", "") == "1"
    if _evaluation_baseline:
        if is_main_process_per_node():
            logger.info("=" * 60)
            logger.info("  EVALUATION BASELINE MODE")
            logger.info("  Running base LLM evaluation (no hypernetwork/LoRA/detach_state)")
            logger.info("=" * 60)

        if val_loader_for_eval is None:
            if is_main_process_per_node():
                logger.error("  [Baseline] No validation set available. Exiting.")
            _profiler_ctx.__exit__(None, None, None)
            cleanup_distributed()
            os._exit(1)

        _tp_run_evaluation(
            model=model,
            val_loader=val_loader_for_eval,
            tp_cfg=tp_cfg,
            sdpa_ctx_factory=make_sdpa_ctx,
            global_step=0,
            use_wandb=use_wandb,
            t_start=t_start,
            max_steps=max_steps,
            ema_time_per_step=0.0,
            distill_loss_fn=None,
            baseline_mode=True,
        )

        if is_main_process() and use_wandb and wandb is not None:
            wandb.finish()
        if is_main_process_per_node():
            logger.info("  [Baseline] Evaluation baseline complete. Exiting.")
        _profiler_ctx.__exit__(None, None, None)
        cleanup_distributed()
        os._exit(0)

    # ==================================================================
    # 8c. Evaluation Export LoRA Mode — export per-repo PEFT LoRA adapters
    # ==================================================================
    _evaluation_export_lora = os.environ.get("EVALUATION_EXPORT_LORA", "") == "1"
    if _evaluation_export_lora:
        _max_traj_str = os.environ.get("EXPORT_LORA_MAX_TRAJ", "")
        if not _max_traj_str:
            if is_main_process_per_node():
                logger.error("EXPORT_LORA_MAX_TRAJ must be set when EVALUATION_EXPORT_LORA=1")
            _profiler_ctx.__exit__(None, None, None)
            cleanup_distributed()
            os._exit(1)
        _max_traj = int(_max_traj_str)

        if is_main_process_per_node():
            logger.info("=" * 60)
            logger.info("  EVALUATION EXPORT LORA MODE (TP)")
            logger.info(f"  Max trajectories per repo: {_max_traj}")
            logger.info("=" * 60)

        if val_dataset is None or len(val_dataset) == 0:
            if is_main_process_per_node():
                logger.error("  [ExportLoRA] No validation set available. Exiting.")
            _profiler_ctx.__exit__(None, None, None)
            cleanup_distributed()
            os._exit(1)

        from collections import defaultdict
        from utils.myloradict import detach_loradict, concat_loradict, loradict_to_peft

        # Step 1: Group val_dataset by repo
        # Validate that dataset has 'repo' field
        _first_sample = val_dataset[0]
        _has_repo_field = (
            "repo" in _first_sample
            or (isinstance(_first_sample.get("extra_info"), dict) and "repo" in _first_sample["extra_info"])
        )
        if not _has_repo_field:
            if is_main_process_per_node():
                logger.error(
                    "  [ExportLoRA] Dataset does not have a 'repo' field. "
                    "evaluation_export_lora mode requires the dataset to provide "
                    "a 'repo' field (either as sample['repo'] or sample['extra_info']['repo']). "
                    "Please use a dataset with repo information (e.g. trajectory_pro_transfer)."
                )
            _profiler_ctx.__exit__(None, None, None)
            cleanup_distributed()
            os._exit(1)

        repo_to_indices = defaultdict(list)
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            repo = sample.get("repo", sample.get("extra_info", {}).get("repo"))
            repo_to_indices[repo].append(idx)

        # Step 2: Assign repos to DP ranks (round-robin)
        all_repos = sorted(repo_to_indices.keys())
        repos_for_this_rank = [all_repos[i] for i in range(dp_rank, len(all_repos), dp_size)]

        if is_main_process_per_node():
            logger.info(
                f"  [ExportLoRA] Total repos: {len(all_repos)}, "
                f"this DP rank ({dp_rank}/{dp_size}): {len(repos_for_this_rank)} repos"
            )

        # Step 3: Process each repo
        model.eval()
        _dataset_name = cfg.data.get("name", "unknown_dataset")
        _export_save_base = os.path.join("./save", exp_name, _dataset_name)
        all_repo_results = {}

        for repo_idx, repo_name in enumerate(repos_for_this_rank):
            indices = repo_to_indices[repo_name][:_max_traj]

            if is_main_process_per_node():
                logger.info(
                    f"  [ExportLoRA] Processing repo {repo_idx+1}/{len(repos_for_this_rank)}: "
                    f"{repo_name} ({len(indices)} trajectories)"
                )

            # Reset detach_state for this repo
            if model.detach_state is not None:
                model.detach_state.reset()
                model.detach_state.init_steps()

            repo_loradicts = []
            repo_losses = []
            repo_baseline_losses = []

            for traj_idx, sample_idx in enumerate(indices):
                sample = val_dataset[sample_idx]
                # Use collator to process single sample into batch format
                batch_list = collator([sample])
                if isinstance(batch_list, list):
                    primary = batch_list[0]
                else:
                    primary = batch_list

                # Move tensors to device
                mb_dev = {}
                for k, v in primary.items():
                    if isinstance(v, torch.Tensor):
                        mb_dev[k] = v.to(my_device, non_blocking=True)
                    else:
                        mb_dev[k] = v

                with torch.no_grad(), make_sdpa_ctx():
                    # Full forward: produces loss AND caches loradicts
                    result, regu_sq_norm, _ = model(
                        context_ids=mb_dev["context_ids"],
                        context_lengths=mb_dev["context_lengths"],
                        conversation_ids=mb_dev["conversation_ids"],
                        labels=mb_dev["labels"],
                        grad_accum_steps=1,
                    )

                    # Extract loss value
                    if isinstance(result, tuple):
                        loss_val = result[0].item()
                    else:
                        loss_val = result.item()
                    repo_losses.append(loss_val)

                    # Get the generated loradict (WITHOUT metalora)
                    # metalora will be added once at the end
                    gen_loradict = model.get_last_generated_loradict()
                    if gen_loradict is not None:
                        repo_loradicts.append(detach_loradict(gen_loradict))

                    # Baseline forward: raw LLM without any loradict/metalora
                    baseline_loss = model.compute_loss(
                        input_ids=mb_dev["conversation_ids"],
                        labels=mb_dev["labels"],
                        loradict=None,
                        nograd_loradict=None,
                        nograd_wdict=None,
                    )
                    baseline_loss_val = baseline_loss.detach().float().item()
                    repo_baseline_losses.append(baseline_loss_val)

                if is_main_process_per_node():
                    import math as _math_log
                    _ppl_val = _math_log.exp(loss_val) if loss_val < 20 else float('inf')
                    _bl_ppl_val = _math_log.exp(baseline_loss_val) if baseline_loss_val < 20 else float('inf')
                    logger.info(
                        f"    [ExportLoRA] Repo {repo_idx+1}/{len(repos_for_this_rank)} "
                        f"({repo_name}) | Traj {traj_idx+1}/{len(indices)} | "
                        f"loss={loss_val:.4f} ppl={_ppl_val:.2f} | "
                        f"baseline_loss={baseline_loss_val:.4f} baseline_ppl={_bl_ppl_val:.2f}"
                    )

                # Update detach_state (write loradict for next trajectory)
                model.post_backward_detach_state(grad_accum_steps=1)
                if model.detach_state is not None:
                    model.detach_state.update_steps(0)
                    model.detach_state.set_last_sq_norms([regu_sq_norm])
                    model.detach_state.maybe_reset_slice(0)

            # Concat all loradicts for this repo, then add metalora once
            if repo_loradicts:
                # Build per-layer concat of generated loradicts (without metalora)
                all_layer_indices = sorted(repo_loradicts[0].keys())
                metalora = model.get_metalora()
                merged_loradict = {}
                for layer_idx in all_layer_indices:
                    layer_loradicts = [ld[layer_idx] for ld in repo_loradicts if ld.get(layer_idx) is not None]
                    # Concat all generated loradicts + one metalora
                    layer_meta = metalora.get(layer_idx, None) if metalora else None
                    parts_to_concat = layer_loradicts[:]
                    if layer_meta is not None:
                        parts_to_concat.append(detach_loradict(layer_meta))
                    if parts_to_concat:
                        merged_loradict[layer_idx] = concat_loradict(parts_to_concat)
                    else:
                        merged_loradict[layer_idx] = None

                # Convert to PEFT and save
                # Sanitize repo_name for filesystem
                safe_repo_name = repo_name.replace("/", "_").replace("\\", "_")
                save_dir = os.path.join(_export_save_base, safe_repo_name)
                loradict_to_peft(
                    merged_loradict,
                    save_path=save_dir,
                    base_model_name_or_path=model_path,
                )

                # Save eval metrics with full statistics
                import math as _math

                avg_loss = sum(repo_losses) / len(repo_losses) if repo_losses else 0.0
                avg_baseline_loss = sum(repo_baseline_losses) / len(repo_baseline_losses) if repo_baseline_losses else 0.0
                avg_ppl = _math.exp(avg_loss) if avg_loss < 20 else float('inf')
                avg_baseline_ppl = _math.exp(avg_baseline_loss) if avg_baseline_loss < 20 else float('inf')

                # Per-trajectory PPL
                per_traj_ppl = [_math.exp(l) if l < 20 else float('inf') for l in repo_losses]
                per_traj_baseline_ppl = [_math.exp(l) if l < 20 else float('inf') for l in repo_baseline_losses]

                # Per-trajectory ratios (model / baseline)
                per_traj_loss_ratio = []
                per_traj_ppl_ratio = []
                for l, bl in zip(repo_losses, repo_baseline_losses):
                    per_traj_loss_ratio.append(l / bl if bl > 0 else float('inf'))
                for p, bp in zip(per_traj_ppl, per_traj_baseline_ppl):
                    per_traj_ppl_ratio.append(p / bp if bp > 0 else float('inf'))

                # Average ratios
                avg_loss_ratio = sum(per_traj_loss_ratio) / len(per_traj_loss_ratio) if per_traj_loss_ratio else 0.0
                avg_ppl_ratio = sum(per_traj_ppl_ratio) / len(per_traj_ppl_ratio) if per_traj_ppl_ratio else 0.0

                # Win rate (fraction of trajectories where model loss < baseline loss)
                win_count = sum(1 for l, bl in zip(repo_losses, repo_baseline_losses) if l < bl)
                win_rate = win_count / len(repo_losses) if repo_losses else 0.0

                # Trend slope (linear regression slope over trajectory index)
                def _compute_slope(values):
                    """Compute slope of linear regression y = a*x + b."""
                    n = len(values)
                    if n < 2:
                        return 0.0
                    x_mean = (n - 1) / 2.0
                    y_mean = sum(values) / n
                    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
                    den = sum((i - x_mean) ** 2 for i in range(n))
                    return num / den if den > 0 else 0.0

                loss_trend_slope = _compute_slope(repo_losses)
                ppl_trend_slope = _compute_slope(per_traj_ppl)
                baseline_loss_trend_slope = _compute_slope(repo_baseline_losses)
                baseline_ppl_trend_slope = _compute_slope(per_traj_baseline_ppl)
                loss_ratio_trend_slope = _compute_slope(per_traj_loss_ratio)
                ppl_ratio_trend_slope = _compute_slope(per_traj_ppl_ratio)

                metrics = {
                    "repo": repo_name,
                    "num_trajectories": len(indices),
                    "avg_loss": avg_loss,
                    "avg_ppl": avg_ppl,
                    "avg_baseline_loss": avg_baseline_loss,
                    "avg_baseline_ppl": avg_baseline_ppl,
                    "avg_loss_ratio": avg_loss_ratio,
                    "avg_ppl_ratio": avg_ppl_ratio,
                    "win_rate": win_rate,
                    "loss_trend_slope": loss_trend_slope,
                    "ppl_trend_slope": ppl_trend_slope,
                    "baseline_loss_trend_slope": baseline_loss_trend_slope,
                    "baseline_ppl_trend_slope": baseline_ppl_trend_slope,
                    "loss_ratio_trend_slope": loss_ratio_trend_slope,
                    "ppl_ratio_trend_slope": ppl_ratio_trend_slope,
                    "per_trajectory_loss": repo_losses,
                    "per_trajectory_baseline_loss": repo_baseline_losses,
                    "per_trajectory_ppl": per_traj_ppl,
                    "per_trajectory_baseline_ppl": per_traj_baseline_ppl,
                    "per_trajectory_loss_ratio": per_traj_loss_ratio,
                    "per_trajectory_ppl_ratio": per_traj_ppl_ratio,
                }
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)

                all_repo_results[repo_name] = metrics

                if is_main_process_per_node():
                    logger.info(
                        f"    [ExportLoRA] Saved {repo_name}: "
                        f"loss={avg_loss:.4f}, ppl={avg_ppl:.2f}, "
                        f"loss_ratio={avg_loss_ratio:.4f}, win_rate={win_rate:.2f}"
                    )
            else:
                if is_main_process_per_node():
                    logger.warning(f"    [ExportLoRA] No loradicts collected for {repo_name}")

            # Clear GPU cache between repos
            if hasattr(model, 'invalidate_input_cache'):
                model.invalidate_input_cache()
            torch.cuda.empty_cache()

        # ============================================================
        # Step 4: Gather all results to rank 0 and generate summary
        # ============================================================
        # Use dist.barrier() to ensure all ranks finish processing before gather
        dist.barrier()

        # Gather all_repo_results from all DP ranks to rank 0
        import math as _math

        # all_gather_object collects a list of objects from all ranks
        gathered_results = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_results, all_repo_results)

        # Only the global rank 0 process generates summary and uploads to wandb
        if is_main_process():
            # Merge all gathered results (each rank contributed different repos)
            merged_repo_results = {}
            for rank_results in gathered_results:
                if rank_results:
                    merged_repo_results.update(rank_results)

            if merged_repo_results:
                # Compute global metrics
                all_avg_losses = [m["avg_loss"] for m in merged_repo_results.values()]
                all_avg_baseline_losses = [m["avg_baseline_loss"] for m in merged_repo_results.values()]
                all_avg_ppls = [m["avg_ppl"] for m in merged_repo_results.values()]
                all_avg_baseline_ppls = [m["avg_baseline_ppl"] for m in merged_repo_results.values()]
                all_loss_ratios = [m["avg_loss_ratio"] for m in merged_repo_results.values()]
                all_ppl_ratios = [m["avg_ppl_ratio"] for m in merged_repo_results.values()]
                all_win_rates = [m["win_rate"] for m in merged_repo_results.values()]
                all_num_traj = [m["num_trajectories"] for m in merged_repo_results.values()]

                global_metrics = {
                    "num_repos": len(merged_repo_results),
                    "total_trajectories": sum(all_num_traj),
                    "avg_loss": sum(all_avg_losses) / len(all_avg_losses),
                    "avg_baseline_loss": sum(all_avg_baseline_losses) / len(all_avg_baseline_losses),
                    "avg_ppl": sum(all_avg_ppls) / len(all_avg_ppls),
                    "avg_baseline_ppl": sum(all_avg_baseline_ppls) / len(all_avg_baseline_ppls),
                    "avg_loss_ratio": sum(all_loss_ratios) / len(all_loss_ratios),
                    "avg_ppl_ratio": sum(all_ppl_ratios) / len(all_ppl_ratios),
                    "avg_win_rate": sum(all_win_rates) / len(all_win_rates),
                    "avg_loss_trend_slope": sum(m["loss_trend_slope"] for m in merged_repo_results.values()) / len(merged_repo_results),
                    "avg_ppl_trend_slope": sum(m["ppl_trend_slope"] for m in merged_repo_results.values()) / len(merged_repo_results),
                    "avg_baseline_loss_trend_slope": sum(m["baseline_loss_trend_slope"] for m in merged_repo_results.values()) / len(merged_repo_results),
                    "avg_baseline_ppl_trend_slope": sum(m["baseline_ppl_trend_slope"] for m in merged_repo_results.values()) / len(merged_repo_results),
                    "avg_loss_ratio_trend_slope": sum(m["loss_ratio_trend_slope"] for m in merged_repo_results.values()) / len(merged_repo_results),
                    "avg_ppl_ratio_trend_slope": sum(m["ppl_ratio_trend_slope"] for m in merged_repo_results.values()) / len(merged_repo_results),
                }

                # Per-repo summary (without per_trajectory arrays for compactness)
                per_repo_summary = {}
                for repo_name, m in sorted(merged_repo_results.items()):
                    per_repo_summary[repo_name] = {
                        "num_trajectories": m["num_trajectories"],
                        "avg_loss": m["avg_loss"],
                        "avg_baseline_loss": m["avg_baseline_loss"],
                        "avg_ppl": m["avg_ppl"],
                        "avg_baseline_ppl": m["avg_baseline_ppl"],
                        "avg_loss_ratio": m["avg_loss_ratio"],
                        "avg_ppl_ratio": m["avg_ppl_ratio"],
                        "win_rate": m["win_rate"],
                        "loss_trend_slope": m["loss_trend_slope"],
                        "ppl_trend_slope": m["ppl_trend_slope"],
                        "baseline_loss_trend_slope": m["baseline_loss_trend_slope"],
                        "baseline_ppl_trend_slope": m["baseline_ppl_trend_slope"],
                        "loss_ratio_trend_slope": m["loss_ratio_trend_slope"],
                        "ppl_ratio_trend_slope": m["ppl_ratio_trend_slope"],
                    }

                summary = {
                    "experiment": exp_name,
                    "dataset": _dataset_name,
                    "max_trajectories_per_repo": _max_traj,
                    "global_metrics": global_metrics,
                    "per_repo": per_repo_summary,
                }

                # Save summary.json
                os.makedirs(_export_save_base, exist_ok=True)
                summary_path = os.path.join(_export_save_base, "summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"  [ExportLoRA] Summary saved to {summary_path}")
                logger.info(
                    f"  [ExportLoRA] Complete. {global_metrics['num_repos']} repos, "
                    f"avg_loss={global_metrics['avg_loss']:.4f}, "
                    f"avg_loss_ratio={global_metrics['avg_loss_ratio']:.4f}, "
                    f"avg_win_rate={global_metrics['avg_win_rate']:.2f}"
                )

                # WandB upload
                if wandb is not None:
                    _wandb_disabled = (
                        os.environ.get("WANDB_MODE", "online") in ("disabled", "offline")
                        or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
                    )
                    if not _wandb_disabled:
                        try:
                            # Initialize a new wandb run for this evaluation
                            # Use same project as training (from cfg or env var)
                            # Tags include "evaluation_export_lora" for easy filtering
                            _wandb_cfg = cfg.get("wandb", {})
                            _wandb_project = os.environ.get("WANDB_PROJECT") or _wandb_cfg.get("project", "SHINE_V2")
                            _wandb_entity = os.environ.get("WANDB_ENTITY") or _wandb_cfg.get("entity")
                            _wandb_job_type = exp_name[:64] if len(exp_name) > 64 else exp_name
                            wandb.init(
                                project=_wandb_project,
                                entity=_wandb_entity,
                                name=f"{exp_name}/{_dataset_name}",
                                group=_dataset_name,
                                job_type=_wandb_job_type,
                                config={
                                    "run_name": exp_name,
                                    "dataset_name": _dataset_name,
                                    "max_traj_per_repo": _max_traj,
                                    "num_repos": global_metrics["num_repos"],
                                    "total_trajectories": global_metrics["total_trajectories"],
                                    "parallel_mode": "tp",
                                },
                                tags=["evaluation_export_lora", _dataset_name, exp_name[:64]],
                                reinit=True,
                            )

                            # Log global scalar metrics as summary (not step-based)
                            # to avoid meaningless line charts with step=0
                            wandb.summary.update({
                                "export_lora/avg_loss": global_metrics["avg_loss"],
                                "export_lora/avg_baseline_loss": global_metrics["avg_baseline_loss"],
                                "export_lora/avg_ppl": global_metrics["avg_ppl"],
                                "export_lora/avg_baseline_ppl": global_metrics["avg_baseline_ppl"],
                                "export_lora/avg_loss_ratio": global_metrics["avg_loss_ratio"],
                                "export_lora/avg_ppl_ratio": global_metrics["avg_ppl_ratio"],
                                "export_lora/avg_win_rate": global_metrics["avg_win_rate"],
                                "export_lora/num_repos": global_metrics["num_repos"],
                                "export_lora/total_trajectories": global_metrics["total_trajectories"],
                                "export_lora/avg_loss_trend_slope": global_metrics["avg_loss_trend_slope"],
                                "export_lora/avg_ppl_trend_slope": global_metrics["avg_ppl_trend_slope"],
                                "export_lora/avg_baseline_loss_trend_slope": global_metrics["avg_baseline_loss_trend_slope"],
                                "export_lora/avg_baseline_ppl_trend_slope": global_metrics["avg_baseline_ppl_trend_slope"],
                                "export_lora/avg_loss_ratio_trend_slope": global_metrics["avg_loss_ratio_trend_slope"],
                                "export_lora/avg_ppl_ratio_trend_slope": global_metrics["avg_ppl_ratio_trend_slope"],
                            })

                            # Log per-repo table
                            table = wandb.Table(columns=[
                                "repo", "num_traj", "avg_loss", "avg_baseline_loss",
                                "loss_ratio", "ppl_ratio", "win_rate",
                                "loss_trend_slope", "ppl_trend_slope",
                                "loss_ratio_trend_slope", "ppl_ratio_trend_slope",
                            ])
                            for repo_name, data in sorted(per_repo_summary.items()):
                                table.add_data(
                                    repo_name, data["num_trajectories"],
                                    data["avg_loss"], data["avg_baseline_loss"],
                                    data["avg_loss_ratio"], data["avg_ppl_ratio"],
                                    data["win_rate"],
                                    data["loss_trend_slope"], data["ppl_trend_slope"],
                                    data["loss_ratio_trend_slope"], data["ppl_ratio_trend_slope"],
                                )
                            wandb.log({"export_lora/per_repo_summary": table})

                            # Log per-trajectory detail table
                            traj_table = wandb.Table(columns=[
                                "repo", "traj_idx", "loss", "baseline_loss",
                                "ppl", "baseline_ppl", "loss_ratio", "ppl_ratio",
                            ])
                            for repo_name, m in sorted(merged_repo_results.items()):
                                for i in range(m["num_trajectories"]):
                                    traj_table.add_data(
                                        repo_name, i,
                                        m["per_trajectory_loss"][i],
                                        m["per_trajectory_baseline_loss"][i],
                                        m["per_trajectory_ppl"][i],
                                        m["per_trajectory_baseline_ppl"][i],
                                        m["per_trajectory_loss_ratio"][i],
                                        m["per_trajectory_ppl_ratio"][i],
                                    )
                            wandb.log({"export_lora/trajectory_details": traj_table})

                            # Save summary.json as artifact
                            artifact = wandb.Artifact(
                                f"export_lora_{_dataset_name}",
                                type="evaluation",
                            )
                            artifact.add_file(summary_path)
                            wandb.log_artifact(artifact)

                            wandb.finish()
                            logger.info("  [ExportLoRA] WandB upload complete.")
                        except Exception as e:
                            logger.warning(f"  [ExportLoRA] WandB upload failed: {e!r}")

        # Final barrier to prevent any rank from exiting early
        dist.barrier()
        _profiler_ctx.__exit__(None, None, None)
        cleanup_distributed()
        os._exit(0)

    # --- Debug Resume: open JSONL dump file if DEBUG_RESUME env var is set ---
    _debug_resume_file = None
    if os.environ.get("DEBUG_RESUME", "") == "1" and is_main_process():
        _debug_dir = os.path.join(cfg.training.log_dir, run_name)
        os.makedirs(_debug_dir, exist_ok=True)
        _debug_path = os.path.join(_debug_dir, "node_0_debug_steps.jsonl")
        _debug_resume_file = open(_debug_path, "a")  # append mode for resume
        logger.info(f"  [DEBUG_RESUME] Dumping per-step state to {_debug_path}")

    # --- save_best_only: keep ONLY the single best-by-val-ppl model (model-only,
    # overwritten in place at checkpoint/<run>/final/model). Disables interval +
    # end-of-training saves to avoid storage blowup. ---
    save_best_only = bool(cfg.training.get("save_best_only", False))
    best_val_ppl = float("inf")
    if save_best_only and is_main_process_per_node():
        logger.info("[save_best_only] keeping only the best-by-val-ppl model at final/model "
                    "(interval/end checkpoints disabled).")

    def _save_best_model(step: int, val_ppl: float):
        """Overwrite final/model with the current model iff val_ppl improved. Collective barrier."""
        nonlocal best_val_ppl
        improved = torch.zeros(1, device=my_device)
        if is_main_process() and val_ppl < best_val_ppl:
            improved[0] = 1.0
        if dist.is_initialized():
            dist.broadcast(improved, src=0)
        if improved.item() > 0:
            if is_main_process():
                best_val_ppl = val_ppl
                final_dir = os.path.join(get_checkpoint_dir(run_name), "final")
                model.save_model(os.path.join(final_dir, "model"))
                ts_dir = os.path.join(final_dir, "training_state")
                os.makedirs(ts_dir, exist_ok=True)
                torch.save({"global_step": step, "epoch": epoch, "is_final": True,
                            "best_val_ppl": float(val_ppl)},
                           os.path.join(ts_dir, "metadata.pt"))
                logger.info(f"[save_best_only] new best val_ppl={val_ppl:.4f} → overwrote "
                            f"{os.path.join(final_dir, 'model')} (step {step})")
            barrier()
    for epoch in range(start_epoch, num_epochs):
        train_loader.set_epoch(epoch) if hasattr(train_loader, "set_epoch") else None
        data_iter = iter(train_loader)

        # Reset epoch-level accumulators (same as PP)
        if epoch != start_epoch or (resume_checkpoint_dir is None or load_model_only_flag):
            epoch_loss_sum = 0.0
            epoch_steps = 0

        while global_step < max_steps:
            try:
                batch_meta = next(data_iter)
            except StopIteration:
                break

            # Notify collator of training progress (for dynamic mask scheduling)
            if hasattr(collator, 'set_training_progress'):
                collator.set_training_progress(global_step, max_steps)

            # PipelineDataLoader returns a dict with "micro_batches"; each
            # micro-batch is a dict of tensors.
            mbs = batch_meta.get("micro_batches", [])
            _accum_distill_loss = 0.0
            _accum_regu_sq_norm = 0.0
            for mb in mbs:
                if global_step >= max_steps:
                    break
                # Move to device
                mb_dev = {
                    "context_ids": mb["context_ids"].to(my_device, non_blocking=True),
                    "conversation_ids": mb["conversation_ids"].to(my_device, non_blocking=True),
                    "labels": mb["labels"].to(my_device, non_blocking=True),
                    "context_lengths": mb["context_lengths"].to(my_device, non_blocking=True),
                }

                # Extract extra_info (non-tensor metadata, e.g. repo name)
                _extra_info = mb.get("extra_info", None)
                _cur_repo = None
                _repo_reset_triggered = False

                # --- Repo-change reset: if repo changed, reset detach_state before forward ---
                if _extra_info is not None and model.detach_state is not None:
                    _cur_repo = _extra_info[0].get("repo") if isinstance(_extra_info, list) and len(_extra_info) > 0 else None
                    if _cur_repo is not None and _prev_repo is not None and _cur_repo != _prev_repo:
                        model.detach_state.reset()
                        model.detach_state.init_steps()
                        running_repo_reset_count += 1
                        _repo_reset_triggered = True
                    if _cur_repo is not None:
                        _prev_repo = _cur_repo

                # Token statistics: accumulate per-window (same as PP)
                step_context_tokens += int(mb_dev["context_lengths"].sum().item())
                step_conv_total_tokens += int((mb_dev["conversation_ids"] != pad_token_id).sum().item())
                step_conv_valid_tokens += int((mb_dev["labels"] != -100).sum().item())

                # Extract distillation batch (if provided by collator)
                _distill_batch = None
                if distill_loss_fn is not None and mb.get("distill") is not None:
                    _distill_batch = {
                        k: v.to(my_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in mb["distill"].items()
                    }

                # Determine whether this micro-batch needs per-token loss
                # (only when detail logging is scheduled for this step)
                _need_detail = (
                    (log_train_detail_sched is not None and log_train_detail_sched.should_run(global_step + 1))
                    or log_train_detail_ppl_threshold > 0
                )
                # Compute train answer-token accuracy only on steps that will be
                # logged (logging_steps cadence) -> negligible extra cost.
                _train_acc_this = _train_acc_on and logging_steps > 0 and ((global_step + 1) % logging_steps == 0)

                accum_loss, stepped, _skipped, _grad_norm_metrics, _per_token_loss, _distill_loss_item, _regu_sq_norm_local = _train_step(
                    model=model,
                    batch=mb_dev,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    tp_cfg=tp_cfg,
                    grad_accum_steps=grad_accum_steps,
                    gradient_clipping=gradient_clipping,
                    micro_step=micro_step,
                    accum_loss=accum_loss,
                    sdpa_ctx_factory=make_sdpa_ctx,
                    nan_inf_tracker=nan_inf_tracker,
                    monitor_nan_inf=monitor_nan_inf,
                    grad_norm_sched=grad_norm_sched,
                    global_step=global_step,
                    distill_batch=_distill_batch,
                    distill_loss_fn=distill_loss_fn,
                    return_per_token_loss=_need_detail,
                )

                # Write loradict into detach_state (must be after backward)
                model.post_backward_detach_state(grad_accum_steps=grad_accum_steps)

                _accum_regu_sq_norm += _regu_sq_norm_local
                if _distill_loss_item is not None:
                    _accum_distill_loss += _distill_loss_item
                if _train_acc_this:
                    _train_acc_corr += int(getattr(model, "_last_eval_acc_correct", 0))
                    _train_acc_tot += int(getattr(model, "_last_eval_acc_total", 0))
                    _train_ans_corr += int(getattr(model, "_last_eval_ans_correct", 0))
                    _train_ans_tot += int(getattr(model, "_last_eval_ans_total", 0))
                micro_step += 1

                # Accumulate detail info for later logging (same as PP)
                if (log_train_detail_sched is not None
                        and log_train_detail_sched.should_run(global_step + 1)
                        and detail_log_accumulator is not None):
                    detail_log_accumulator.append({
                        "context_ids": mb_dev["context_ids"].cpu(),
                        "conversation_ids": mb_dev["conversation_ids"].cpu(),
                        "labels": mb_dev["labels"].cpu(),
                        "context_lengths": mb_dev["context_lengths"].cpu(),
                        "per_token_loss": _per_token_loss,  # (B, S-1) float32 or None
                        "distill": {
                            "conversation_ids": _distill_batch["conversation_ids"].cpu(),
                            "labels": _distill_batch["labels"].cpu(),
                        } if _distill_batch is not None else None,
                    })

                # Accumulate detail info for ppl-threshold-triggered logging
                if log_train_detail_ppl_threshold > 0:
                    detail_log_accumulator_ppl.append({
                        "context_ids": mb_dev["context_ids"].cpu(),
                        "conversation_ids": mb_dev["conversation_ids"].cpu(),
                        "labels": mb_dev["labels"].cpu(),
                        "context_lengths": mb_dev["context_lengths"].cpu(),
                        "per_token_loss": _per_token_loss,  # (B, S-1) float32 or None
                        "distill": {
                            "conversation_ids": _distill_batch["conversation_ids"].cpu(),
                            "labels": _distill_batch["labels"].cpu(),
                        } if _distill_batch is not None else None,
                    })

                if stepped:
                    avg_loss = accum_loss / grad_accum_steps

                    if not _skipped:
                        running_loss += avg_loss
                        if _accum_distill_loss > 0:
                            running_distill_loss += _accum_distill_loss / grad_accum_steps
                        # epoch_loss_sum accumulates CE loss only (same as PP)
                        avg_distill_this_step = _accum_distill_loss / grad_accum_steps if _accum_distill_loss > 0 else 0.0
                        epoch_loss_sum += (avg_loss - avg_distill_this_step)
                        global_step += 1
                        epoch_steps += 1
                    else:
                        # NaN/Inf skip: still increment step so training progresses
                        global_step += 1
                        epoch_steps += 1
                        # Check if consecutive bad steps exceed threshold → terminate training
                        if nan_inf_stop_steps > 0 and nan_inf_tracker is not None:
                            if nan_inf_tracker.consecutive_bad >= nan_inf_stop_steps:
                                logger.error(
                                    f"  [FATAL] NaN/Inf consecutive bad steps ({nan_inf_tracker.consecutive_bad}) "
                                    f"reached threshold ({nan_inf_stop_steps}). Terminating training."
                                )
                                if use_wandb and is_main_process():
                                    wandb.log({"monitor/nan_inf_terminated": 1.0}, step=global_step)
                                    wandb.finish(exit_code=1)
                                dist.barrier()
                                raise RuntimeError(
                                    f"Training terminated: {nan_inf_tracker.consecutive_bad} consecutive "
                                    f"NaN/Inf steps exceeded threshold {nan_inf_stop_steps}."
                                )

                    accum_loss = 0.0

                    # --- DetachState: sync sq_norm across TP group and update threshold ---
                    _regu_sq_norm_full = 0.0
                    _reset_ratio = 0.0
                    _mean_update_step = 0.0
                    if model.detach_state is not None and _accum_regu_sq_norm > 0:
                        # Each TP rank has partial sq_norm (its shard only).
                        # All_reduce SUM across TP group to get full ||W||².
                        _sq_tensor = torch.tensor([_accum_regu_sq_norm], dtype=torch.float64, device=my_device)
                        tp_group = tp_cfg.get("tp_process_group")
                        if tp_group is not None and tp_cfg["tensor_parallel_size"] > 1:
                            dist.all_reduce(_sq_tensor, op=dist.ReduceOp.SUM, group=tp_group)
                        _regu_sq_norm_full = _sq_tensor.item()
                        # Average across micro-batches to get per-mb sq_norm
                        # (consistent with PP mode which stores per-mb values)
                        _regu_sq_norm_full /= grad_accum_steps
                        # Store sq_norms for threshold-based reset (expand to per-sample)
                        _regu_sq_norms_per_sample = [_regu_sq_norm_full] * local_batch_size
                        model.detach_state.set_last_sq_norms(_regu_sq_norms_per_sample)
                        # Perform reset check at end of step (so next step starts clean)
                        if _skipped:
                            # Skip step: force reset wdict to zero so next step starts fresh
                            model.detach_state.reset()
                            model.detach_state.init_steps()
                        else:
                            # Step+1 first, then record stats, then threshold reset
                            for _si in range(local_batch_size):
                                model.detach_state.update_steps(_si)
                        _reset_ratio, _mean_update_step = model.detach_state.get_reset_stats()
                        if not _skipped:
                            for _si in range(local_batch_size):
                                model.detach_state.maybe_reset_slice(_si)
                    elif model.detach_state is not None:
                        if _skipped:
                            # Skip step without regu: still force reset wdict
                            model.detach_state.reset()
                            model.detach_state.init_steps()
                        else:
                            # No regu, but still maintain _update_steps counter
                            for _si in range(local_batch_size):
                                model.detach_state.update_steps(_si)
                        _reset_ratio, _mean_update_step = model.detach_state.get_reset_stats()
                        if not _skipped:
                            for _si in range(local_batch_size):
                                model.detach_state.maybe_reset_slice(_si)
                    running_regu_sq_norm += _regu_sq_norm_full
                    running_reset_ratio += _reset_ratio
                    running_mean_update_step += _mean_update_step
                    _accum_regu_sq_norm = 0.0

                    # --- Debug Resume Dump: write per-step state to JSONL for verification ---
                    if _debug_resume_file is not None and is_main_process():
                        import hashlib
                        _ctx_hash = hashlib.md5(mb_dev["context_ids"].cpu().numpy().tobytes()).hexdigest()
                        _conv_hash = hashlib.md5(mb_dev["conversation_ids"].cpu().numpy().tobytes()).hexdigest()
                        _labels_hash = hashlib.md5(mb_dev["labels"].cpu().numpy().tobytes()).hexdigest()
                        _lr_now = lr_scheduler.get_last_lr()[0] if lr_scheduler else 0.0
                        _ds_update_steps = list(model.detach_state._update_steps) if model.detach_state is not None else []
                        _debug_record = {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": float(avg_loss),
                            "lr": float(_lr_now),
                            "prev_repo": _prev_repo,
                            "cur_repo": _cur_repo,
                            "repo_reset_triggered": _repo_reset_triggered,
                            "update_steps": _ds_update_steps,
                            "reset_ratio": float(_reset_ratio),
                            "mean_update_step": float(_mean_update_step),
                            "regu_sq_norm": float(_regu_sq_norm_full),
                            "batch_counter": train_loader._batch_counter,
                            "context_ids_hash": _ctx_hash,
                            "conv_ids_hash": _conv_hash,
                            "labels_hash": _labels_hash,
                            "context_lengths": mb_dev["context_lengths"].cpu().tolist(),
                            "skipped": _skipped,
                            "grad_norm": _grad_norm_metrics.get("grad_norm/total", None) if _grad_norm_metrics else None,
                            "running_loss": float(running_loss),
                            "epoch_loss_sum": float(epoch_loss_sum),
                            "epoch_steps": epoch_steps,
                        }
                        _debug_resume_file.write(json.dumps(_debug_record) + "\n")
                        _debug_resume_file.flush()

                    # --- P0 Monitor: Param norm and generated LoRA norm ---
                    # Computed after optimizer step so we see updated weights.
                    # In TP mode all params are replicated, so main GPU can
                    # compute directly without gather (unlike PP).
                    _param_norm_metrics = {}
                    _gen_lora_norm_metrics = {}
                    if is_main_process() and not _skipped:
                        _should_log_param_norm = (
                            param_norm_sched is not None
                            and param_norm_sched.should_run(global_step)
                        )
                        _should_log_gen_lora_norm = (
                            gen_lora_norm_sched is not None
                            and gen_lora_norm_sched.should_run(global_step)
                        )
                        if _should_log_param_norm:
                            _param_norm_metrics = compute_param_norms(model, my_device)
                        if _should_log_gen_lora_norm:
                            _gen_lora_norm_metrics = compute_generated_lora_norms(model, my_device)

                    # --- Checkpointing (before timing so save overhead is included in step_time, same as PP) ---
                    save_duration = 0.0
                    if save_sched.should_run(global_step) and not save_best_only:
                        save_t0 = time.time()
                        # Rank 0 saves model, scheduler, metadata
                        if is_main_process():
                            _tp_save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                global_step=global_step,
                                epoch=epoch,
                                micro_step=micro_step,
                                run_name=run_name,
                                forever_save_steps=forever_save_steps,
                                save_total_limit=save_total_limit,
                                running_loss=running_loss,
                                ema_time_per_step=ema_time_per_step,
                                wandb_run_id=wandb_run_id,
                                t_start=t_start,
                                max_steps=max_steps,
                                train_loader=train_loader,
                                total_context_tokens=total_context_tokens,
                                total_conv_total_tokens=total_conv_total_tokens,
                                total_conv_valid_tokens=total_conv_valid_tokens,
                                config_selections=_resolved_configs,
                                launch_cmd=os.environ.get("LAUNCH_CMD", ""),
                                prev_repo=_prev_repo,
                                tp_size=tp_cfg["tensor_parallel_size"],
                                sp_size=tp_cfg["sequence_parallel_size"],
                                gpus_per_node=tp_cfg["total_gpus"],
                                num_nodes=dist.get_world_size() // tp_cfg["total_gpus"],
                            )
                        barrier()  # Ensure rank 0 finishes saving model/metadata before optimizer shards

                        # All ranks on node 0 save their optimizer shard
                        # (all nodes have identical shards, so only node 0 needs to save)
                        _node_rank = dist.get_rank() // tp_cfg["total_gpus"]
                        if _node_rank == 0:
                            step_dir = get_step_checkpoint_dir(run_name, global_step)
                            training_state_dir = os.path.join(step_dir, "training_state")
                            os.makedirs(training_state_dir, exist_ok=True)
                            _local_rank = dist.get_rank() % tp_cfg["total_gpus"]
                            _tp_save_optimizer_state_sharded(
                                model, optimizer, training_state_dir, _local_rank
                            )

                        save_duration = time.time() - save_t0
                        barrier()  # Ensure all ranks finish saving optimizer shards

                        if is_main_process():
                            logger.info(
                                f"[checkpoint] saved step {global_step} "
                                f"({save_duration:.1f}s) → {get_step_checkpoint_dir(run_name, global_step)}"
                            )

                        # Save detach_state on ALL ranks (each TP rank has different wdict shard)
                        # NOTE: Within an SP group, all ranks have the same wdict
                        # (because memory_states is all_reduced across SP). So only
                        # sp_rank=0 needs to save to avoid file conflicts.
                        if model.detach_state is not None:
                            _sp_rank_for_save = tp_cfg.get("sp_rank", 0)
                            if _sp_rank_for_save == 0:
                                ds_state = model.detach_state.state_dict()
                                if ds_state:
                                    step_dir = get_step_checkpoint_dir(run_name, global_step)
                                    ds_dir = os.path.join(step_dir, "detach_state")
                                    os.makedirs(ds_dir, exist_ok=True)
                                    ds_path = os.path.join(
                                        ds_dir,
                                        f"dp{tp_cfg['data_parallel_rank']}_tp{tp_cfg['tp_rank']}.pt"
                                    )
                                    torch.save(ds_state, ds_path)
                        barrier()  # Ensure all ranks finish saving detach_state

                    # --- EMA time-per-step tracking (after save so step_time includes save overhead, same as PP) ---
                    now = time.time()
                    last_step_duration = 0.0
                    if step_start_time > 0:
                        last_step_duration = now - step_start_time
                        if ema_time_per_step <= 0:
                            ema_time_per_step = last_step_duration  # first step: initialize
                        else:
                            ema_time_per_step = ema_alpha * last_step_duration + (1 - ema_alpha) * ema_time_per_step
                    step_start_time = now

                    # --- Logging (after save so step_time and ETA include save overhead, same as PP) ---
                    _should_log = (global_step % logging_steps == 0)
                    if _should_log:
                        # DP loss aggregation: average loss across all DP replicas
                        # PP uses running_loss / (logging_steps * grad_accum_steps) but TP
                        # accumulates avg_loss (already divided by grad_accum_steps) into running_loss,
                        # so we divide by logging_steps only.
                        _local_avg_loss = running_loss / logging_steps if logging_steps > 0 else avg_loss
                        _local_avg_distill_loss = running_distill_loss / logging_steps if logging_steps > 0 else 0.0
                        # Epoch-level averages
                        epoch_avg_loss = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0

                        dp_group = tp_cfg.get("dp_process_group")
                        dp_size = tp_cfg["data_parallel_size"]
                        if dp_group is not None and dp_size > 1:
                            loss_tensor = torch.tensor(
                                [_local_avg_loss, _local_avg_distill_loss, epoch_avg_loss],
                                device=my_device,
                            )
                            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG, group=dp_group)
                            _global_avg_loss = loss_tensor[0].item()
                            _global_avg_distill_loss = loss_tensor[1].item()
                            _global_epoch_avg_loss = loss_tensor[2].item()
                        else:
                            _global_avg_loss = _local_avg_loss
                            _global_avg_distill_loss = _local_avg_distill_loss
                            _global_epoch_avg_loss = epoch_avg_loss

                        # running_loss accumulates total_loss (CE + distill).
                        # To align with PP: loss/ppl = CE only, total_loss = CE + distill.
                        _global_avg_ce_loss = _global_avg_loss - _global_avg_distill_loss
                        _global_ce_ppl = math.exp(min(_global_avg_ce_loss, 20.0))
                        _global_epoch_avg_ppl = math.exp(min(_global_epoch_avg_loss, 20.0))
                        _global_total_loss = _global_avg_loss
                        _global_total_ppl = math.exp(min(_global_total_loss, 20.0))

                        # Update loss running average with global loss (same as PP)
                        # loss_running_avg tracks CE loss for spike detection (same as PP)
                        if loss_running_avg is None or loss_running_avg <= 0:
                            loss_running_avg = _global_avg_ce_loss
                        else:
                            loss_running_avg = _loss_avg_alpha * _global_avg_ce_loss + (1 - _loss_avg_alpha) * loss_running_avg

                        # Accumulate token counts from all DP replicas (same as PP)
                        if dp_group is not None and dp_size > 1:
                            token_tensor = torch.tensor(
                                [step_context_tokens, step_conv_total_tokens, step_conv_valid_tokens],
                                dtype=torch.long, device=my_device,
                            )
                            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM, group=dp_group)
                            total_context_tokens += token_tensor[0].item()
                            total_conv_total_tokens += token_tensor[1].item()
                            total_conv_valid_tokens += token_tensor[2].item()
                        else:
                            total_context_tokens += step_context_tokens
                            total_conv_total_tokens += step_conv_total_tokens
                            total_conv_valid_tokens += step_conv_valid_tokens

                        # Train answer-token accuracy: SUM counts over DP only.
                        _tr_ac, _tr_at = _train_acc_corr, _train_acc_tot
                        _tr_nc, _tr_nt = _train_ans_corr, _train_ans_tot
                        if dp_group is not None and dp_size > 1:
                            _tr_t = torch.tensor([float(_tr_ac), float(_tr_at), float(_tr_nc), float(_tr_nt)],
                                                 device=my_device)
                            dist.all_reduce(_tr_t, op=dist.ReduceOp.SUM, group=dp_group)
                            _tr_ac, _tr_at, _tr_nc, _tr_nt = (_tr_t[0].item(), _tr_t[1].item(),
                                                              _tr_t[2].item(), _tr_t[3].item())
                        _train_token_acc = (_tr_ac / _tr_at) if _tr_at > 0 else None
                        _train_answer_acc = (_tr_nc / _tr_nt) if _tr_nt > 0 else None
                        # All-reduce detach_state metrics across DP replicas
                        if model.detach_state is not None:
                            _local_regu = running_regu_sq_norm / logging_steps if logging_steps > 0 else 0.0
                            _local_reset = running_reset_ratio / logging_steps if logging_steps > 0 else 0.0
                            _local_update = running_mean_update_step / logging_steps if logging_steps > 0 else 0.0
                            _local_repo_reset = running_repo_reset_count / logging_steps if logging_steps > 0 else 0.0
                            dp_group = tp_cfg.get("dp_process_group")
                            dp_size = tp_cfg["data_parallel_size"]
                            if dp_group is not None and dp_size > 1:
                                _ds_tensor = torch.tensor(
                                    [_local_regu, _local_reset, _local_update, _local_repo_reset],
                                    device=my_device,
                                )
                                dist.all_reduce(_ds_tensor, op=dist.ReduceOp.AVG, group=dp_group)
                                _global_avg_regu = _ds_tensor[0].item()
                                _global_avg_reset = _ds_tensor[1].item()
                                _global_avg_update = _ds_tensor[2].item()
                                _global_avg_repo_reset = _ds_tensor[3].item()
                            else:
                                _global_avg_regu = _local_regu
                                _global_avg_reset = _local_reset
                                _global_avg_update = _local_update
                                _global_avg_repo_reset = _local_repo_reset
                        else:
                            _global_avg_regu = 0.0
                            _global_avg_reset = 0.0
                            _global_avg_update = 0.0
                            _global_avg_repo_reset = 0.0

                        if is_main_process():
                            elapsed = now - t_start
                            lr_now = lr_scheduler.get_last_lr()[0] if lr_scheduler else 0
                            steps_remaining = max_steps - global_step
                            eta_seconds = ema_time_per_step * steps_remaining if ema_time_per_step > 0 else 0.0
                            save_suffix = f", save_time={format_duration(save_duration)}" if save_duration > 0 else ""
                            _distill_suffix = ""
                            if _global_avg_distill_loss > 0:
                                _distill_suffix = (
                                    f", distill_loss={_global_avg_distill_loss:.4f}"
                                    f", total_loss={_global_total_loss:.4f}, total_ppl={_global_total_ppl:.2f}"
                                )
                            _tr_acc_suffix = ""
                            if _train_token_acc is not None:
                                _tr_acc_suffix = f",\ttoken_acc={_train_token_acc:.4f},\tanswer_acc={_train_answer_acc:.4f}"
                            _regu_suffix = ""
                            if model.detach_state is not None:
                                _regu_suffix = (
                                    f",\tregu_sq_norm={_global_avg_regu:.4f}"
                                    f",\treset_ratio={_global_avg_reset:.4f}"
                                    f",\trepo_reset_ratio={_global_avg_repo_reset:.4f}"
                                    f",\tmean_upd_step={_global_avg_update:.1f}"
                                )
                            logger.info(
                                f"  [Step {global_step}/{max_steps}]\t"
                                f"epoch={epoch},\tloss={_global_avg_ce_loss:.4f},\tppl={_global_ce_ppl:.2f}{_tr_acc_suffix}{_distill_suffix},\t"
                                f"epoch_avg_loss={_global_epoch_avg_loss:.4f},\tepoch_avg_ppl={_global_epoch_avg_ppl:.2f}{_regu_suffix},\t"
                                f"lr={lr_now:.2e},\t"
                                f"step_time={format_duration(last_step_duration)}{save_suffix},\t"
                                f"elapsed={format_duration(elapsed)},\teta={format_duration(eta_seconds)}"
                            )
                            if use_wandb:
                                wandb_metrics = {
                                    "wall_time": elapsed,
                                    "train/loss": _global_avg_ce_loss,
                                    "train/ppl": _global_ce_ppl,
                                    "train/distill_loss": _global_avg_distill_loss,
                                    "train/total_loss": _global_total_loss,
                                    "train/total_ppl": _global_total_ppl,
                                    "train/epoch_avg_loss": _global_epoch_avg_loss,
                                    "train/epoch_avg_ppl": _global_epoch_avg_ppl,
                                    "train/lr": lr_now,
                                    "train/total_context_tokens": total_context_tokens,
                                    "train/total_conv_total_tokens": total_conv_total_tokens,
                                    "train/total_conv_valid_tokens": total_conv_valid_tokens,
                                    "train/step_time": last_step_duration,
                                }
                                # DetachState metrics (regu_sq_norm, reset_ratio, mean_update_step)
                                if model.detach_state is not None:
                                    wandb_metrics["train/regu_sq_norm"] = _global_avg_regu
                                    wandb_metrics["train/reset_ratio"] = _global_avg_reset
                                    wandb_metrics["train/mean_update_step"] = _global_avg_update
                                    wandb_metrics["train/repo_reset_ratio"] = _global_avg_repo_reset
                                # Grad norm metrics
                                if _grad_norm_metrics:
                                    wandb_metrics.update(_grad_norm_metrics)
                                # Param norm metrics
                                if _param_norm_metrics:
                                    wandb_metrics.update(_param_norm_metrics)
                                # Generated LoRA norm metrics
                                if _gen_lora_norm_metrics:
                                    wandb_metrics.update(_gen_lora_norm_metrics)
                                # Update ratio (grad_norm * lr / param_norm)
                                if _grad_norm_metrics and _param_norm_metrics and lr_now > 0:
                                    for gn_key, gn_val in _grad_norm_metrics.items():
                                        if not gn_key.startswith("grad_norm/"):
                                            continue
                                        pn_key = "param_norm/" + gn_key[len("grad_norm/"):]
                                        pn_val = _param_norm_metrics.get(pn_key)
                                        if pn_val is not None and pn_val > 1e-12:
                                            ur_key = "update_ratio/" + gn_key[len("grad_norm/"):]
                                            wandb_metrics[ur_key] = gn_val * lr_now / pn_val
                                # Loss spike metrics
                                if monitor_loss_spike and loss_running_avg is not None:
                                    _spike_metrics = compute_loss_spike_metrics(
                                        [{"avg_loss": _global_avg_ce_loss}], loss_running_avg,
                                    )
                                    wandb_metrics.update(_spike_metrics)
                                # NaN/Inf metrics
                                if monitor_nan_inf and nan_inf_tracker is not None:
                                    wandb_metrics.update(nan_inf_tracker.get_wandb_metrics())
                                wandb.log(wandb_metrics, step=global_step)

                        # Reset running loss and token counters after logging (same as PP)
                        running_loss = 0.0
                        running_distill_loss = 0.0
                        running_regu_sq_norm = 0.0
                        running_reset_ratio = 0.0
                        running_mean_update_step = 0.0
                        running_repo_reset_count = 0
                        step_context_tokens = 0
                        step_conv_total_tokens = 0
                        step_conv_valid_tokens = 0

                    # --- [DEBUG] Cross-node parameter consistency check ---
                    if check_consistency_sched is not None and check_consistency_sched.should_run(global_step):
                        # Build a minimal parallel_cfg dict compatible with check_dp_param_consistency
                        _compat_cfg = {
                            "dp_process_group": tp_cfg.get("dp_process_group"),
                            "data_parallel_rank": tp_cfg["data_parallel_rank"],
                            "stage": 0,
                            "total_stages": 1,
                            "node_process_group": None,
                        }
                        check_dp_param_consistency(
                            model=model,
                            my_device=my_device,
                            parallel_cfg=_compat_cfg,
                            global_step=global_step,
                        )
                        # Also check TP-group consistency for replicated params
                        check_tp_param_consistency(
                            model=model,
                            my_device=my_device,
                            tp_cfg=tp_cfg,
                            global_step=global_step,
                        )
                        # Also check SP-group consistency for replicated params
                        check_sp_param_consistency(
                            model=model,
                            my_device=my_device,
                            tp_cfg=tp_cfg,
                            global_step=global_step,
                        )

                    # --- Detailed training log ---
                    if (log_train_detail_sched is not None
                            and log_train_detail_sched.should_run(global_step)
                            and is_main_process_per_node()
                            and detail_log_accumulator):
                        if _detail_tokenizer is None:
                            from utils.mytokenizer import create_tokenizer
                            _detail_tokenizer = create_tokenizer(
                                model_path, tokenizer_cfg=cfg.tokenizer
                            )
                        log_training_detail(
                            detail_log_accumulator=detail_log_accumulator,
                            tokenizer=_detail_tokenizer,
                            global_step=global_step,
                            epoch=epoch,
                            num_mem_token=num_mem_token,
                            pad_token_id=pad_token_id,
                        )
                        detail_log_accumulator.clear()

                    # --- Detailed training log triggered by ppl threshold ---
                    if (log_train_detail_ppl_threshold > 0
                            and is_main_process_per_node()
                            and detail_log_accumulator_ppl
                            and _global_ce_ppl > log_train_detail_ppl_threshold):
                        if _detail_tokenizer is None:
                            from utils.mytokenizer import create_tokenizer
                            _detail_tokenizer = create_tokenizer(
                                model_path, tokenizer_cfg=cfg.tokenizer
                            )
                        log_training_detail(
                            detail_log_accumulator=detail_log_accumulator_ppl,
                            tokenizer=_detail_tokenizer,
                            global_step=global_step,
                            epoch=epoch,
                            num_mem_token=num_mem_token,
                            pad_token_id=pad_token_id,
                            logger_name="debug.training_detail_ppl_threshold",
                        )
                    detail_log_accumulator_ppl.clear()

                    # --- Peak memory reporting ---
                    if log_peak_memory_sched is not None and log_peak_memory_sched.should_run(global_step):
                        _report_peak_memory(f"Step {global_step}")

                    # Evaluation (if configured)
                    if eval_sched.should_run(global_step) and val_loader_for_eval is not None:
                        # Disable masking during validation (same as PP)
                        if hasattr(collator, 'set_eval_mode'):
                            collator.set_eval_mode(True)
                        _eval_out = _tp_run_evaluation(
                            model=model,
                            val_loader=val_loader_for_eval,
                            tp_cfg=tp_cfg,
                            sdpa_ctx_factory=make_sdpa_ctx,
                            global_step=global_step,
                            use_wandb=use_wandb,
                            t_start=t_start,
                            max_steps=max_steps,
                            ema_time_per_step=ema_time_per_step,
                            distill_loss_fn=distill_loss_fn,
                        )
                        # Re-enable masking after validation
                        if hasattr(collator, 'set_eval_mode'):
                            collator.set_eval_mode(False)
                        if save_best_only and _eval_out is not None:
                            _save_best_model(global_step, _eval_out[1])

                    # Generation eval: per-type free-decode accuracy (+ optional
                    # deferred-recall probe). Heavy (greedy decode), so gated on
                    # its own schedule. Runs on main rank; others wait at barrier.
                    # Main-only decode => only safe at TP=1 (TP>1 forward has
                    # collectives that would deadlock); skip under TP>1.
                    if (gen_eval_sched.should_run(global_step)
                            and tp_cfg["tensor_parallel_size"] == 1):
                        from eval_memory_gen import run_memory_qa_gen_inloop
                        for _ge_recall, _ge_prefix in ([(False, "gen")]
                                + ([(True, "gen_recall")] if _gen_eval_recall else [])):
                            _ge_hit, _ge_tot = run_memory_qa_gen_inloop(
                                model, cfg, tp_cfg, my_device,
                                recall=_ge_recall, n_hist=_gen_eval_nhist,
                            )
                            if _ge_tot is not None and is_main_process_per_node():
                                logger.info(
                                    f"  [GenEval {_ge_prefix} step {global_step}] "
                                    + ", ".join(f"{t}={_ge_hit[t]}/{_ge_tot[t]}" for t in sorted(_ge_tot))
                                    + f", overall={sum(_ge_hit.values())}/{max(1, sum(_ge_tot.values()))}"
                                )
                            if _ge_tot is not None and is_main_process() and use_wandb and wandb is not None:
                                _ge_metrics = {f"{_ge_prefix}/acc_{t}": _ge_hit[t] / max(1, _ge_tot[t]) for t in _ge_tot}
                                _ge_metrics[f"{_ge_prefix}/acc_overall"] = sum(_ge_hit.values()) / max(1, sum(_ge_tot.values()))
                                wandb.log(_ge_metrics, step=global_step)

                    # --- Profiler step (no-op when disabled) ---
                    if _profiler.step():
                        if is_main_process_per_node():
                            logger.info("[Profiler] Profiling complete. Exiting training loop.")
                        break  # exit for mb in mbs loop

        if global_step >= max_steps:
            break

        if _profiler.should_exit:
            break

        # Epoch end logging (same as PP)
        if is_main_process_per_node():
            epoch_avg = epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
            epoch_ppl = math.exp(epoch_avg) if epoch_avg < 20 else float("inf")
            logger.info(
                f"  Epoch {epoch} finished (global_step={global_step}, "
                f"epoch_avg_loss={epoch_avg:.4f}, epoch_avg_ppl={epoch_ppl:.2f})"
            )

    _profiler_ctx.__exit__(None, None, None)

    # ==================================================================
    # 9. Finish — save final checkpoint and cleanup (same as PP)
    # ==================================================================
    _report_peak_memory("After training")

    # Final evaluation (only if eval is enabled in config)
    _eval_enabled = (eval_steps_raw is not None and eval_steps_raw > 0)
    if _eval_enabled and val_loader_for_eval is not None:
        # Skip if the last training step already triggered an eval
        _already_evaled = eval_sched.should_run(global_step)
        if not _already_evaled:
            if is_main_process_per_node():
                logger.info(f"  [Final] Running final evaluation at step {global_step}...")
            if hasattr(collator, 'set_eval_mode'):
                collator.set_eval_mode(True)
            _eval_out = _tp_run_evaluation(
                model=model,
                val_loader=val_loader_for_eval,
                tp_cfg=tp_cfg,
                sdpa_ctx_factory=make_sdpa_ctx,
                global_step=global_step,
                use_wandb=use_wandb,
                t_start=t_start,
                max_steps=max_steps,
                ema_time_per_step=ema_time_per_step,
                distill_loss_fn=distill_loss_fn,
            )
            if hasattr(collator, 'set_eval_mode'):
                collator.set_eval_mode(False)
            if save_best_only and _eval_out is not None:
                _save_best_model(global_step, _eval_out[1])
        else:
            if is_main_process_per_node():
                logger.info(f"  [Final] Skipping final evaluation (already evaluated at step {global_step})")

    # Save final checkpoint (model-only, for downstream annealing/SFT).
    # In save_best_only mode the best model is ALREADY at final/model — don't
    # overwrite it with the last-step model.
    if save_best_only:
        if is_main_process_per_node():
            logger.info(f"  [Final] save_best_only: keeping best model (val_ppl={best_val_ppl:.4f}) "
                        f"at final/model; skipping last-step final save.")
    elif is_main_process_per_node():
        logger.info(f"  [Final] Saving final checkpoint for run '{run_name}'...")
    if not save_best_only and is_main_process():
        final_dir = os.path.join(get_checkpoint_dir(run_name), "final")
        model_dir = os.path.join(final_dir, "model")
        model.save_model(model_dir)
        # Save minimal metadata
        training_state_dir = os.path.join(final_dir, "training_state")
        os.makedirs(training_state_dir, exist_ok=True)
        metadata = {
            "global_step": global_step,
            "epoch": epoch,
            "is_final": True,
        }
        torch.save(metadata, os.path.join(training_state_dir, "metadata.pt"))
        logger.info(f"[checkpoint] saved final checkpoint → {final_dir}")
    barrier()
    if is_main_process_per_node():
        logger.info(f"  [Final] Final checkpoint saved.")

    flush_debug_loggers(_debug_categories)

    if use_wandb:
        wandb.finish()

    cleanup_distributed()
    if is_main_process_per_node():
        total_time = time.time() - t_start
        logger.info(f"Training complete: {global_step} steps in {format_duration(total_time)}")


# ---------------------------------------------------------------------------
# Contract stubs for the coworker's meta_train.py dispatch
# (`_load_parallel_mode_functions`). The actual TP path is invoked by
# meta_train.main() calling `tp_main(cfg)` and returning before it would
# have called `train_step` / `run_evaluation`, so these are unreachable
# in normal flow. They exist only so the import in meta_train.py's
# `_load_parallel_mode_functions` succeeds; if hit, they signal a bug
# (the short-circuit dispatch was bypassed).
# ---------------------------------------------------------------------------


def train_step(*args, **kwargs):
    raise RuntimeError(
        "meta_train_tp.train_step called — TP path uses tp_main(cfg), "
        "not the per-step contract. Check meta_train.main() short-circuit."
    )


def run_evaluation(*args, **kwargs):
    raise RuntimeError(
        "meta_train_tp.run_evaluation called — TP path uses tp_main(cfg). "
        "Check meta_train.main() short-circuit."
    )


def nccl_p2p_warmup(model, parallel_cfg, my_device):
    """No-op: TP uses collective ops, not point-to-point."""
    return


@hydra.main(version_base=None, config_path="configs", config_name="main_pretrain")
def _standalone(cfg: DictConfig):
    tp_main(cfg)


if __name__ == "__main__":
    _standalone()
