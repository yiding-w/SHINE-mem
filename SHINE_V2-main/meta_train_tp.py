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


def _broadcast_mem_tokens(model: 'TPModelHypernetwork', tp_cfg) -> None:
    """Broadcast mem_tokens from TP rank 0 and DP rank 0.

    mem_tokens lives on model.llm.model and is zero-initialised, so all
    ranks are naturally identical at construction time.  However, after
    resume from checkpoint the loaded values may differ across ranks if
    the checkpoint was saved by a single rank.  This function ensures
    bit-identical mem_tokens across both TP and DP groups.
    """
    mem = getattr(model.llm.model, "mem_tokens", None)
    if mem is None:
        return

    # TP-group broadcast
    tp_group = tp_cfg.get("tp_process_group")
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        src_global = dist.get_global_rank(tp_group, 0)
        dist.broadcast(mem.data, src=src_global, group=tp_group)

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
                f"[Optimizer] ZeRO-1 over DP={dist.get_world_size(dp_group)}: "
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

    # Save optimizer state (keyed by globally unique param names)
    _tp_save_optimizer_state(model, optimizer, training_state_dir)

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


def _tp_save_optimizer_state(model: TPModelHypernetwork, optimizer, training_state_dir: str):
    """Save optimizer state with globally unique param keys (PP-compatible).

    Uses safetensors for tensor data and .pt for non-tensor metadata.
    Keys use the same ``hypernet.<name>`` and ``metalora_layer<N>_tensor<M>``
    convention as PP's save.

    Note: If using ZeroRedundancyOptimizer, the caller must call
    ``optimizer.consolidate_state_dict(to=0)`` before this function
    (it's a collective op that all ranks must participate in).
    """
    from safetensors.torch import save_file
    from utils.myloradict import collect_loradict_tensors

    # Access the underlying module if torch.compile wrapped it
    hypernet = model.hypernetwork
    if hasattr(hypernet, "_orig_mod"):
        hypernet = hypernet._orig_mod

    # Build param_id → globally unique key mapping
    flat_params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            flat_params.append(p)

    id_to_flat_idx = {id(p): idx for idx, p in enumerate(flat_params)}
    idx_to_key = {}

    for name, param in hypernet.named_parameters():
        pid = id(param)
        if pid in id_to_flat_idx:
            idx_to_key[id_to_flat_idx[pid]] = f"hypernet.{name}"

    # Metalora tensors (same key convention as PP: metalora_layer<N>_tensor<M>)
    if hasattr(model, 'metalora') and model.metalora is not None:
        for layer_idx, layer_lora in model.metalora.items():
            tensors = collect_loradict_tensors(layer_lora)
            for t_idx, t in enumerate(tensors):
                pid = id(t)
                if pid in id_to_flat_idx:
                    idx_to_key[id_to_flat_idx[pid]] = (
                        f"metalora_layer{layer_idx}_tensor{t_idx}"
                    )

    # W-Transform parameters
    for wt_name in ['w_transform_context', 'w_transform_conversation']:
        wt_module = getattr(model, wt_name, None)
        if wt_module is not None:
            for pname, param in wt_module.named_parameters():
                pid = id(param)
                if pid in id_to_flat_idx:
                    idx_to_key[id_to_flat_idx[pid]] = (
                        f"w_transform_{wt_name}_{pname}"
                    )

    # Extract optimizer state
    opt_sd = optimizer.state_dict()
    tensors_dict = {}
    meta_dict = {}

    for str_idx, state in opt_sd["state"].items():
        idx = int(str_idx) if isinstance(str_idx, str) else str_idx
        key = idx_to_key.get(idx)
        if key is not None:
            param_meta = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    tensors_dict[f"{key}__{k}"] = v.cpu()
                else:
                    param_meta[k] = v
            if param_meta:
                meta_dict[key] = param_meta

    # Save param_groups metadata
    groups_meta = []
    for g in opt_sd["param_groups"]:
        meta = {k: v for k, v in g.items() if k != "params"}
        groups_meta.append(meta)

    # Write files (use stage0 to match PP convention)
    if tensors_dict:
        save_file(tensors_dict, os.path.join(
            training_state_dir, "optimizer_tensors_stage0.safetensors"))

    payload = {
        "non_tensor_state": meta_dict,
        "param_groups_meta": groups_meta,
    }
    torch.save(payload, os.path.join(
        training_state_dir, "optimizer_meta_stage0.pt"))


def _tp_load_checkpoint(
    model: TPModelHypernetwork,
    optimizer,
    lr_scheduler,
    checkpoint_dir: str,
    my_device: torch.device,
) -> dict:
    """Load a full checkpoint (model + optimizer + scheduler + metadata).

    Compatible with both TP-saved and PP-saved checkpoints.

    Returns training metadata dict.
    """
    model_dir = os.path.join(checkpoint_dir, "model")
    training_state_dir = os.path.join(checkpoint_dir, "training_state")

    # Load model parameters (PP-compatible)
    model.load_model(model_dir)

    # Load optimizer state
    _tp_load_optimizer_state(model, optimizer, training_state_dir, my_device)

    # Load scheduler state
    sched_path = os.path.join(training_state_dir, "scheduler.pt")
    if os.path.exists(sched_path):
        sched_state = torch.load(sched_path, map_location="cpu")
        lr_scheduler.load_state_dict(sched_state)
    else:
        logger.warning("No scheduler checkpoint found at %s", sched_path)

    # Load training metadata
    metadata_path = os.path.join(training_state_dir, "metadata.pt")
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, map_location="cpu")
    else:
        metadata = {}

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


def _tp_load_optimizer_state(
    model: TPModelHypernetwork, optimizer, training_state_dir: str, my_device: torch.device
):
    """Load optimizer state from PP-compatible safetensors files.

    Reads optimizer_tensors_stage*.safetensors and optimizer_meta_stage*.pt,
    loading only the keys that belong to this model's hypernetwork params.
    """
    import glob
    from safetensors import safe_open

    st_files = sorted(glob.glob(
        os.path.join(training_state_dir, "optimizer_tensors_stage*.safetensors")))
    meta_files = sorted(glob.glob(
        os.path.join(training_state_dir, "optimizer_meta_stage*.pt")))

    if not st_files:
        raise FileNotFoundError(
            f"[_tp_load_optimizer_state] STRICT LOAD FAILED — "
            f"No optimizer_tensors_stage*.safetensors files found in {training_state_dir}. "
            f"Cannot resume training without optimizer state."
        )

    # Access the underlying module if torch.compile wrapped it
    hypernet = model.hypernetwork
    if hasattr(hypernet, "_orig_mod"):
        hypernet = hypernet._orig_mod

    # Build param key mapping for this model
    flat_params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            flat_params.append(p)

    id_to_flat_idx = {id(p): idx for idx, p in enumerate(flat_params)}
    idx_to_key = {}
    for name, param in hypernet.named_parameters():
        pid = id(param)
        if pid in id_to_flat_idx:
            idx_to_key[id_to_flat_idx[pid]] = f"hypernet.{name}"

    # Metalora tensors (same key convention as PP)
    from utils.myloradict import collect_loradict_tensors
    if hasattr(model, 'metalora') and model.metalora is not None:
        for layer_idx, layer_lora in model.metalora.items():
            tensors = collect_loradict_tensors(layer_lora)
            for t_idx, t in enumerate(tensors):
                pid = id(t)
                if pid in id_to_flat_idx:
                    idx_to_key[id_to_flat_idx[pid]] = (
                        f"metalora_layer{layer_idx}_tensor{t_idx}"
                    )

    # W-Transform parameters
    for wt_name in ['w_transform_context', 'w_transform_conversation']:
        wt_module = getattr(model, wt_name, None)
        if wt_module is not None:
            for pname, param in wt_module.named_parameters():
                pid = id(param)
                if pid in id_to_flat_idx:
                    idx_to_key[id_to_flat_idx[pid]] = (
                        f"w_transform_{wt_name}_{pname}"
                    )

    key_to_idx = {v: k for k, v in idx_to_key.items()}
    my_param_keys = set(key_to_idx.keys())

    # Load needed tensors from all stage files
    device_str = str(my_device)
    needed_tensors = {}
    for f in st_files:
        with safe_open(f, framework="pt", device=device_str) as sf:
            for flat_key in sf.keys():
                sep_idx = flat_key.rfind("__")
                if sep_idx == -1:
                    continue
                param_key = flat_key[:sep_idx]
                if param_key in my_param_keys:
                    needed_tensors[flat_key] = sf.get_tensor(flat_key)

    # Load non-tensor metadata
    non_tensor_state = {}
    groups_meta = None
    for f in meta_files:
        payload = torch.load(f, map_location="cpu")
        for k, v in payload.get("non_tensor_state", {}).items():
            if k in my_param_keys:
                non_tensor_state[k] = v
        if groups_meta is None:
            groups_meta = payload.get("param_groups_meta")

    # Reconstruct named state
    merged_named_state = {}
    for flat_key, tensor in needed_tensors.items():
        sep_idx = flat_key.rfind("__")
        param_key = flat_key[:sep_idx]
        state_name = flat_key[sep_idx + 2:]
        if param_key not in merged_named_state:
            merged_named_state[param_key] = {}
        merged_named_state[param_key][state_name] = tensor

    for param_key, meta in non_tensor_state.items():
        if param_key not in merged_named_state:
            merged_named_state[param_key] = {}
        merged_named_state[param_key].update(meta)

    # Reconstruct optimizer state_dict
    # Note: Do NOT call optimizer.state_dict() here — for ZeroRedundancyOptimizer
    # it requires consolidate_state_dict() first (which is a collective op).
    # Instead, build param_groups directly from the optimizer's live param_groups.
    new_state = {}
    for key, state in merged_named_state.items():
        idx = key_to_idx.get(key)
        if idx is not None:
            new_state[idx] = state

    # Build param_groups from the optimizer's current param_groups attribute
    # (this is always accessible without consolidation)
    current_param_groups = []
    for group in optimizer.param_groups:
        # Copy group metadata (lr, betas, eps, weight_decay, etc.) but replace
        # 'params' with integer indices matching the flat param list
        group_copy = {k: v for k, v in group.items() if k != "params"}
        group_copy["params"] = [id_to_flat_idx[id(p)] for p in group["params"]
                                if id(p) in id_to_flat_idx]
        current_param_groups.append(group_copy)

    load_sd = {
        "state": new_state,
        "param_groups": current_param_groups,
    }

    # Restore param_groups hyperparameters from saved
    if groups_meta is not None and len(groups_meta) == len(load_sd["param_groups"]):
        for saved_meta, group in zip(groups_meta, load_sd["param_groups"]):
            for k, v in saved_meta.items():
                if k in group:
                    group[k] = v

    # --- Strict check: fail if any parameter is missing optimizer state ---
    missing_keys = my_param_keys - set(merged_named_state.keys())
    if missing_keys:
        raise RuntimeError(
            f"[_tp_load_optimizer_state] STRICT LOAD FAILED — "
            f"{len(missing_keys)}/{len(my_param_keys)} parameters have NO optimizer state "
            f"in checkpoint at {training_state_dir}.\n"
            f"  First 5 missing: {sorted(missing_keys)[:5]}\n"
            f"  Scanned files: {[os.path.basename(f) for f in st_files]}\n"
            f"  This means the checkpoint is incomplete or incompatible. "
            f"All parameters must have saved optimizer state for resume."
        )

    optimizer.load_state_dict(load_sd)

    # Fix device mismatch for fused Adam: after load_state_dict, 'step' tensors
    # may remain on CPU (they were saved as int in non_tensor_state). Fused Adam
    # requires all state tensors (including step) on the same CUDA device.
    target_device = my_device
    # For ZeroRedundancyOptimizer, access the underlying local optimizer's state
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

    # After backward: write detach_state (must be after backward to avoid
    # inplace conflicts with autograd graph)
    model.post_backward_detach_state(grad_accum_steps=grad_accum_steps)

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
        # then DP-avg across replicas.
        torch.cuda.nvtx.range_push("GradSync")
        _tp_sum_grads(model.hypernetwork, tp_cfg.get("tp_process_group"), tp_cfg["tensor_parallel_size"])
        _dp_avg_grads(model.hypernetwork, tp_cfg.get("dp_process_group"))

        # Metalora gradient sync: TP-sum + DP-avg (same as PP's _sync_metalora_gradients)
        for layer_idx, layer_lora in model.metalora.items():
            _tp_sum_grads_tensors(layer_lora, tp_cfg.get("tp_process_group"), tp_cfg["tensor_parallel_size"])
            _dp_avg_grads_tensors(layer_lora, tp_cfg.get("dp_process_group"))

        # W-Transform gradient sync: TP-sum + DP-avg
        # (L and R have partial grads on each TP rank due to sharded W)
        for wt_name in ('w_transform_context', 'w_transform_conversation'):
            wt_module = getattr(model, wt_name, None)
            if wt_module is not None:
                # Access underlying module if torch.compile wrapped it
                _wt = wt_module._orig_mod if hasattr(wt_module, '_orig_mod') else wt_module
                _tp_sum_grads(_wt, tp_cfg.get("tp_process_group"), tp_cfg["tensor_parallel_size"])
                _dp_avg_grads(_wt, tp_cfg.get("dp_process_group"))
        torch.cuda.nvtx.range_pop()  # GradSync

        # NaN/Inf: broadcast skip decision across DP replicas
        _skip_step = False
        if monitor_nan_inf:
            my_device = tp_cfg["device"]
            _skip_tensor = torch.tensor([1.0 if _local_nan_inf else 0.0],
                                         dtype=torch.float32, device=my_device)
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
    _eval_acc_correct = 0  # teacher-forced answer-token accuracy (numerator)
    _eval_acc_total = 0    # answer tokens seen (denominator)

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
                            return_acc=True,
                        )
                    total_loss += loss.detach().float().item()
                    _eval_acc_correct += int(getattr(model, "_last_eval_acc_correct", 0))
                    _eval_acc_total += int(getattr(model, "_last_eval_acc_total", 0))
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
                            return_acc=True,
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
                            return_acc=True,
                        )
                        total_loss += loss.detach().float().item()

                # Answer-token accuracy (teacher-forced) — stashed on the model
                # by compute_loss(return_acc=True).
                _eval_acc_correct += int(getattr(model, "_last_eval_acc_correct", 0))
                _eval_acc_total += int(getattr(model, "_last_eval_acc_total", 0))

                # Write detach_state for eval accumulation (no backward needed in eval)
                model.post_backward_detach_state(grad_accum_steps=1)

                # Accumulate regu_sq_norm
                _eval_running_regu_sq_norm += _eval_regu_sq if _eval_regu_sq else 0.0

                # Step+1, get_reset_stats, then threshold reset (same order as training)
                if model.detach_state is not None:
                    _ds_batch_size = getattr(model.detach_state, "_local_batch_size", 0)
                    for _si in range(_ds_batch_size):
                        model.detach_state.update_steps(_si)
                    _eval_reset_ratio, _eval_mean_upd = model.detach_state.get_reset_stats()
                    _eval_running_reset_ratio += _eval_reset_ratio
                    _eval_running_mean_update_step += _eval_mean_upd
                    if _eval_regu_sq and _eval_regu_sq > 0:
                        # All-reduce across TP for full sq_norm before threshold check
                        _sq_t = torch.tensor([_eval_regu_sq], dtype=torch.float64, device=my_device)
                        tp_group = tp_cfg.get("tp_process_group")
                        if tp_group is not None and tp_cfg["tensor_parallel_size"] > 1:
                            dist.all_reduce(_sq_t, op=dist.ReduceOp.SUM, group=tp_group)
                        _eval_sq_norms_per_sample = [_sq_t.item()] * _ds_batch_size
                        model.detach_state.set_last_sq_norms(_eval_sq_norms_per_sample)
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

    # Answer-token accuracy: SUM counts across DP replicas only (TP ranks
    # duplicate the same data + same full-vocab logits, so they must NOT be
    # summed). token_acc = correct / total over the whole eval set.
    _acc_correct, _acc_total = _eval_acc_correct, _eval_acc_total
    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        _acc_tensor = torch.tensor([float(_acc_correct), float(_acc_total)], device=my_device)
        dist.all_reduce(_acc_tensor, op=dist.ReduceOp.SUM, group=dp_group)
        _acc_correct, _acc_total = _acc_tensor[0].item(), _acc_tensor[1].item()
    eval_token_acc = (_acc_correct / _acc_total) if _acc_total > 0 else 0.0

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
            f"val_loss={avg_loss:.4f}, val_ppl={avg_ppl:.2f}, "
            f"val_token_acc={eval_token_acc:.4f} ({int(_acc_correct)}/{int(_acc_total)})"
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
            "eval/token_acc": eval_token_acc,
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
    if cfg.parallel.get("pipeline_parallel_size", 1) > 1 and is_main_process_per_node():
        logger.warning(
            f"[main] parallel.pipeline_parallel_size={cfg.parallel.pipeline_parallel_size} "
            f"is ignored on the TP path — using tensor_parallel_size={tensor_parallel_size}."
        )
    tp_cfg = setup_tensor_parallel(
        total_gpus=cfg.parallel.total_gpus,
        tensor_parallel_size=tensor_parallel_size,
    )
    my_device = tp_cfg["device"]
    if is_main_process_per_node():
        logger.info(
            f"TP={tp_cfg['tensor_parallel_size']} DP={tp_cfg['data_parallel_size']} "
            f"tp_rank={tp_cfg['tp_rank']} dp_rank={tp_cfg['data_parallel_rank']} device={my_device}"
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
        dtype=torch.bfloat16,
        activation_checkpointing=cfg.training.get("tp_knobs", {}).get("activation_checkpointing", True),
        ckpt_skip_stride=cfg.training.get("tp_knobs", {}).get("ckpt_skip_stride", 0),
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
        dp_group=tp_cfg.get("dp_process_group"),
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
    load_model_only_flag = os.environ.get("MEMORY_QA_GEN", "") == "1"

    # Read resume_from config (same key as PP: supports "latest", path, or null/None)
    _resume_from_raw = cfg.training.get("resume_from", "latest")
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
        # Re-broadcast params after loading (ensure all TP + DP replicas are in sync)
        _broadcast_trainable_from_tp_rank0(model.hypernetwork, tp_cfg)
        _broadcast_metalora_from_tp_rank0(model.metalora, tp_cfg)
        for _wt_name in ('w_transform_context', 'w_transform_conversation'):
            _wt_module = getattr(model, _wt_name, None)
            if _wt_module is not None:
                _broadcast_trainable_from_tp_rank0(_wt_module, tp_cfg)
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
    running_regu_sq_norm = 0.0
    running_reset_ratio = 0.0
    running_mean_update_step = 0.0
    running_repo_reset_count = 0  # Count of repo-change resets
    _prev_repo = None  # Track previous repo name for repo-change reset
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

                # --- Repo-change reset: if repo changed, reset detach_state before forward ---
                if _extra_info is not None and model.detach_state is not None:
                    _cur_repo = _extra_info[0].get("repo") if isinstance(_extra_info, list) and len(_extra_info) > 0 else None
                    if _cur_repo is not None and _prev_repo is not None and _cur_repo != _prev_repo:
                        model.detach_state.reset()
                        model.detach_state.init_steps()
                        running_repo_reset_count += 1
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
                _accum_regu_sq_norm += _regu_sq_norm_local
                if _distill_loss_item is not None:
                    _accum_distill_loss += _distill_loss_item
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
                        # ZeRO-1 consolidate is a collective op — all ranks must participate
                        from torch.distributed.optim import ZeroRedundancyOptimizer
                        if isinstance(optimizer, ZeroRedundancyOptimizer):
                            optimizer.consolidate_state_dict(to=0)
                        if is_main_process():
                            save_duration = _tp_save_checkpoint(
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
                            )
                            logger.info(
                                f"[checkpoint] saved step {global_step} "
                                f"({save_duration:.1f}s) → {get_step_checkpoint_dir(run_name, global_step)}"
                            )
                        barrier()  # Ensure rank 0 finishes saving before others continue

                        # Save detach_state on ALL ranks (each TP rank has different wdict shard)
                        if model.detach_state is not None:
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
                                f"epoch={epoch},\tloss={_global_avg_ce_loss:.4f},\tppl={_global_ce_ppl:.2f}{_distill_suffix},\t"
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

    # Final evaluation (always run regardless of eval schedule)
    if val_loader_for_eval is not None:
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
    elif is_main_process():
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
