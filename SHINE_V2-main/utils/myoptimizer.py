#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimizer Utilities for Pipeline Parallel Hypernetwork Training

This module provides:
- AdamW optimizer creation with linear warmup + linear decay scheduler
- Parameter auditing: inspect every parameter in ModelHypernetwork,
  report requires_grad status and whether it is covered by the optimizer.

Usage:
    from utils.myoptimizer import create_optimizer_and_scheduler, audit_parameters
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_loader, cfg)
    audit_parameters(model, optimizer)
"""

import math
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils.myparallel import get_pipeline_config, is_main_process_per_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer + Scheduler
# ---------------------------------------------------------------------------

def _get_linear_schedule_with_warmup_and_min_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a schedule with linear warmup from 0 to 1, then linear decay
    from 1 to min_lr_ratio (instead of decaying all the way to 0).

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a ratio of peak LR (0.0 = decay to 0).

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup: 0 -> 1
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay: 1 -> min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    num_training_steps: int,
    learning_rate: float = 1e-4,
    min_learning_rate: float = 0.0,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    warmup_steps: int = 1000,
) -> Tuple[AdamW, LambdaLR]:
    """
    Create an AdamW optimizer and a linear-warmup + linear-decay LR scheduler
    for all **trainable** hypernetwork parameters and metalora tensors on the
    current pipeline stage.

    The LR decays linearly from ``learning_rate`` to ``min_learning_rate``
    (default 0) after warmup.

    Only parameters that satisfy ALL of the following are included:
    1. ``requires_grad is True``
    2. Not on ``meta`` device (i.e. actually materialised on this stage)
    3. On the current pipeline stage's device

    The LLM parameters are frozen and excluded automatically.

    Args:
        model:              The ``ModelHypernetwork`` instance.
        num_training_steps: Total number of optimizer steps (after grad accum).
        learning_rate:      Peak learning rate.
        min_learning_rate:  Minimum learning rate at end of decay (default 0).
        weight_decay:       L2 weight decay coefficient.
        beta1:              AdamW beta1.
        beta2:              AdamW beta2.
        eps:                AdamW epsilon.
        warmup_steps:       Number of linear warmup steps.

    Returns:
        (optimizer, lr_scheduler)
    """
    from utils.myloradict import collect_loradict_tensors

    parallel_cfg = get_pipeline_config()
    my_device = parallel_cfg["device"]

    # Collect trainable hypernetwork parameters on this stage
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []

    for name, param in model.hypernetwork.named_parameters():
        if param.device.type == "meta":
            continue
        if param.device != my_device:
            continue
        if not param.requires_grad:
            continue
        # Bias and LayerNorm weights typically should not be decayed
        if "bias" in name or "norm" in name.lower() or "layernorm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Collect trainable metalora tensors on this stage
    metalora_tensors: List[torch.Tensor] = []
    if hasattr(model, 'metalora') and model.metalora is not None:
        all_metalora = collect_loradict_tensors(model.metalora)
        for t in all_metalora:
            if t.device.type == "meta":
                continue
            if t.device != my_device:
                continue
            if not t.requires_grad:
                continue
            metalora_tensors.append(t)

    grouped_params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": metalora_tensors, "weight_decay": 0.0},  # metalora: no weight decay
    ]

    total_params = len(decay_params) + len(no_decay_params) + len(metalora_tensors)
    total_numel = (
        sum(p.numel() for p in decay_params)
        + sum(p.numel() for p in no_decay_params)
        + sum(t.numel() for t in metalora_tensors)
    )

    if is_main_process_per_node():
        logger.info(
            f"[Optimizer] Stage {parallel_cfg['stage']}: "
            f"{total_params} trainable params ({total_numel:,} elements), "
            f"decay={len(decay_params)}, no_decay={len(no_decay_params)}, "
            f"metalora={len(metalora_tensors)}"
        )

    optimizer = AdamW(
        grouped_params,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )

    # Compute min_lr_ratio for the scheduler
    min_lr_ratio = min_learning_rate / learning_rate if learning_rate > 0 else 0.0

    lr_scheduler = _get_linear_schedule_with_warmup_and_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )

    if is_main_process_per_node():
        logger.info(
            f"[Optimizer] AdamW created: lr={learning_rate}, min_lr={min_learning_rate}, "
            f"weight_decay={weight_decay}, betas=({beta1}, {beta2}), eps={eps}"
        )
        logger.info(
            f"[Scheduler] Linear warmup ({warmup_steps} steps) + "
            f"linear decay to min_lr={min_learning_rate} (total {num_training_steps} steps)"
        )

    return optimizer, lr_scheduler


# ---------------------------------------------------------------------------
# Parameter Audit
# ---------------------------------------------------------------------------

def _audit_local_parameters(
    model: torch.nn.Module,
    optimizer: Optional[AdamW],
    my_device: torch.device,
) -> Dict:
    """
    Audit parameters on the **current GPU** only.

    Returns a serialisable dict with all audit info for this GPU.
    """
    # Build set of optimizer param ids for fast lookup
    optim_param_ids: set = set()
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                optim_param_ids.add(id(p))

    params: List[Dict] = []  # per-parameter records

    # Counters
    llm_total = 0
    llm_on_gpu = 0
    hyper_total = 0
    hyper_on_gpu = 0
    hyper_trainable = 0
    hyper_in_optim = 0
    hyper_trainable_not_in_optim = 0
    llm_unexpected = 0

    # --- LLM parameters ---
    for name, param in model.llm.named_parameters():
        on_meta = param.device.type == "meta"
        on_gpu = (not on_meta) and (param.device == my_device)
        in_optim = id(param) in optim_param_ids
        llm_total += 1
        if on_gpu:
            llm_on_gpu += 1
            if param.requires_grad:
                llm_unexpected += 1
            if in_optim:
                llm_unexpected += 1
            params.append({
                "name": f"llm.{name}",
                "module": "llm",
                "requires_grad": param.requires_grad,
                "device": str(param.device),
                "shape": list(param.shape),
                "in_optimizer": in_optim,
            })

    # --- Hypernetwork parameters ---
    for name, param in model.hypernetwork.named_parameters():
        on_meta = param.device.type == "meta"
        on_gpu = (not on_meta) and (param.device == my_device)
        in_optim = id(param) in optim_param_ids
        hyper_total += 1
        if on_gpu:
            hyper_on_gpu += 1
            if param.requires_grad:
                hyper_trainable += 1
                if in_optim:
                    hyper_in_optim += 1
                else:
                    hyper_trainable_not_in_optim += 1
            params.append({
                "name": f"hypernetwork.{name}",
                "module": "hypernetwork",
                "requires_grad": param.requires_grad,
                "device": str(param.device),
                "shape": list(param.shape),
                "in_optimizer": in_optim,
            })

    # --- Metalora tensors ---
    metalora_total = 0
    metalora_on_gpu = 0
    metalora_trainable = 0
    metalora_in_optim = 0
    if hasattr(model, 'metalora') and model.metalora is not None:
        from utils.myloradict import collect_loradict_tensors
        metalora_tensors = collect_loradict_tensors(model.metalora)
        for i, t in enumerate(metalora_tensors):
            on_meta = t.device.type == "meta"
            on_gpu = (not on_meta) and (t.device == my_device)
            in_optim = id(t) in optim_param_ids
            metalora_total += 1
            if on_gpu:
                metalora_on_gpu += 1
                if t.requires_grad:
                    metalora_trainable += 1
                    if in_optim:
                        metalora_in_optim += 1
                params.append({
                    "name": f"metalora.tensor_{i}",
                    "module": "metalora",
                    "requires_grad": t.requires_grad,
                    "device": str(t.device),
                    "shape": list(t.shape),
                    "in_optimizer": in_optim,
                })

    parallel_cfg = get_pipeline_config()
    return {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "stage": parallel_cfg["stage"],
        "device": str(my_device),
        "llm_total": llm_total,
        "llm_on_gpu": llm_on_gpu,
        "hyper_total": hyper_total,
        "hyper_on_gpu": hyper_on_gpu,
        "hyper_trainable": hyper_trainable,
        "hyper_in_optim": hyper_in_optim,
        "hyper_trainable_not_in_optim": hyper_trainable_not_in_optim,
        "llm_unexpected": llm_unexpected,
        "metalora_total": metalora_total,
        "metalora_on_gpu": metalora_on_gpu,
        "metalora_trainable": metalora_trainable,
        "metalora_in_optim": metalora_in_optim,
        "params": params,
    }


def audit_parameters(
    model: torch.nn.Module,
    optimizer: Optional[AdamW] = None,
) -> Dict[str, Dict]:
    """
    Audit **all** parameters in ``ModelHypernetwork`` across every GPU
    within the current node, then gather and print a consolidated report
    on the node-local main GPU (local_rank 0).

    Every GPU audits its own parameters locally, then the results are
    gathered via ``dist.gather_object`` on the intra-node process group
    so that the main GPU outputs the audit for all 8 GPUs in one block.

    Args:
        model:     The ``ModelHypernetwork`` instance.
        optimizer: The optimizer (optional). If provided, the audit checks
                   whether each parameter is in the optimizer's param groups.

    Returns:
        Local audit dict for this GPU (the full gathered list is only
        available on the node-local main process).
    """
    parallel_cfg = get_pipeline_config()
    my_device = parallel_cfg["device"]
    total_gpus = parallel_cfg.get("total_gpus", 8)
    node_group = parallel_cfg.get("node_process_group", None)

    # --- Step 1: each GPU audits its own parameters ---
    local_audit = _audit_local_parameters(model, optimizer, my_device)

    # --- Step 2: gather all audits to node-local main GPU ---
    all_audits: Optional[List[Dict]] = None
    if dist.is_initialized() and node_group is not None:
        dst_global = dist.get_global_rank(node_group, 0)
        if is_main_process_per_node():
            all_audits = [None] * total_gpus
        else:
            all_audits = None
        dist.gather_object(local_audit, all_audits, dst=dst_global, group=node_group)
    else:
        # Single-GPU or non-distributed fallback
        all_audits = [local_audit] if is_main_process_per_node() else None

    # --- Step 3: main GPU prints consolidated report ---
    if is_main_process_per_node() and all_audits is not None:
        # Sort by rank for deterministic output
        all_audits = sorted(
            [a for a in all_audits if a is not None],
            key=lambda a: a["rank"],
        )

        logger.info("\n" + "=" * 80)
        logger.info("  Parameter Audit — All GPUs on this node")
        logger.info("=" * 80)

        global_ok = True

        for audit in all_audits:
            rank = audit["rank"]
            stage = audit["stage"]
            device = audit["device"]

            logger.info(f"\n  {'─' * 76}")
            logger.info(f"  GPU {rank}  (stage {stage}, device {device})")
            logger.info(f"  {'─' * 76}")

            # LLM summary
            logger.info(
                f"    LLM: {audit['llm_total']} total params, "
                f"{audit['llm_on_gpu']} on this GPU (all frozen)"
            )

            # Hypernetwork summary
            logger.info(
                f"    Hypernetwork: {audit['hyper_total']} total params, "
                f"{audit['hyper_on_gpu']} on this GPU, "
                f"{audit['hyper_trainable']} trainable"
            )
            if optimizer is not None:
                logger.info(
                    f"    Optimizer coverage: "
                    f"{audit['hyper_in_optim']}/{audit['hyper_trainable']} "
                    f"trainable params in optimizer"
                )
                if audit["hyper_trainable_not_in_optim"] > 0:
                    logger.error(
                        f"    ✗ {audit['hyper_trainable_not_in_optim']} "
                        f"trainable params NOT in optimizer!"
                    )
                    global_ok = False

            # Detailed hypernetwork params
            hyper_params = [
                p for p in audit["params"] if p["module"] == "hypernetwork"
            ]
            if hyper_params:
                logger.info(
                    f"    {'Name':<58s} {'Grad':>5s} {'Optim':>5s} {'Shape'}"
                )
                for p in sorted(hyper_params, key=lambda x: x["name"]):
                    grad_str = "✓" if p["requires_grad"] else "✗"
                    optim_str = "✓" if p["in_optimizer"] else "✗"
                    short = p["name"].replace("hypernetwork.", "")
                    logger.info(
                        f"    {short:<58s} {grad_str:>5s} "
                        f"{optim_str:>5s} {p['shape']}"
                    )

            # LLM unexpected check
            if audit["llm_unexpected"] > 0:
                logger.error(
                    f"    ✗ LLM has {audit['llm_unexpected']} unexpected "
                    f"(requires_grad=True or in optimizer)!"
                )
                # List the offending LLM params
                for p in audit["params"]:
                    if p["module"] == "llm":
                        if p["requires_grad"]:
                            logger.warning(
                                f"      UNEXPECTED: {p['name']} requires_grad=True!"
                            )
                        if p["in_optimizer"]:
                            logger.error(
                                f"      UNEXPECTED: {p['name']} is in optimizer!"
                            )
                global_ok = False
            else:
                logger.info(
                    f"    LLM: {audit['llm_on_gpu']} params on GPU "
                    f"(all frozen, none in optimizer — OK)"
                )

        # --- Global summary ---
        logger.info(f"\n  {'─' * 76}")
        total_hyper_trainable = sum(a["hyper_trainable"] for a in all_audits)
        total_hyper_in_optim = sum(a["hyper_in_optim"] for a in all_audits)
        total_llm_on_node = sum(a["llm_on_gpu"] for a in all_audits)
        total_hyper_on_node = sum(a["hyper_on_gpu"] for a in all_audits)
        logger.info(
            f"  Node summary: {len(all_audits)} GPUs, "
            f"LLM params on node={total_llm_on_node}, "
            f"Hypernetwork params on node={total_hyper_on_node}, "
            f"trainable={total_hyper_trainable}, "
            f"in optimizer={total_hyper_in_optim}"
        )
        logger.info("=" * 80)

        if global_ok:
            logger.info("  Parameter Audit PASSED ✓")
        else:
            logger.error("  Parameter Audit FAILED ✗")

    return local_audit
