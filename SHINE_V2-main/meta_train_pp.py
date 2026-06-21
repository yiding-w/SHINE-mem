#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline Parallel (PP) specific training functions for meta_train.py.

This module contains the PP-specific implementations of:
  - train_step(): forward + backward with multi micro-batch pipeline parallelism
  - run_evaluation(): evaluation loop with pipeline communication
  - nccl_p2p_warmup(): NCCL P2P communicator warmup for pipeline stages

These functions are loaded dynamically by meta_train.py based on cfg.parallel.mode.
"""

import os
import torch
import math
import time
import logging

import torch.distributed as dist
import wandb

from utils.myparallel import (
    get_pipeline_config,
    is_main_process, is_main_process_per_node,
    pipeline_send, pipeline_recv,
    barrier,
)
from utils.mylog import format_duration
from hypernetwork.model_hypernetwork import ModelHypernetwork

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _ShapeOnlyTensor import (used for non-data-loading stages)
# When meta_train.py is run as __main__, we need to import from __main__
# to avoid re-executing the entire module.
# ---------------------------------------------------------------------------
import sys
if '__main__' in sys.modules and hasattr(sys.modules['__main__'], '_ShapeOnlyTensor'):
    _ShapeOnlyTensor = sys.modules['__main__']._ShapeOnlyTensor
else:
    from meta_train import _ShapeOnlyTensor


# ---------------------------------------------------------------------------
# PP train_step
# ---------------------------------------------------------------------------

def train_step(
    model: ModelHypernetwork,
    context_ids_list: list,
    context_lengths_list: list,
    conversation_ids_list: list,
    labels_list: list,
    micro_batch_size: int,
    batch_id: str,
    my_device: torch.device,
    norm_stage: int = 5,
    log_detail: bool = False,
    grad_accum_steps: int = 1,
    distill_conversation_ids_list: list = None,
    distill_labels_list: list = None,
    distill_micro_batch_size: int = None,
    distill_loss_fn=None,
):
    """
    Execute one forward + backward pass with multi micro-batch pipeline
    parallelism to reduce bubble time.

    The forward is split into five phases:
      Phase A:  All micro-batches do Step1 (context → memory_states)
      Phase A': All distill micro-batches do teacher forward (no_grad, no lora)
      Phase B:  All micro-batches do Hypernetwork + lora_scatter
      Phase C:  All micro-batches do Step4 (conversation → hidden_states)
      Phase C': All distill micro-batches do student forward (with grad on loradict)

    Then labels are transferred and losses computed for all micro-batches.
    Distillation loss is computed from teacher/student outputs and added to main loss.
    Backward processes micro-batches in REVERSE order for FIFO matching.

    Args:
        context_ids_list:      List of (mb_size, S_ctx) tensors per micro-batch.
        context_lengths_list:  List of (mb_size,) tensors per micro-batch.
        conversation_ids_list: List of (mb_size, S_conv) tensors per micro-batch.
        labels_list:           List of (mb_size, S_conv) tensors per micro-batch.
        micro_batch_size:      Size of each micro-batch.
        batch_id:              Unique identifier for this batch.
        my_device:             Current GPU device.
        norm_stage:            Pipeline stage that owns norm + lm_head.
        log_detail:            If True, also return per-token loss.
        grad_accum_steps:      Number of gradient accumulation steps.
        distill_conversation_ids_list: List of (distill_mb_size, S_conv) tensors for distillation.
                                       None if distillation is disabled.
        distill_labels_list:   List of (distill_mb_size, S_conv) label tensors for distillation masking.
                               None if distillation is disabled.
        distill_micro_batch_size: Size of each distill micro-batch.
        distill_loss_fn:       DistillLossWrapper instance (or None if disabled).
                               Called as: distill_loss_fn(teacher_output, student_output, labels)

    Returns:
        loss_value (float): average loss across all micro-batches on stage 0,
            0.0 on all other stages.
    """
    num_mb = len(context_ids_list)
    do_distill = (distill_loss_fn is not None and distill_conversation_ids_list is not None)
    num_distill_mb = len(distill_conversation_ids_list) if do_distill else 0
    distill_mode = distill_loss_fn.mode if do_distill else None

    # ================================================================
    # Forward: 5-phase multi micro-batch pipeline
    # ================================================================
    (all_hidden_states, all_step1_anchors, all_step4_anchors,
     all_memory_states, distill_teacher_outputs, distill_student_outputs,
     per_mb_sq_norms) = \
        model.pipeline_forward_train_multi_mb(
            context_ids_list=context_ids_list,
            context_lengths_list=context_lengths_list,
            conversation_ids_list=conversation_ids_list,
            micro_batch_size=micro_batch_size,
            batch_id=batch_id,
            distill_conversation_ids_list=distill_conversation_ids_list if do_distill else None,
            distill_micro_batch_size=distill_micro_batch_size if do_distill else None,
            distill_mode=distill_mode,
            grad_accum_steps=grad_accum_steps,
        )

    # ================================================================
    # Transfer labels and compute loss for each micro-batch
    #
    # FIFO order on embed_stage→norm_stage channel:
    #   [labels_mb0, labels_mb1, ..., labels_mbN-1]
    # All micro-batches' labels are sent/received in order.
    # ================================================================
    parallel_cfg = get_pipeline_config()
    my_stage = parallel_cfg["stage"]
    embed_stage = model._embed_stage

    # Transfer all labels first (in mb order for FIFO)
    norm_labels_list = []
    for mb_idx in range(num_mb):
        labels = labels_list[mb_idx]
        if embed_stage != norm_stage:
            if my_stage == embed_stage and labels is not None:
                pipeline_send(labels.contiguous(), dst_stage=norm_stage, tag=9998)
            if my_stage == norm_stage:
                hidden = all_hidden_states[mb_idx]
                # On norm stage, labels is None (not loaded here), so get seq_len from hidden_states
                seq_len = hidden.size(1) if hidden is not None else conversation_ids_list[mb_idx].size(1)
                recv_labels = torch.empty(
                    (micro_batch_size, seq_len),
                    dtype=torch.long, device=my_device,
                )
                pipeline_recv(recv_labels, src_stage=embed_stage, tag=9998)
                norm_labels_list.append(recv_labels)
            else:
                norm_labels_list.append(None)
        else:
            norm_labels_list.append(labels)

    # Transfer distill labels (if distillation is enabled)
    # FIFO order on embed_stage→norm_stage channel (tag=9995):
    #   [distill_labels_mb0, distill_labels_mb1, ..., distill_labels_mbN-1]
    norm_distill_labels_list = []
    if do_distill:
        for mb_idx in range(num_distill_mb):
            d_labels = distill_labels_list[mb_idx] if distill_labels_list is not None else None
            if embed_stage != norm_stage:
                if my_stage == embed_stage and d_labels is not None:
                    pipeline_send(d_labels.contiguous(), dst_stage=norm_stage, tag=9995)
                if my_stage == norm_stage:
                    d_student = distill_student_outputs[mb_idx] if distill_student_outputs else None
                    if d_student is not None:
                        d_seq_len = d_student.size(1)
                    elif distill_conversation_ids_list is not None:
                        d_seq_len = distill_conversation_ids_list[mb_idx].size(1)
                    else:
                        d_seq_len = conversation_ids_list[0].size(1) if hasattr(conversation_ids_list[0], 'size') else 1
                    recv_d_labels = torch.empty(
                        (distill_micro_batch_size, d_seq_len),
                        dtype=torch.long, device=my_device,
                    )
                    pipeline_recv(recv_d_labels, src_stage=embed_stage, tag=9995)
                    norm_distill_labels_list.append(recv_d_labels)
                else:
                    norm_distill_labels_list.append(None)
            else:
                norm_distill_labels_list.append(d_labels)

    # Compute loss for each micro-batch
    all_losses = []
    all_distill_losses = [None] * num_mb  # Separate distill losses for Phase C' backward
    all_loss_values = []
    all_per_token_losses = []
    all_distill_loss_values = []
    for mb_idx in range(num_mb):
        hidden_states = all_hidden_states[mb_idx]
        labels = norm_labels_list[mb_idx]

        if hidden_states is not None:
            if not isinstance(labels, torch.Tensor):
                raise RuntimeError(
                    f"Loss computation on norm stage {norm_stage} requires real labels, "
                    f"but got {type(labels).__name__}."
                )
            shift_hs = hidden_states[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            B, S_minus_1, H = shift_hs.shape
            flat_hs = shift_hs.reshape(B * S_minus_1, H)
            flat_labels = shift_labels.reshape(B * S_minus_1)

            # Liger Fused Linear Cross-Entropy: fuses lm_head GEMM + softmax +
            # NLL + backward into one Triton kernel, never materialising the
            # [B*T, V=248k] logits tensor. Set SHINE_FLCE=0 to fall back.
            use_flce = os.environ.get("SHINE_FLCE", "1") not in ("0", "", "false")
            if use_flce:
                from utils.liger_patch import fused_lm_head_loss
                loss = fused_lm_head_loss(
                    flat_hs, model.llm.lm_head.weight, flat_labels,
                    ignore_index=-100, reduction="mean",
                )
            else:
                # Chunked CE fallback: never materialise full [B*T, V] logits
                chunk = max(1, 1024 // max(1, B))
                total_loss = torch.zeros((), device=flat_hs.device, dtype=torch.float32)
                total_count = torch.zeros((), device=flat_hs.device, dtype=torch.long)
                for i in range(0, flat_hs.shape[0], chunk):
                    j = min(i + chunk, flat_hs.shape[0])
                    chunk_logits = model.llm.lm_head(flat_hs[i:j])
                    chunk_loss = torch.nn.functional.cross_entropy(
                        chunk_logits, flat_labels[i:j], ignore_index=-100, reduction="sum",
                    )
                    total_loss = total_loss + chunk_loss.float()
                    total_count = total_count + (flat_labels[i:j] != -100).sum()
                loss = (total_loss / total_count.clamp(min=1)).to(flat_hs.dtype)

            # Scale loss by 1/(num_mb * grad_accum_steps) so gradients are
            # averaged across micro-batches AND gradient accumulation steps
            loss = loss / (num_mb * grad_accum_steps)
            all_losses.append(loss)
            all_loss_values.append(loss.item() * num_mb * grad_accum_steps)  # unscaled for logging

            if log_detail:
                with torch.no_grad():
                    # Chunked per-token loss (debug only): materialise logits
                    # in chunks to avoid allocating full [B*T, V] tensor.
                    per_token_flat = torch.zeros(B * S_minus_1, device=flat_hs.device, dtype=torch.float32)
                    chunk = max(1, 1024 // max(1, B))
                    for i in range(0, flat_hs.shape[0], chunk):
                        j = min(i + chunk, flat_hs.shape[0])
                        chunk_logits = model.llm.lm_head(flat_hs[i:j])
                        chunk_ptl = torch.nn.functional.cross_entropy(
                            chunk_logits, flat_labels[i:j], ignore_index=-100, reduction="none",
                        ).float()
                        per_token_flat[i:j] = chunk_ptl
                    all_per_token_losses.append(per_token_flat)
        else:
            all_losses.append(None)
            all_loss_values.append(0.0)
            if log_detail:
                all_per_token_losses.append(None)

    # ================================================================
    # Compute distillation loss for each distill micro-batch
    #
    # The distill loss is computed on the norm stage where both teacher
    # and student outputs are available. The loss is added to the
    # corresponding main loss tensor so backward propagates through both.
    #
    # For logits mode: student_output goes through lm_head to get logits.
    # For hidden_states mode: student_output is used directly.
    # ================================================================
    if do_distill:
        for mb_idx in range(num_distill_mb):
            teacher_out = distill_teacher_outputs[mb_idx] if distill_teacher_outputs else None
            student_hidden = distill_student_outputs[mb_idx] if distill_student_outputs else None
            d_labels = norm_distill_labels_list[mb_idx] if norm_distill_labels_list else None

            if student_hidden is not None and teacher_out is not None:
                # Get student output in the appropriate format
                if distill_mode == "logits":
                    student_out = model.llm.lm_head(student_hidden)
                else:
                    student_out = student_hidden

                # Compute distill loss (coefficient is already applied inside)
                d_loss = distill_loss_fn(teacher_out, student_out, d_labels)
                # Scale by 1/(num_distill_mb * grad_accum_steps)
                d_loss = d_loss / (num_distill_mb * grad_accum_steps)

                # Store distill loss separately (NOT added to all_losses)
                # It will be passed to pipeline_backward_multi_mb for Phase C' backward
                loradict_idx = mb_idx % num_mb
                if all_distill_losses[loradict_idx] is None:
                    all_distill_losses[loradict_idx] = d_loss
                else:
                    all_distill_losses[loradict_idx] = all_distill_losses[loradict_idx] + d_loss

                all_distill_loss_values.append(d_loss.item() * num_distill_mb * grad_accum_steps)
            else:
                all_distill_loss_values.append(0.0)

    # Add distill losses to main losses for combined backward on norm stage
    # Phase C' backward handles distill_losses separately (with retain_graph=True).
    # Phase C+B backward only needs the CE losses (all_losses), because:
    #   - On norm stage: CE loss.backward() propagates CE gradients through Phase C
    #   - Distill gradients were already propagated in Phase C' backward
    # Note: On non-norm stages, the anchors handle both paths via retain_graph.

    # ================================================================
    # Backward: multi micro-batch pipeline backward
    # ================================================================
    torch.cuda.nvtx.range_push("Backward")
    model.pipeline_backward_multi_mb(
        losses=all_losses,
        all_step1_anchors=all_step1_anchors,
        all_step4_anchors=all_step4_anchors,
        all_memory_states=all_memory_states,
        distill_losses=all_distill_losses,
    )
    torch.cuda.nvtx.range_pop()  # Backward

    # ================================================================
    # Transfer loss from norm stage → stage 0 for logging
    #
    # FIFO order on norm_stage→stage0 channel:
    #   [avg_loss_scalar, avg_distill_loss_scalar]
    # ================================================================
    avg_loss_value = sum(all_loss_values) / num_mb if all_loss_values else 0.0
    avg_distill_loss_value = (
        sum(all_distill_loss_values) / num_distill_mb
        if all_distill_loss_values else 0.0
    )

    if norm_stage != 0:
        if my_stage == norm_stage:
            loss_tensor = torch.tensor(
                [avg_loss_value, avg_distill_loss_value],
                dtype=torch.float32, device=my_device,
            )
            pipeline_send(loss_tensor, dst_stage=0, tag=9999)
        elif my_stage == 0:
            loss_tensor = torch.empty(2, dtype=torch.float32, device=my_device)
            pipeline_recv(loss_tensor, src_stage=norm_stage, tag=9999)
            avg_loss_value = loss_tensor[0].item()
            avg_distill_loss_value = loss_tensor[1].item()

    # Transfer per-token loss for detail logging
    # FIFO order on norm_stage→embed_stage: [length, per_token_loss] (after scalar loss)
    per_token_loss_on_embed = None
    if log_detail:
        # Concatenate per-token losses from all micro-batches
        if my_stage == norm_stage:
            non_none_ptl = [p for p in all_per_token_losses if p is not None]
            per_token_loss_local = torch.cat(non_none_ptl) if non_none_ptl else None
        else:
            per_token_loss_local = None

        if norm_stage != embed_stage:
            if my_stage == norm_stage and per_token_loss_local is not None:
                length_t = torch.tensor(
                    [per_token_loss_local.numel()], dtype=torch.long, device=my_device)
                pipeline_send(length_t, dst_stage=embed_stage, tag=9997)
                pipeline_send(per_token_loss_local.contiguous(), dst_stage=embed_stage, tag=9996)
            elif my_stage == embed_stage:
                length_t = torch.empty(1, dtype=torch.long, device=my_device)
                pipeline_recv(length_t, src_stage=norm_stage, tag=9997)
                num_elements = length_t.item()
                per_token_loss_on_embed = torch.empty(
                    num_elements, dtype=torch.float32, device=my_device)
                pipeline_recv(per_token_loss_on_embed, src_stage=norm_stage, tag=9996)
        else:
            if my_stage == norm_stage:
                per_token_loss_on_embed = per_token_loss_local

    # Clean up
    del all_hidden_states, all_losses, all_memory_states
    del all_step1_anchors, all_step4_anchors
    del distill_teacher_outputs, distill_student_outputs

    # Compute avg_regu_sq_norm (mean across micro-batches) for logging
    avg_regu_sq_norm = sum(per_mb_sq_norms) / len(per_mb_sq_norms) if per_mb_sq_norms else 0.0

    if log_detail:
        return avg_loss_value, per_token_loss_on_embed, avg_distill_loss_value, avg_regu_sq_norm, per_mb_sq_norms
    return avg_loss_value, avg_distill_loss_value, avg_regu_sq_norm, per_mb_sq_norms


# ---------------------------------------------------------------------------
# Full training step (data → forward → backward → optimizer → logging)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PP run_evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    val_loader,
    model,
    my_device,
    parallel_cfg,
    make_sdpa_ctx,
    global_step: int,
    local_batch_size: int,
    local_micro_batch_size: int,
    vocab_size: int,
    context_seq_len: int,
    conv_seq_len: int,
    num_mem_token: int,
    use_wandb: bool = False,
    node_rank: int = 0,
    t_start: float = 0.0,
    max_steps: int = 0,
    ema_time_per_step: float = 0.0,
    distill_loss_fn=None,
    distill_micro_batch_size: int = 1,
    baseline_mode: bool = False,
):
    """
    Run evaluation on the validation set.

    Iterates over the entire validation set, computing forward-only loss
    for each batch using the same pipeline forward as training (but without
    backward or optimizer step).

    This is a collective operation: ALL pipeline stages and DP ranks must
    call it simultaneously.

    Args:
        val_loader: PipelineDataLoader for the validation set.
        model: The ModelHypernetwork model.
        my_device: Current GPU device.
        parallel_cfg: Pipeline parallel config dict.
        make_sdpa_ctx: Factory for SDPA context manager.
        global_step: Current training step (for logging).
        local_batch_size: Per-node batch size.
        local_micro_batch_size: Micro-batch size.
        vocab_size: Vocabulary size.
        context_seq_len: Context sequence length.
        conv_seq_len: Conversation sequence length.
        num_mem_token: Number of memory token placeholders.
        use_wandb: Whether to log to wandb.
        node_rank: Node rank for logging.
        t_start: Training start time for elapsed calculation.
        max_steps: Total training steps for ETA calculation.
        ema_time_per_step: EMA-smoothed time per step for ETA.
        baseline_mode: If True, run base LLM only (no hypernetwork, no LoRA,
            no detach_state). Used for computing a baseline loss/ppl.

    Returns:
        (avg_loss, avg_ppl) on the main process, (0.0, 0.0) on others.
    """
    if val_loader is None:
        return 0.0, 0.0

    _eval_start_time = time.time()
    model.eval()

    # Use a fresh zero-initialized detach_state for evaluation.
    # Training state is saved and restored after eval completes.
    # In baseline_mode, skip eval_context entirely (no detach_state involvement)
    if not baseline_mode:
        _ds_ctx = model.detach_state.eval_context(eval_local_batch_size=local_batch_size) if model.detach_state is not None else None
    else:
        _ds_ctx = None
    if _ds_ctx is not None:
        _ds_ctx.__enter__()

    num_micro_batches = local_batch_size // local_micro_batch_size
    val_loader.set_epoch(0)  # Fixed epoch for reproducibility
    val_iter = iter(val_loader)

    total_loss = 0.0
    total_distill_loss = 0.0
    num_batches = 0
    do_eval_distill = (distill_loss_fn is not None) and not baseline_mode
    _eval_prev_repo_per_mb = [None] * num_micro_batches  # Track repo for repo-change reset during eval
    _eval_running_reset_ratio = 0.0
    _eval_running_mean_update_step = 0.0
    _eval_running_regu_sq_norm = 0.0
    _eval_repo_reset_count = 0

    while True:
        try:
            batch_data = next(val_iter)
        except StopIteration:
            break

        # --- Extract tensors and split into micro-batch lists ---
        if val_loader.is_first_stage and batch_data["micro_batches"]:
            mbs = batch_data["micro_batches"]
            context_ids_list = [mb["context_ids"].to(my_device) for mb in mbs]
            conversation_ids_list = [mb["conversation_ids"].to(my_device) for mb in mbs]
            labels_list = [mb["labels"].to(my_device) for mb in mbs]
            ctx_lengths_list = [mb["context_lengths"].to(my_device) for mb in mbs]
            # Distillation data for evaluation
            distill_list = [
                {k: v.to(my_device) if isinstance(v, torch.Tensor) else v
                 for k, v in mb["distill"].items()}
                if mb.get("distill") is not None else None
                for mb in mbs
            ]
            # Build distill micro-batch lists
            if do_eval_distill and any(d is not None for d in distill_list):
                all_distill_conv = torch.cat(
                    [d["conversation_ids"] for d in distill_list if d is not None], dim=0
                )
                all_distill_labels = torch.cat(
                    [d["labels"] for d in distill_list if d is not None], dim=0
                )
                num_distill_mbs = all_distill_conv.size(0) // distill_micro_batch_size
                distill_conversation_ids_list = [
                    all_distill_conv[i * distill_micro_batch_size:(i + 1) * distill_micro_batch_size]
                    for i in range(num_distill_mbs)
                ]
                distill_labels_list = [
                    all_distill_labels[i * distill_micro_batch_size:(i + 1) * distill_micro_batch_size]
                    for i in range(num_distill_mbs)
                ]
            else:
                distill_conversation_ids_list = None
                distill_labels_list = None
        else:
            context_ids_list = [
                _ShapeOnlyTensor(
                    (local_micro_batch_size, context_seq_len + num_mem_token),
                    name=f"val_context_ids_mb{i}",
                ) for i in range(num_micro_batches)
            ]
            conversation_ids_list = [
                _ShapeOnlyTensor(
                    (local_micro_batch_size, conv_seq_len),
                    name=f"val_conversation_ids_mb{i}",
                ) for i in range(num_micro_batches)
            ]
            labels_list = [None] * num_micro_batches
            ctx_lengths_list = [
                torch.full((local_micro_batch_size,), context_seq_len, dtype=torch.long, device=my_device)
                for _ in range(num_micro_batches)
            ]
            # Non-first stages must also participate in distill pipeline communication
            if do_eval_distill:
                num_distill_mbs = local_batch_size // distill_micro_batch_size
                distill_conversation_ids_list = [
                    _ShapeOnlyTensor(
                        (distill_micro_batch_size, conv_seq_len),
                        name=f"val_distill_conv_ids_mb{i}",
                    ) for i in range(num_distill_mbs)
                ]
                distill_labels_list = [None] * num_distill_mbs
            else:
                distill_conversation_ids_list = None
                distill_labels_list = None

        # --- Baseline mode: simple LLM forward without hypernetwork/LoRA ---
        if baseline_mode:
            batch_id = f"eval_baseline_s{global_step}_b{num_batches}"
            with make_sdpa_ctx():
                num_mb = len(conversation_ids_list)
                my_stage = parallel_cfg["stage"]
                embed_stage = model._embed_stage
                norm_stage = model._norm_stage

                # Forward each micro-batch through base LLM (no LoRA)
                all_hidden_states = []
                with torch.no_grad():
                    for mb_idx in range(num_mb):
                        hidden, _ = model.llm_forward_no_grad(
                            input_ids=conversation_ids_list[mb_idx],
                            attention_mask=None,
                            loradict=None,
                            use_mem_token=False,
                            batch_id=f"{batch_id}_mb{mb_idx}",
                        )
                        all_hidden_states.append(hidden)

                # Transfer labels from embed stage to norm stage
                norm_labels_list = []
                for mb_idx in range(num_mb):
                    labels = labels_list[mb_idx]
                    if embed_stage != norm_stage:
                        if my_stage == embed_stage and labels is not None:
                            pipeline_send(labels.contiguous(), dst_stage=norm_stage, tag=9998)
                        if my_stage == norm_stage:
                            hidden = all_hidden_states[mb_idx]
                            seq_len = hidden.size(1) if hidden is not None else conv_seq_len
                            recv_labels = torch.empty(
                                (local_micro_batch_size, seq_len),
                                dtype=torch.long, device=my_device,
                            )
                            pipeline_recv(recv_labels, src_stage=embed_stage, tag=9998)
                            norm_labels_list.append(recv_labels)
                        else:
                            norm_labels_list.append(None)
                    else:
                        norm_labels_list.append(labels)

                # Compute loss on norm stage
                all_loss_values = []
                for mb_idx in range(num_mb):
                    hidden_states = all_hidden_states[mb_idx]
                    labels = norm_labels_list[mb_idx]
                    if hidden_states is not None:
                        shift_hs = hidden_states[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        B, S_minus_1, H = shift_hs.shape
                        flat_hs = shift_hs.reshape(B * S_minus_1, H)
                        flat_labels = shift_labels.reshape(B * S_minus_1)

                        use_flce = os.environ.get("SHINE_FLCE", "1") not in ("0", "", "false")
                        if use_flce:
                            from utils.liger_patch import fused_lm_head_loss
                            loss = fused_lm_head_loss(
                                flat_hs, model.llm.lm_head.weight, flat_labels,
                                ignore_index=-100, reduction="mean",
                            )
                        else:
                            chunk = max(1, 1024 // max(1, B))
                            _total_loss_bl = torch.zeros((), device=flat_hs.device, dtype=torch.float32)
                            _total_count_bl = torch.zeros((), device=flat_hs.device, dtype=torch.long)
                            for i in range(0, flat_hs.shape[0], chunk):
                                j = min(i + chunk, flat_hs.shape[0])
                                chunk_logits = model.llm.lm_head(flat_hs[i:j])
                                chunk_loss = torch.nn.functional.cross_entropy(
                                    chunk_logits, flat_labels[i:j], ignore_index=-100, reduction="sum",
                                )
                                _total_loss_bl = _total_loss_bl + chunk_loss.float()
                                _total_count_bl = _total_count_bl + (flat_labels[i:j] != -100).sum()
                            loss = (_total_loss_bl / _total_count_bl.clamp(min=1)).to(flat_hs.dtype)
                        all_loss_values.append(loss.item())
                    else:
                        all_loss_values.append(0.0)

                # Transfer loss from norm stage -> stage 0
                avg_loss_value = sum(all_loss_values) / num_mb if all_loss_values else 0.0
                if norm_stage != 0:
                    if my_stage == norm_stage:
                        loss_tensor = torch.tensor(
                            [avg_loss_value, 0.0],
                            dtype=torch.float32, device=my_device,
                        )
                        pipeline_send(loss_tensor, dst_stage=0, tag=9999)
                    elif my_stage == 0:
                        loss_tensor = torch.empty(2, dtype=torch.float32, device=my_device)
                        pipeline_recv(loss_tensor, src_stage=norm_stage, tag=9999)
                        avg_loss_value = loss_tensor[0].item()

                del all_hidden_states

            # Clear model caches after each eval batch
            model.invalidate_input_cache()

            total_loss += avg_loss_value
            num_batches += 1
            continue

        # --- Repo-change reset during eval (same logic as training) ---
        if model.detach_state is not None and val_loader.is_first_stage:
            for _mb_i in range(len(mbs) if val_loader.is_first_stage and batch_data["micro_batches"] else 0):
                _ei = mbs[_mb_i].get("extra_info", None) if _mb_i < len(mbs) else None
                if _ei is not None and isinstance(_ei, list) and len(_ei) > 0:
                    _cur_repo = _ei[0].get("repo") if isinstance(_ei[0], dict) else None
                    if _cur_repo is not None:
                        if (_mb_i < len(_eval_prev_repo_per_mb) and
                                _eval_prev_repo_per_mb[_mb_i] is not None and
                                _cur_repo != _eval_prev_repo_per_mb[_mb_i]):
                            # Reset all samples in this micro-batch
                            for _si in range(_mb_i * local_micro_batch_size, (_mb_i + 1) * local_micro_batch_size):
                                model.detach_state.reset_slice(_si)
                            _eval_repo_reset_count += 1
                        _eval_prev_repo_per_mb[_mb_i] = _cur_repo

        # --- Forward only (no backward) ---
        batch_id = f"eval_s{global_step}_b{num_batches}"
        with make_sdpa_ctx():
            # Use train_step's forward logic but only compute loss
            num_mb = len(context_ids_list)
            _eval_do_distill = (do_eval_distill and distill_conversation_ids_list is not None)
            _eval_distill_mode = distill_loss_fn.mode if _eval_do_distill else None
            _eval_num_distill_mb = len(distill_conversation_ids_list) if _eval_do_distill else 0

            all_hidden_states, all_step1_anchors, all_step4_anchors, all_memory_states, \
                distill_teacher_outputs, distill_student_outputs, _eval_sq_norms = \
                model.pipeline_forward_train_multi_mb(
                    context_ids_list=context_ids_list,
                    context_lengths_list=ctx_lengths_list,
                    conversation_ids_list=conversation_ids_list,
                    micro_batch_size=local_micro_batch_size,
                    batch_id=batch_id,
                    distill_conversation_ids_list=distill_conversation_ids_list if _eval_do_distill else None,
                    distill_micro_batch_size=distill_micro_batch_size if _eval_do_distill else None,
                    distill_mode=_eval_distill_mode,
                )

            # Transfer labels and compute loss
            my_stage = parallel_cfg["stage"]
            embed_stage = model._embed_stage
            norm_stage = model._norm_stage

            norm_labels_list = []
            for mb_idx in range(num_mb):
                labels = labels_list[mb_idx]
                if embed_stage != norm_stage:
                    if my_stage == embed_stage and labels is not None:
                        pipeline_send(labels.contiguous(), dst_stage=norm_stage, tag=9998)
                    if my_stage == norm_stage:
                        hidden = all_hidden_states[mb_idx]
                        seq_len = hidden.size(1) if hidden is not None else conv_seq_len
                        recv_labels = torch.empty(
                            (local_micro_batch_size, seq_len),
                            dtype=torch.long, device=my_device,
                        )
                        pipeline_recv(recv_labels, src_stage=embed_stage, tag=9998)
                        norm_labels_list.append(recv_labels)
                    else:
                        norm_labels_list.append(None)
                else:
                    norm_labels_list.append(labels)

            all_loss_values = []
            for mb_idx in range(num_mb):
                hidden_states = all_hidden_states[mb_idx]
                labels = norm_labels_list[mb_idx]
                if hidden_states is not None:
                    shift_hs = hidden_states[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    B, S_minus_1, H = shift_hs.shape
                    flat_hs = shift_hs.reshape(B * S_minus_1, H)
                    flat_labels = shift_labels.reshape(B * S_minus_1)

                    use_flce = os.environ.get("SHINE_FLCE", "1") not in ("0", "", "false")
                    if use_flce:
                        from utils.liger_patch import fused_lm_head_loss
                        loss = fused_lm_head_loss(
                            flat_hs, model.llm.lm_head.weight, flat_labels,
                            ignore_index=-100, reduction="mean",
                        )
                    else:
                        chunk = max(1, 1024 // max(1, B))
                        total_loss = torch.zeros((), device=flat_hs.device, dtype=torch.float32)
                        total_count = torch.zeros((), device=flat_hs.device, dtype=torch.long)
                        for i in range(0, flat_hs.shape[0], chunk):
                            j = min(i + chunk, flat_hs.shape[0])
                            chunk_logits = model.llm.lm_head(flat_hs[i:j])
                            chunk_loss = torch.nn.functional.cross_entropy(
                                chunk_logits, flat_labels[i:j], ignore_index=-100, reduction="sum",
                            )
                            total_loss = total_loss + chunk_loss.float()
                            total_count = total_count + (flat_labels[i:j] != -100).sum()
                        loss = (total_loss / total_count.clamp(min=1)).to(flat_hs.dtype)
                    all_loss_values.append(loss.item())
                else:
                    all_loss_values.append(0.0)

            # Compute distill loss for evaluation (if enabled)
            eval_distill_loss_values = []
            if _eval_do_distill:
                # Transfer distill labels from embed stage to norm stage
                eval_norm_distill_labels = []
                for mb_idx in range(_eval_num_distill_mb):
                    d_labels = distill_labels_list[mb_idx] if distill_labels_list is not None else None
                    if embed_stage != norm_stage:
                        if my_stage == embed_stage and d_labels is not None:
                            pipeline_send(d_labels.contiguous(), dst_stage=norm_stage, tag=9995)
                        if my_stage == norm_stage:
                            d_student = distill_student_outputs[mb_idx] if distill_student_outputs else None
                            if d_student is not None:
                                d_seq_len = d_student.size(1)
                            else:
                                d_seq_len = conv_seq_len
                            recv_d_labels = torch.empty(
                                (distill_micro_batch_size, d_seq_len),
                                dtype=torch.long, device=my_device,
                            )
                            pipeline_recv(recv_d_labels, src_stage=embed_stage, tag=9995)
                            eval_norm_distill_labels.append(recv_d_labels)
                        else:
                            eval_norm_distill_labels.append(None)
                    else:
                        eval_norm_distill_labels.append(d_labels)

                # Compute distill loss on norm stage
                for mb_idx in range(_eval_num_distill_mb):
                    teacher_out = distill_teacher_outputs[mb_idx] if distill_teacher_outputs else None
                    student_hidden = distill_student_outputs[mb_idx] if distill_student_outputs else None
                    d_labels = eval_norm_distill_labels[mb_idx] if eval_norm_distill_labels else None

                    if student_hidden is not None and teacher_out is not None:
                        if _eval_distill_mode == "logits":
                            student_out = model.llm.lm_head(student_hidden)
                        else:
                            student_out = student_hidden
                        # Compute raw distill loss (without coefficient for fair comparison)
                        d_loss = distill_loss_fn.loss_fn(teacher_out, student_out, d_labels)
                        eval_distill_loss_values.append(d_loss.item())
                    else:
                        eval_distill_loss_values.append(0.0)

            avg_eval_distill_loss = (
                sum(eval_distill_loss_values) / len(eval_distill_loss_values)
                if eval_distill_loss_values else 0.0
            )

            # Transfer loss from norm stage -> stage 0
            avg_loss_value = sum(all_loss_values) / num_mb if all_loss_values else 0.0
            if norm_stage != 0:
                if my_stage == norm_stage:
                    loss_tensor = torch.tensor(
                        [avg_loss_value, avg_eval_distill_loss],
                        dtype=torch.float32, device=my_device,
                    )
                    pipeline_send(loss_tensor, dst_stage=0, tag=9999)
                elif my_stage == 0:
                    loss_tensor = torch.empty(2, dtype=torch.float32, device=my_device)
                    pipeline_recv(loss_tensor, src_stage=norm_stage, tag=9999)
                    avg_loss_value = loss_tensor[0].item()
                    avg_eval_distill_loss = loss_tensor[1].item()

            # Clean up
            del all_hidden_states, all_memory_states
            del all_step1_anchors, all_step4_anchors
            del distill_teacher_outputs, distill_student_outputs

        # Clear model caches after each eval batch
        model.invalidate_input_cache()

        # Accumulate regu_sq_norm from eval forward
        if model.detach_state is not None:
            # _eval_sq_norms is per-mb sq_norms (local to this stage)
            _eval_regu_local = sum(_eval_sq_norms) / len(_eval_sq_norms) if _eval_sq_norms else 0.0
            # Sync across stages for full sq_norm
            node_group = parallel_cfg.get("node_process_group")
            if node_group is not None:
                _sq_t = torch.tensor(_eval_sq_norms, dtype=torch.float64, device=my_device)
                dist.all_reduce(_sq_t, op=dist.ReduceOp.SUM, group=node_group)
                _eval_sq_norms_synced = _sq_t.tolist()
            else:
                _eval_sq_norms_synced = _eval_sq_norms
            _eval_running_regu_sq_norm += sum(_eval_sq_norms_synced) / len(_eval_sq_norms_synced) if _eval_sq_norms_synced else 0.0

            # Step+1 for each sample position
            # _eval_sq_norms_synced is per-micro-batch; expand to per-sample
            _eval_sq_norms_per_sample = []
            for _sq in _eval_sq_norms_synced:
                _eval_sq_norms_per_sample.extend([_sq] * local_micro_batch_size)
            for _si in range(len(_eval_sq_norms_per_sample)):
                model.detach_state.update_steps(_si)
            # Record stats (after step+1, before threshold reset)
            _eval_reset_ratio, _eval_mean_upd = model.detach_state.get_reset_stats()
            _eval_running_reset_ratio += _eval_reset_ratio
            _eval_running_mean_update_step += _eval_mean_upd
            # Threshold reset
            model.detach_state.set_last_sq_norms(_eval_sq_norms_per_sample)
            for _si in range(len(_eval_sq_norms_per_sample)):
                model.detach_state.maybe_reset_slice(_si)

        total_loss += avg_loss_value
        total_distill_loss += avg_eval_distill_loss
        num_batches += 1

    # Exit eval_context: restore training detach_state, release eval state
    if _ds_ctx is not None:
        _ds_ctx.__exit__(None, None, None)

    model.train()

    if num_batches == 0:
        if is_main_process_per_node():
            logger.warning(f"  [Eval Step {global_step}] No validation batches processed")
        return 0.0, 0.0, 0.0

    val_avg_loss = total_loss / num_batches
    val_avg_distill_loss = total_distill_loss / num_batches

    # ================================================================
    # Aggregate evaluation results across nodes
    #
    # Step 1: Each node's stage 0 already has the per-node val_avg_loss
    #         (loss was transferred from norm_stage → stage 0 inside the
    #         eval loop above).
    # Step 2: All stage-0 processes across nodes participate in
    #         gather_object via dp_process_group to collect per-node
    #         losses to node 0's stage 0.
    # Step 3: Node 0's stage 0 computes global average and writes to
    #         evaluation.log.
    #
    # IMPORTANT: gather_object is a collective op — ALL members of the
    # dp_process_group for stage 0 must call it.  Non-stage-0 processes
    # do NOT participate (they have no dp_process_group membership for
    # stage 0's group, and their loss is already forwarded to stage 0).
    # ================================================================
    dp_group = parallel_cfg.get("dp_process_group")
    dp_size = parallel_cfg["data_parallel_size"]
    my_stage = parallel_cfg["stage"]

    global_val_loss = 0.0
    global_val_ppl = 0.0
    global_val_distill_loss = 0.0

    if my_stage == 0:
        # Stage 0 on every node has the per-node val loss.
        # Gather across DP replicas (= across nodes) to node 0.
        if dp_group is not None and dp_size > 1:
            local_metrics = {"val_loss": val_avg_loss, "val_distill_loss": val_avg_distill_loss, "num_batches": num_batches, "node_rank": node_rank,
                             "val_regu_sq_norm": _eval_running_regu_sq_norm / num_batches if num_batches > 0 else 0.0,
                             "val_reset_ratio": _eval_running_reset_ratio / num_batches if num_batches > 0 else 0.0,
                             "val_mean_update_step": _eval_running_mean_update_step / num_batches if num_batches > 0 else 0.0,
                             "val_repo_reset_ratio": _eval_repo_reset_count / num_batches if num_batches > 0 else 0.0}
            dst_global = dist.get_global_rank(dp_group, 0)
            if is_main_process():  # node 0, stage 0
                gathered_metrics = [None] * dp_size
            else:
                gathered_metrics = None
            dist.gather_object(
                local_metrics, gathered_metrics,
                dst=dst_global, group=dp_group,
            )
        else:
            # Single node
            gathered_metrics = [{"val_loss": val_avg_loss, "val_distill_loss": val_avg_distill_loss, "num_batches": num_batches, "node_rank": 0,
                                 "val_regu_sq_norm": _eval_running_regu_sq_norm / num_batches if num_batches > 0 else 0.0,
                                 "val_reset_ratio": _eval_running_reset_ratio / num_batches if num_batches > 0 else 0.0,
                                 "val_mean_update_step": _eval_running_mean_update_step / num_batches if num_batches > 0 else 0.0,
                                 "val_repo_reset_ratio": _eval_repo_reset_count / num_batches if num_batches > 0 else 0.0}]

        # Node 0, stage 0: compute global average and log
        if is_main_process() and gathered_metrics is not None:
            all_val_losses = [m["val_loss"] for m in gathered_metrics]
            all_val_distill_losses = [m["val_distill_loss"] for m in gathered_metrics]
            global_val_loss = sum(all_val_losses) / len(all_val_losses)
            global_val_distill_loss = sum(all_val_distill_losses) / len(all_val_distill_losses)
            global_val_ppl = math.exp(global_val_loss) if global_val_loss < 20 else float("inf")
            # Total loss = CE + coefficient * distill (coefficient applied in distill_loss_fn)
            _distill_coeff = distill_loss_fn.coefficient if distill_loss_fn is not None else 1.0
            global_val_total_loss = global_val_loss + _distill_coeff * global_val_distill_loss
            global_val_total_ppl = math.exp(global_val_total_loss) if global_val_total_loss < 20 else float("inf")

            # Compute elapsed and ETA for eval output
            eval_elapsed = time.time() - t_start if t_start > 0 else 0.0
            eval_steps_remaining = max_steps - global_step
            eval_eta = ema_time_per_step * eval_steps_remaining if ema_time_per_step > 0 else 0.0
            eval_duration = time.time() - _eval_start_time if _eval_start_time > 0 else 0.0

            # Log to main logger (stdout / node_0.log)
            # val_loss/val_ppl = CE loss (backward compatible with pre-distill runs)
            _eval_distill_suffix = ""
            if global_val_distill_loss > 0:
                _eval_distill_suffix = (
                    f", val_distill_loss={global_val_distill_loss:.4f}, "
                    f"val_total_loss={global_val_total_loss:.4f}, "
                    f"val_total_ppl={global_val_total_ppl:.2f}"
                )
            logger.info(
                f"  [Eval Step {global_step}] "
                f"val_loss={global_val_loss:.4f}, val_ppl={global_val_ppl:.2f}"
                f"{_eval_distill_suffix}, "
                f"val_batches={num_batches}, "
                f"eval_time={format_duration(eval_duration)}, "
                f"elapsed={format_duration(eval_elapsed)}, eta={format_duration(eval_eta)}"
            )

            # Log to dedicated evaluation.log
            _eval_logger = logging.getLogger("debug.evaluation")
            _eval_logger.info(
                f"[Eval Step {global_step}] "
                f"val_loss={global_val_loss:.6f}, val_ppl={global_val_ppl:.4f}, "
                f"val_distill_loss={global_val_distill_loss:.6f}, "
                f"val_total_loss={global_val_total_loss:.6f}, val_total_ppl={global_val_total_ppl:.4f}, "
                f"val_batches_per_node={num_batches}, "
                f"eval_time={format_duration(eval_duration)}, "
                f"elapsed={format_duration(eval_elapsed)}, eta={format_duration(eval_eta)}"
            )

            if use_wandb:
                _eval_wall_time = time.time() - t_start if t_start > 0 else 0.0
                _eval_wandb_metrics = {
                    "wall_time": _eval_wall_time,
                    "eval/loss": global_val_loss,
                    "eval/ppl": global_val_ppl,
                    "eval/distill_loss": global_val_distill_loss,
                    "eval/total_loss": global_val_total_loss,
                    "eval/total_ppl": global_val_total_ppl,
                }
                # DetachState metrics for eval (averaged across DP replicas)
                all_val_regu = [m.get("val_regu_sq_norm", 0.0) for m in gathered_metrics]
                all_val_reset = [m.get("val_reset_ratio", 0.0) for m in gathered_metrics]
                all_val_upd = [m.get("val_mean_update_step", 0.0) for m in gathered_metrics]
                all_val_repo_reset = [m.get("val_repo_reset_ratio", 0.0) for m in gathered_metrics]
                _eval_wandb_metrics["eval/regu_sq_norm"] = sum(all_val_regu) / len(all_val_regu)
                _eval_wandb_metrics["eval/reset_ratio"] = sum(all_val_reset) / len(all_val_reset)
                _eval_wandb_metrics["eval/mean_update_step"] = sum(all_val_upd) / len(all_val_upd)
                _eval_wandb_metrics["eval/repo_reset_ratio"] = sum(all_val_repo_reset) / len(all_val_repo_reset)
                wandb.log(_eval_wandb_metrics, step=global_step)

    return global_val_loss, global_val_ppl, global_val_distill_loss


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PP NCCL P2P warmup
# ---------------------------------------------------------------------------

def nccl_p2p_warmup(model, parallel_cfg, my_device):
    """
    Warmup NCCL P2P communicators for pipeline parallelism.

    NCCL uses lazy initialization for P2P communicators. The first P2P
    operation on a ProcessGroup requires ALL ranks to participate
    simultaneously. In our multi-phase pipeline, some stages (e.g. GPU7)
    may not participate in Phase A P2P ops, causing NCCL init to hang.
    Solution: do a dummy send/recv between all pipeline stage pairs that
    will communicate during training, so all communicators are pre-initialized.
    """
    if is_main_process_per_node():
        logger.info("  [INIT] Warming up NCCL P2P communicators...")
    my_stage = parallel_cfg["stage"]
    total_stages = parallel_cfg["total_stages"]
    warmup_tensor = torch.zeros(1, device=my_device)

    # Warmup all P2P pairs used in training.
    # Strategy: for each pair (src, dst), src sends and dst recvs simultaneously.
    # We iterate over all pairs that will communicate during training.
    # To avoid deadlock, we process pairs in a specific order where
    # each rank does at most one blocking operation at a time.

    if total_stages > 1:
        # 1. Adjacent forward pairs: stage i → stage i+1 (used in LLM pipeline)
        for i in range(total_stages - 1):
            if my_stage == i:
                pipeline_send(warmup_tensor, dst_stage=i + 1)
            elif my_stage == i + 1:
                pipeline_recv(warmup_tensor, src_stage=i)
        # 2. Adjacent backward pairs: stage i+1 → stage i
        for i in range(total_stages - 1):
            if my_stage == i + 1:
                pipeline_send(warmup_tensor, dst_stage=i)
            elif my_stage == i:
                pipeline_recv(warmup_tensor, src_stage=i + 1)
        # 3. Non-adjacent pairs: mem_gather (LLM stages → GPU6)
        mem_gather_target = model._mem_gather_target_stage
        for src_stage in model._mem_gather_stages:
            if src_stage == mem_gather_target:
                continue
            # Skip adjacent pairs (already warmed up above)
            if abs(src_stage - mem_gather_target) == 1:
                continue
            if my_stage == src_stage:
                pipeline_send(warmup_tensor, dst_stage=mem_gather_target)
            elif my_stage == mem_gather_target:
                pipeline_recv(warmup_tensor, src_stage=src_stage)
        # 4. Non-adjacent pairs: mem_gather grad (GPU6 → LLM stages)
        for src_stage in model._mem_gather_stages:
            if src_stage == mem_gather_target:
                continue
            if abs(src_stage - mem_gather_target) == 1:
                continue
            if my_stage == mem_gather_target:
                pipeline_send(warmup_tensor, dst_stage=src_stage)
            elif my_stage == src_stage:
                pipeline_recv(warmup_tensor, src_stage=mem_gather_target)
        # 5. Non-adjacent pairs: lora_scatter (GPU7 → LLM stages)
        m2p_norm_stage = model.hypernetwork._m2p_norm_stage
        for dst_stage in model._mem_gather_stages:
            if dst_stage == m2p_norm_stage:
                continue
            if abs(dst_stage - m2p_norm_stage) == 1:
                continue
            if my_stage == m2p_norm_stage:
                pipeline_send(warmup_tensor, dst_stage=dst_stage)
            elif my_stage == dst_stage:
                pipeline_recv(warmup_tensor, src_stage=m2p_norm_stage)
        # 6. Non-adjacent pairs: lora_scatter grad (LLM stages → GPU7)
        for dst_stage in model._mem_gather_stages:
            if dst_stage == m2p_norm_stage:
                continue
            if abs(dst_stage - m2p_norm_stage) == 1:
                continue
            if my_stage == dst_stage:
                pipeline_send(warmup_tensor, dst_stage=m2p_norm_stage)
            elif my_stage == m2p_norm_stage:
                pipeline_recv(warmup_tensor, src_stage=dst_stage)
        # 7. Labels transfer: embed_stage → norm_stage
        embed_stage = model._embed_stage
        norm_stage = model._norm_stage
        if embed_stage != norm_stage and abs(embed_stage - norm_stage) > 1:
            if my_stage == embed_stage:
                pipeline_send(warmup_tensor, dst_stage=norm_stage)
            elif my_stage == norm_stage:
                pipeline_recv(warmup_tensor, src_stage=embed_stage)
            # Reverse for loss transfer
            if my_stage == norm_stage:
                pipeline_send(warmup_tensor, dst_stage=embed_stage)
            elif my_stage == embed_stage:
                pipeline_recv(warmup_tensor, src_stage=norm_stage)

    del warmup_tensor
    barrier()
    if is_main_process_per_node():
        logger.info("  [INIT] NCCL P2P warmup complete.")
