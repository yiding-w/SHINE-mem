#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meta Training Script with Pipeline and Data Parallelism

Training flow per iteration:
    1. context_ids → LLM (no grad, use_mem_token=True) → memory_states
    2. memory_states → Hypernetwork (with grad) → loradict
    3. conversation_ids + loradict → LLM (with grad) → hidden_states
    4. hidden_states → lm_head → logits + labels → cross-entropy loss
    5. pipeline_backward(loss) → optimizer.step()

Usage:
    torchrun --nproc_per_node=8 --nnodes=<num_nodes> --node_rank=<rank> \
        --master_addr=<master_ip> --master_port=<port> \
        meta_train.py parallel.pipeline_parallel_size=4
"""

import warnings
import torch, os, math, time, logging, sys, importlib
import torch._functorch.config
from contextlib import contextmanager, nullcontext
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

# Suppress harmless NCCL P2P communicator warning from pipeline parallelism
warnings.filterwarnings("ignore", message="An unbatched P2P op")

# Utility imports
from utils.myparallel import (
    init_distributed, cleanup_distributed,
    setup_pipeline_parallel, get_pipeline_config,
    get_rank, get_world_size, is_main_process,
    is_main_process_per_node, is_node0,
    is_first_stage, is_last_stage,
    pipeline_send, pipeline_recv,
    sync_gradients_across_dp,
    barrier,
)
from utils.mygpu import gpu_stats, all_gpu_stats, cross_node_bandwidth_test, _report_peak_memory
from hypernetwork.model_hypernetwork import ModelHypernetwork
from utils.mydata import (
    PipelineDataLoader,
    resolve_pad_token_id, resolve_special_token_id,
    create_dataset_from_config,
)
from utils.myoptimizer import create_optimizer_and_scheduler, audit_parameters
from utils.mytraining_debug import (
    DebugSchedule,
    log_training_detail,
    broadcast_trainable_params_from_dp_rank0,
    check_dp_param_consistency,
    compute_grad_norms,
    compute_post_clip_grad_norm,
    compute_loss_spike_metrics,
    NanInfTracker,
    compute_param_norms,
    compute_generated_lora_norms,
    log_nograd_loradict_check,
)
from utils.mylog import format_duration, setup_debug_loggers, flush_debug_loggers
import wandb

logger = logging.getLogger(__name__)


from utils.mysaveload import (
    get_checkpoint_dir,
    get_step_checkpoint_dir,
    list_checkpoints,
    get_latest_checkpoint,
    resolve_forever_save_steps,
    save_checkpoint,
    load_checkpoint,
    build_checkpoint_run_name,
    get_pretrain_final_checkpoint,
    get_pretrain_annealing_final_checkpoint,
    save_final_checkpoint,
    load_model_only,
    save_detach_state_all_nodes,
)


# ---------------------------------------------------------------------------
# Sentinel tensor: only exposes .shape, raises on any real access
# ---------------------------------------------------------------------------

class _ShapeOnlyTensor:
    """
    A lightweight proxy that carries only ``shape`` (and derived properties
    like ``size()``).  Any attempt to use it as a real tensor (indexing,
    arithmetic, ``.to()``, ``.view()``, etc.) raises ``RuntimeError``
    immediately, making it impossible to silently use dummy data on
    non-data-loading pipeline stages.
    """

    def __init__(self, shape: tuple, *, name: str = "tensor"):
        # Use object.__setattr__ to bypass our __setattr__ guard
        object.__setattr__(self, "_shape", torch.Size(shape))
        object.__setattr__(self, "_name", name)

    # --- Allowed attributes ---------------------------------------------------
    @property
    def shape(self) -> torch.Size:
        return self._shape

    def size(self, dim=None):
        if dim is None:
            return self._shape
        return self._shape[dim]

    def dim(self):
        return len(self._shape)

    def __repr__(self):
        return f"_ShapeOnlyTensor(name={self._name!r}, shape={tuple(self._shape)})"

    # --- Everything else is forbidden -----------------------------------------
    def __getattr__(self, name):
        raise RuntimeError(
            f"Attempted to access attribute '{name}' on a _ShapeOnlyTensor "
            f"(name={self._name!r}, shape={tuple(self._shape)}). "
            f"This tensor is a shape-only sentinel on a non-data-loading "
            f"pipeline stage.  Real data must be explicitly transferred "
            f"via pipeline_send/pipeline_recv before use."
        )

    def __getitem__(self, key):
        raise RuntimeError(
            f"Attempted to index _ShapeOnlyTensor (name={self._name!r}). "
            f"Real data must be explicitly transferred before use."
        )

    def __setattr__(self, name, value):
        raise RuntimeError(
            f"Attempted to set attribute '{name}' on _ShapeOnlyTensor "
            f"(name={self._name!r}).  This is a read-only sentinel."
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise RuntimeError(
            f"Attempted to use _ShapeOnlyTensor (name={self._name!r}) in "
            f"torch operation {func.__name__}.  Real data must be "
            f"explicitly transferred before use."
        )


# ---------------------------------------------------------------------------
# Helper: verify SDPA Flash Attention is working
# ---------------------------------------------------------------------------

def _verify_sdpa_flash_attention(model, my_device):
    """
    Verify that SDPA Flash Attention backend is properly triggered.

    Checks:
      1. Model dtype is bf16/fp16 (required by Flash kernel)
      2. config._attn_implementation is 'sdpa'
      3. create_causal_mask returns None when attention_mask=None
         (which causes sdpa_attention_forward to set is_causal=True)
      4. Flash Attention backend actually works on this GPU
    """
    if not is_main_process_per_node():
        return

    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    logger.info("\n" + "=" * 70)
    logger.info("  SDPA Flash Attention Verification")
    logger.info("=" * 70)

    all_ok = True

    # 1. Check model dtype
    model_dtype = model._dtype
    dtype_ok = model_dtype in (torch.bfloat16, torch.float16)
    status = "✓" if dtype_ok else "✗"
    logger.info(f"  {status} Model dtype: {model_dtype} "
                f"({'OK - Flash compatible' if dtype_ok else 'FAIL - Flash requires bf16/fp16'})")
    if not dtype_ok:
        all_ok = False

    # 2. Check _attn_implementation
    text_config = model._text_config
    attn_impl = getattr(text_config, '_attn_implementation', 'unknown')
    # transformers sets this during from_pretrained; 'sdpa' is the default
    # Note: it may show as 'sdpa' or None (None defaults to sdpa at runtime)
    impl_ok = attn_impl in ('sdpa', None)
    status = "✓" if impl_ok else "✗"
    logger.info(f"  {status} config._attn_implementation: '{attn_impl}' "
                f"({'OK - SDPA enabled' if impl_ok else 'FAIL - expected sdpa'})")
    if not impl_ok:
        all_ok = False

    # 3. Check create_causal_mask returns None with attention_mask=None
    try:
        from transformers.masking_utils import create_causal_mask
        hidden_size = text_config.hidden_size
        fake_embeds = torch.empty(1, 64, 1, device=my_device, dtype=model_dtype)
        causal_mask = create_causal_mask(
            config=text_config,
            inputs_embeds=fake_embeds,
            attention_mask=None,
            past_key_values=None,
        )
        mask_ok = causal_mask is None
        status = "✓" if mask_ok else "✗"
        logger.info(f"  {status} create_causal_mask(attention_mask=None) returns: "
                    f"{'None (triggers is_causal=True → Flash)' if mask_ok else f'shape={causal_mask.shape} (BLOCKS Flash!)'}")
        if not mask_ok:
            all_ok = False
    except Exception as e:
        logger.info(f"  ⚠ Could not verify create_causal_mask: {e}")
        all_ok = False

    # 4. Test Flash Attention backend directly
    if torch.cuda.is_available():
        try:
            B, H, S, D = 1, 8, 128, 64
            Q = torch.randn(B, H, S, D, device=my_device, dtype=model_dtype)
            K = torch.randn(B, H, S, D, device=my_device, dtype=model_dtype)
            V = torch.randn(B, H, S, D, device=my_device, dtype=model_dtype)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                _ = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
            logger.info(f"  ✓ Flash Attention backend: WORKS "
                        f"(bf16 + no mask + is_causal=True on {torch.cuda.get_device_name(my_device)})")
        except Exception as e:
            logger.info(f"  ✗ Flash Attention backend: FAILED ({type(e).__name__}: {e})")
            all_ok = False
    else:
        logger.info(f"  ⚠ No CUDA device available, skipping Flash backend test")
        all_ok = False

    # Summary
    if all_ok:
        logger.info(f"  ★ ALL CHECKS PASSED: SDPA Flash Attention will be used ★")
    else:
        logger.info(f"  ⚠ SOME CHECKS FAILED: Flash Attention may NOT be triggered")
    logger.info("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Helper: sync metalora gradients across data-parallel replicas
# ---------------------------------------------------------------------------

def _sync_metalora_gradients(model, my_device, parallel_cfg):
    """Sync metalora tensor gradients across DP replicas via all_reduce AVG."""
    from utils.myloradict import collect_loradict_tensors

    group = parallel_cfg.get("dp_process_group")
    if group is None:
        return
    dp_size = dist.get_world_size(group)
    if dp_size <= 1:
        return

    metalora_tensors = collect_loradict_tensors(model.metalora)
    for t in metalora_tensors:
        if t.requires_grad and t.grad is not None and t.device == my_device:
            dist.all_reduce(t.grad, op=dist.ReduceOp.AVG, group=group)

# ---------------------------------------------------------------------------
# Parallel-mode-specific functions (train_step, run_evaluation, nccl_p2p_warmup)
# are loaded conditionally based on cfg.parallel.mode in main().
# Default to PP for backward compatibility.
# These will be set by _load_parallel_mode_functions() at runtime:
# ---------------------------------------------------------------------------
train_step = None
run_evaluation = None
_nccl_p2p_warmup = None


def _load_parallel_mode_functions(parallel_mode: str):
    """
    Load the correct train_step, run_evaluation, and nccl_p2p_warmup
    implementations based on the parallel mode ('pp' or 'tp').
    """
    global train_step, run_evaluation, _nccl_p2p_warmup

    if parallel_mode == "tp":
        from meta_train_tp import train_step as _ts, run_evaluation as _re, nccl_p2p_warmup as _warmup
    else:
        # Default: pipeline parallel
        from meta_train_pp import train_step as _ts, run_evaluation as _re, nccl_p2p_warmup as _warmup

    train_step = _ts
    run_evaluation = _re
    _nccl_p2p_warmup = _warmup



def do_train_step(
    data_iter,
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    my_device,
    parallel_cfg,
    make_sdpa_ctx,
    epoch: int,
    global_step: int,
    micro_step: int,
    running_loss: float,
    running_distill_loss: float,
    running_regu_sq_norm: float,
    running_reset_ratio: float,
    running_mean_update_step: float,
    t_start: float,
    grad_accum_steps: int,
    gradient_clipping: float,
    logging_steps: int,
    max_steps: int,
    local_batch_size: int,
    local_micro_batch_size: int,
    vocab_size: int,
    context_seq_len: int,
    conv_seq_len: int,
    num_mem_token: int,
    save_sched: DebugSchedule = None,
    forever_save_steps: set = None,
    save_total_limit: int = 5,
    run_name: str = "",
    log_peak_memory_sched: DebugSchedule = None,
    log_train_detail_sched: DebugSchedule = None,
    log_train_detail_ppl_threshold: float = -1,
    use_wandb: bool = False,
    node_rank: int = 0,
    epoch_loss_sum: float = 0.0,
    epoch_steps: int = 0,
    detail_log_accumulator: Optional[List[Dict[str, Any]]] = None,
    detail_log_accumulator_ppl: Optional[List[Dict[str, Any]]] = None,
    tokenizer=None,
    pad_token_id: int = 0,
    check_consistency_sched: DebugSchedule = None,
    check_nograd_loradict_sched: DebugSchedule = None,
    ema_time_per_step: float = 0.0,
    step_start_time: float = 0.0,
    ema_alpha: float = 0.1,
    step_context_tokens: int = 0,
    step_conv_total_tokens: int = 0,
    step_conv_valid_tokens: int = 0,
    total_context_tokens: int = 0,
    total_conv_total_tokens: int = 0,
    total_conv_valid_tokens: int = 0,
    wandb_run_id: Optional[str] = None,
    # --- P0 Monitor parameters ---
    grad_norm_sched: DebugSchedule = None,
    param_norm_sched: DebugSchedule = None,
    gen_lora_norm_sched: DebugSchedule = None,
    monitor_loss_spike: bool = False,
    monitor_nan_inf: bool = False,
    nan_inf_tracker: Optional[Any] = None,
    nan_inf_stop_steps: int = -1,
    loss_running_avg: float = 0.0,
    loss_ema_alpha: float = 0.1,
    # --- Distillation parameters ---
    distill_loss_fn=None,
    distill_micro_batch_size: int = 1,
    # --- Config tracking for checkpoint ---
    config_selections: Optional[dict] = None,
    launch_cmd: Optional[str] = None,
):
    """
    Execute one training iteration inside the training loop.

    This includes: fetching data, splitting into micro-batches, running
    multi micro-batch pipeline forward+backward, and (when grad_accum_steps
    iterations have been processed) optimizer step + logging.

    The data for one iteration is local_batch_size samples, which are split
    into (local_batch_size / local_micro_batch_size) micro-batches. All
    micro-batches are processed together in a single pipeline_forward_train_multi_mb
    call, with inter-stage overlap to reduce bubble time.

    Returns:
        (global_step, micro_step, running_loss, stop_epoch, epoch_loss_sum,
         epoch_steps, ema_time_per_step, step_start_time) where stop_epoch
         is True if the data iterator is exhausted (StopIteration).
    """
    # --- Get next batch ---
    try:
        batch_data = next(data_iter)
    except StopIteration:
        return global_step, micro_step, running_loss, running_distill_loss, running_regu_sq_norm, running_reset_ratio, running_mean_update_step, True, epoch_loss_sum, epoch_steps, ema_time_per_step, step_start_time, step_context_tokens, step_conv_total_tokens, step_conv_valid_tokens, total_context_tokens, total_conv_total_tokens, total_conv_valid_tokens, loss_running_avg  # end of epoch

    # --- Extract tensors and split into micro-batch lists ---
    num_micro_batches = local_batch_size // local_micro_batch_size

    if train_loader.is_first_stage and batch_data["micro_batches"]:
        mbs = batch_data["micro_batches"]
        # mbs is already split into micro-batches by PipelineDataLoader
        context_ids_list = [mb["context_ids"].to(my_device) for mb in mbs]
        conversation_ids_list = [mb["conversation_ids"].to(my_device) for mb in mbs]
        labels_list = [mb["labels"].to(my_device) for mb in mbs]
        ctx_lengths_list = [mb["context_lengths"].to(my_device) for mb in mbs]
        # Extract extra_info (non-tensor metadata, e.g. repo name)
        extra_info_list = [mb.get("extra_info", None) for mb in mbs]
        # Distillation data: extract and split into distill micro-batches
        distill_list = [
            {k: v.to(my_device) if isinstance(v, torch.Tensor) else v
             for k, v in mb["distill"].items()}
            if mb.get("distill") is not None else None
            for mb in mbs
        ]
        # Build distill micro-batch lists for train_step
        if distill_loss_fn is not None and any(d is not None for d in distill_list):
            # Concatenate all distill micro-batch conversation_ids and labels,
            # then re-split by distill_micro_batch_size
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
        # Non-data-loading stages: use shape-only sentinels per micro-batch
        context_ids_list = [
            _ShapeOnlyTensor(
                (local_micro_batch_size, context_seq_len + num_mem_token),
                name=f"context_ids_mb{i}",
            ) for i in range(num_micro_batches)
        ]
        conversation_ids_list = [
            _ShapeOnlyTensor(
                (local_micro_batch_size, conv_seq_len),
                name=f"conversation_ids_mb{i}",
            ) for i in range(num_micro_batches)
        ]
        labels_list = [None] * num_micro_batches
        ctx_lengths_list = [
            torch.full((local_micro_batch_size,), context_seq_len, dtype=torch.long, device=my_device)
            for _ in range(num_micro_batches)
        ]
        # Non-first stages don't have extra_info
        extra_info_list = [None] * num_micro_batches
        # Non-first stages must also participate in distill pipeline communication
        # (Phase A' and C' require send/recv on all pipeline stages)
        if distill_loss_fn is not None:
            num_distill_mbs = local_batch_size // distill_micro_batch_size
            distill_conversation_ids_list = [
                _ShapeOnlyTensor(
                    (distill_micro_batch_size, conv_seq_len),
                    name=f"distill_conv_ids_mb{i}",
                ) for i in range(num_distill_mbs)
            ]
            distill_labels_list = [None] * num_distill_mbs
        else:
            distill_conversation_ids_list = None
            distill_labels_list = None

    # --- Determine if this iteration needs detail logging ---
    upcoming_step = global_step + 1
    need_detail = (
        (log_train_detail_sched is not None
         and log_train_detail_sched.should_run(upcoming_step))
        or log_train_detail_ppl_threshold > 0
    )

    # --- Repo-change reset: check each micro-batch's repo before forward ---
    if model.detach_state is not None and train_loader.is_first_stage:
        num_micro_batches_local = len(extra_info_list)
        for _mb_i in range(num_micro_batches_local):
            _ei = extra_info_list[_mb_i]
            if _ei is not None and isinstance(_ei, list) and len(_ei) > 0:
                _cur_repo = _ei[0].get("repo") if isinstance(_ei[0], dict) else None
                if _cur_repo is not None:
                    if (hasattr(model, '_prev_repo_per_mb') and
                            _mb_i < len(model._prev_repo_per_mb) and
                            model._prev_repo_per_mb[_mb_i] is not None and
                            _cur_repo != model._prev_repo_per_mb[_mb_i]):
                        # Reset all samples in this micro-batch's wdict slice
                        _sample_start = _mb_i * local_micro_batch_size
                        for _si in range(_sample_start, _sample_start + local_micro_batch_size):
                            model.detach_state.reset_slice(_si)
                        model._running_repo_reset_count += 1
                    model._prev_repo_per_mb[_mb_i] = _cur_repo

    # --- Forward + backward (all micro-batches in one pipeline call) ---
    batch_id = f"e{epoch}_s{global_step}_m{micro_step}"
    try:
        with make_sdpa_ctx():
            result = train_step(
                model=model,
                context_ids_list=context_ids_list,
                context_lengths_list=ctx_lengths_list,
                conversation_ids_list=conversation_ids_list,
                labels_list=labels_list,
                micro_batch_size=local_micro_batch_size,
                batch_id=batch_id,
                my_device=my_device,
                norm_stage=model._norm_stage,
                log_detail=need_detail,
                grad_accum_steps=grad_accum_steps,
                distill_conversation_ids_list=distill_conversation_ids_list,
                distill_labels_list=distill_labels_list,
                distill_micro_batch_size=distill_micro_batch_size,
                distill_loss_fn=distill_loss_fn,
            )
    except Exception as e:
        import traceback, sys
        err_msg = (
            f"\n{'='*80}\n"
            f"[FATAL] train_step FAILED on stage={parallel_cfg['stage']}, "
            f"batch_id={batch_id}, micro_batch_size={local_micro_batch_size}, "
            f"num_micro_batches={local_batch_size // local_micro_batch_size}\n"
            f"{'='*80}\n"
            f"{traceback.format_exc()}"
            f"{'='*80}\n"
        )
        sys.stderr.write(err_msg)
        sys.stderr.flush()
        # Also write to logger in case stderr is redirected
        import builtins
        builtins.print(err_msg, flush=True)
        raise

    if need_detail:
        loss_val, per_token_loss, distill_loss_val, regu_sq_norm_local, per_mb_sq_norms_local = result
    else:
        loss_val, distill_loss_val, regu_sq_norm_local, per_mb_sq_norms_local = result
        per_token_loss = None

    # Accumulate detail info for later printing (only on embed stage / stage 0)
    if need_detail and detail_log_accumulator is not None and train_loader.is_first_stage:
        # Concatenate micro-batch lists back into full batch tensors for logging
        all_ctx = torch.cat([c.cpu() for c in context_ids_list if isinstance(c, torch.Tensor)], dim=0) \
            if isinstance(context_ids_list[0], torch.Tensor) else None
        all_conv = torch.cat([c.cpu() for c in conversation_ids_list if isinstance(c, torch.Tensor)], dim=0) \
            if isinstance(conversation_ids_list[0], torch.Tensor) else None
        all_labels = torch.cat([l.cpu() for l in labels_list if l is not None], dim=0) \
            if labels_list[0] is not None else None
        all_ctx_len = torch.cat([c.cpu() for c in ctx_lengths_list], dim=0)
        # Concatenate distill data if available
        all_distill = None
        if distill_list[0] is not None:
            all_distill = {
                "conversation_ids": torch.cat(
                    [d["conversation_ids"].cpu() for d in distill_list if d is not None], dim=0
                ),
                "labels": torch.cat(
                    [d["labels"].cpu() for d in distill_list if d is not None], dim=0
                ),
            }
        detail_log_accumulator.append({
            "context_ids": all_ctx,
            "conversation_ids": all_conv,
            "labels": all_labels,
            "context_lengths": all_ctx_len,
            "per_token_loss": per_token_loss.cpu() if per_token_loss is not None else None,
            "distill": all_distill,
        })

    # Accumulate detail info for ppl-threshold-triggered logging
    if need_detail and detail_log_accumulator_ppl is not None and train_loader.is_first_stage:
        if log_train_detail_sched is None or not log_train_detail_sched.should_run(upcoming_step):
            # Only append to ppl accumulator when NOT already appended to the regular one
            all_ctx_ppl = torch.cat([c.cpu() for c in context_ids_list if isinstance(c, torch.Tensor)], dim=0) \
                if isinstance(context_ids_list[0], torch.Tensor) else None
            all_conv_ppl = torch.cat([c.cpu() for c in conversation_ids_list if isinstance(c, torch.Tensor)], dim=0) \
                if isinstance(conversation_ids_list[0], torch.Tensor) else None
            all_labels_ppl = torch.cat([l.cpu() for l in labels_list if l is not None], dim=0) \
                if labels_list[0] is not None else None
            all_ctx_len_ppl = torch.cat([c.cpu() for c in ctx_lengths_list], dim=0)
            all_distill_ppl = None
            if distill_list[0] is not None:
                all_distill_ppl = {
                    "conversation_ids": torch.cat(
                        [d["conversation_ids"].cpu() for d in distill_list if d is not None], dim=0
                    ),
                    "labels": torch.cat(
                        [d["labels"].cpu() for d in distill_list if d is not None], dim=0
                    ),
                }
            detail_log_accumulator_ppl.append({
                "context_ids": all_ctx_ppl,
                "conversation_ids": all_conv_ppl,
                "labels": all_labels_ppl,
                "context_lengths": all_ctx_len_ppl,
                "per_token_loss": per_token_loss.cpu() if per_token_loss is not None else None,
                "distill": all_distill_ppl,
            })

    # Count tokens for this micro-batch
    if train_loader.is_first_stage and labels_list[0] is not None:
        # Context tokens: sum of context_lengths (excludes memory tokens and padding)
        for ctx_len in ctx_lengths_list:
            if isinstance(ctx_len, torch.Tensor):
                step_context_tokens += int(ctx_len.sum().item())
        # Conversation total tokens: non-padding tokens in conversation_ids
        for conv in conversation_ids_list:
            if isinstance(conv, torch.Tensor):
                step_conv_total_tokens += int((conv != pad_token_id).sum().item())
        # Conversation valid tokens: tokens with labels != -100 (participate in loss)
        for lbl in labels_list:
            if isinstance(lbl, torch.Tensor):
                step_conv_valid_tokens += int((lbl != -100).sum().item())

    # --- P0 Monitor: NaN/Inf detection (before accumulating loss) ---
    # loss_val is only meaningful on stage 0 (received from norm_stage).
    # On other stages it is 0.0. The actual skip decision is broadcast
    # inside the optimizer step block (after DP sync) to all processes.
    _local_nan_inf = False
    _prev_consecutive_bad = nan_inf_tracker.consecutive_bad if nan_inf_tracker is not None else 0
    if monitor_nan_inf and nan_inf_tracker is not None:
        my_stage = parallel_cfg["stage"]
        if my_stage == 0:
            _local_nan_inf = nan_inf_tracker.check_and_record(loss_val, global_step + 1)

    running_loss += loss_val
    running_distill_loss += distill_loss_val
    # Sum regu_sq_norm across all pipeline stages within the node.
    # Each stage computes only its own layers' contribution; all_reduce SUM
    # gives the node-level total per micro-batch position.
    # All stages receive the synced per-mb norms for threshold-based reset.
    node_group = parallel_cfg.get("node_process_group")
    if node_group is not None:
        # Sync per-mb sq_norms: each element is SUM across stages
        _per_mb_tensor = torch.tensor(per_mb_sq_norms_local, dtype=torch.float64, device=my_device)
        dist.all_reduce(_per_mb_tensor, op=dist.ReduceOp.SUM, group=node_group)
        per_mb_sq_norms_synced = _per_mb_tensor.tolist()
        # Node-level regu_sq_norm = mean across micro-batches of the summed norms
        regu_sq_norm_node = sum(per_mb_sq_norms_synced) / len(per_mb_sq_norms_synced) if per_mb_sq_norms_synced else 0.0
    else:
        per_mb_sq_norms_synced = per_mb_sq_norms_local
        regu_sq_norm_node = regu_sq_norm_local

    # Store synced per-mb norms in detach_state for threshold-based reset
    # and perform reset check at end of step (so next step starts clean)
    if model.detach_state is not None:
        # Expand per-mb sq_norms to per-sample (same value for all samples in a micro-batch)
        _per_sample_sq_norms = []
        for _sq in per_mb_sq_norms_synced:
            _per_sample_sq_norms.extend([_sq] * local_micro_batch_size)
        model.detach_state.set_last_sq_norms(_per_sample_sq_norms)
        # Step+1 for each sample position
        for _si in range(len(_per_sample_sq_norms)):
            model.detach_state.update_steps(_si)
        # Collect reset stats for logging (after step+1, before threshold reset)
        _reset_ratio, _mean_update_step = model.detach_state.get_reset_stats()
        # Perform threshold-based reset for each sample position
        for _si in range(len(_per_sample_sq_norms)):
            model.detach_state.maybe_reset_slice(_si)
    else:
        _reset_ratio, _mean_update_step = 0.0, 0.0

    running_regu_sq_norm += regu_sq_norm_node
    running_reset_ratio += _reset_ratio
    running_mean_update_step += _mean_update_step
    epoch_loss_sum += loss_val
    micro_step += 1

    # --- Optimizer step (every grad_accum_steps micro-batches) ---
    if micro_step % grad_accum_steps == 0:
        # ---- Data-parallel gradient sync (all_reduce AVG) ----
        # This is a collective op: ALL DP replicas must participate,
        # even if we are about to skip the optimizer step due to NaN/Inf.
        sync_gradients_across_dp(
            model.hypernetwork, my_device,
            group=parallel_cfg.get("dp_process_group"),
        )

        # Sync metalora gradients across DP replicas
        _sync_metalora_gradients(model, my_device, parallel_cfg)

        # --- P0 Monitor: Broadcast NaN/Inf skip decision to all processes ---
        # Communication pattern:
        #   1. Intra-node: stage 0 broadcasts to all stages (node_process_group)
        #   2. Inter-node: all_reduce MAX across DP replicas (dp_process_group)
        #      so that if ANY node detects NaN/Inf, ALL processes skip.
        _skip_step = False
        if monitor_nan_inf:
            _skip_tensor = torch.tensor([1.0 if _local_nan_inf else 0.0],
                                         dtype=torch.float32, device=my_device)
            # Step 1: Intra-node broadcast from stage 0
            node_group = parallel_cfg.get("node_process_group")
            if node_group is not None:
                src_global = dist.get_global_rank(node_group, 0)
                dist.broadcast(_skip_tensor, src=src_global, group=node_group)
            # Step 2: Inter-node MAX across DP replicas
            dp_group = parallel_cfg.get("dp_process_group")
            if dp_group is not None:
                dist.all_reduce(_skip_tensor, op=dist.ReduceOp.MAX, group=dp_group)
            _skip_step = _skip_tensor.item() > 0.5
            # If skip was triggered by another node (not this one), update
            # the local tracker on stage 0 so wandb counts are accurate.
            if _skip_step and not _local_nan_inf and nan_inf_tracker is not None:
                my_stage = parallel_cfg["stage"]
                if my_stage == 0:
                    # total_steps was already incremented by check_and_record above,
                    # so only update the bad-step counters here.
                    # Restore consecutive_bad that was reset by check_and_record
                    # (since local loss was OK but remote triggered skip).
                    nan_inf_tracker.nan_count += 1  # count as NaN (unknown remote source)
                    nan_inf_tracker.consecutive_bad = _prev_consecutive_bad + 1
                    logger.warning(
                        f"  [MONITOR] NaN/Inf skip triggered by another DP replica "
                        f"at step {global_step + 1} (this node's loss was OK)."
                    )

        # --- P0 Monitor: Pre-clip gradient norms ---
        _grad_norm_metrics = {}
        _upcoming_step = global_step + 1
        _should_log_grad_norm = (
            grad_norm_sched is not None
            and grad_norm_sched.should_run(_upcoming_step)
            and not _skip_step
        )
        if _should_log_grad_norm:
            # Each stage computes norms for its own parameters
            _local_grad_norms = compute_grad_norms(model, my_device)
            # Gather from all stages to stage 0 via node_process_group
            # gather_object is a collective: ALL ranks in the group must call it.
            node_group = parallel_cfg.get("node_process_group")
            if node_group is not None:
                total_gpus = parallel_cfg.get("total_gpus", 8)
                dst_global = dist.get_global_rank(node_group, 0)
                if is_main_process_per_node():
                    gathered_norms = [None] * total_gpus
                else:
                    gathered_norms = None
                dist.gather_object(
                    _local_grad_norms, gathered_norms,
                    dst=dst_global, group=node_group,
                )
                # Merge all stages' metrics on stage 0 (local_rank 0)
                if parallel_cfg["stage"] == 0 and gathered_norms is not None:
                    # Collect all per-param norms from all stages
                    _all_per_param = {}
                    for stage_norms in gathered_norms:
                        if stage_norms is None:
                            continue
                        for k, v in stage_norms.items():
                            # Only collect per-param norms, skip summary keys
                            # Summary keys: */layer_avg/*, */param_avg/*, *_avg
                            if "/layer_avg/" in k or "/param_avg/" in k or k.endswith("_avg"):
                                continue
                            _all_per_param[k] = v

                    # Add all per-param norms to metrics
                    _grad_norm_metrics.update(_all_per_param)

                    # Recompute per-layer avg, per-param-type avg, and total avg
                    # for both hypernetwork and metalora across all stages
                    for prefix in ("grad_norm/hyper", "grad_norm/meta"):
                        # Collect layer-level norms: {layer_idx: {param_name: norm}}
                        _layer_norms = {}
                        _global_norms = []
                        for k, v in _all_per_param.items():
                            if not k.startswith(prefix + "/"):
                                continue
                            rest = k[len(prefix) + 1:]  # e.g. "layer0/q_proj_weight" or "global/norm_weight"
                            if rest.startswith("layer"):
                                # "layer0/q_proj_weight" -> layer_idx=0, param="q_proj_weight"
                                slash_pos = rest.index("/")
                                layer_str = rest[:slash_pos]  # "layer0"
                                param_name = rest[slash_pos + 1:]
                                layer_idx = int(layer_str.replace("layer", ""))
                                if layer_idx not in _layer_norms:
                                    _layer_norms[layer_idx] = {}
                                _layer_norms[layer_idx][param_name] = v
                                _global_norms.append(v)
                            elif rest.startswith("global/"):
                                _global_norms.append(v)

                        # Per-layer average
                        for layer_idx, param_norms in sorted(_layer_norms.items()):
                            if param_norms:
                                avg = sum(param_norms.values()) / len(param_norms)
                                _grad_norm_metrics[f"{prefix}/layer_avg/layer{layer_idx}"] = avg

                        # Per-param-type average across layers
                        _param_type_norms = {}
                        for layer_idx, param_norms in _layer_norms.items():
                            for pname, norm_val in param_norms.items():
                                if pname not in _param_type_norms:
                                    _param_type_norms[pname] = []
                                _param_type_norms[pname].append(norm_val)
                        for ptype, norms in _param_type_norms.items():
                            _grad_norm_metrics[f"{prefix}/param_avg/{ptype}"] = sum(norms) / len(norms)

                        # Total average
                        avg_key = prefix + "_avg"
                        if _global_norms:
                            _grad_norm_metrics[avg_key] = sum(_global_norms) / len(_global_norms)
                        else:
                            _grad_norm_metrics[avg_key] = 0.0
            else:
                # Single GPU fallback
                _grad_norm_metrics = _local_grad_norms

        # Gradient clipping (after DP sync, before optimizer step)
        # Uses GLOBAL norm across all pipeline stages (intra-node all_reduce)
        # so that all stages apply the same scaling factor.
        if gradient_clipping > 0:
            from utils.myloradict import collect_loradict_tensors
            params_to_clip = [
                p for p in model.hypernetwork.parameters()
                if p.requires_grad and p.grad is not None
                and p.device == my_device
            ]
            metalora_tensors = collect_loradict_tensors(model.metalora)
            params_to_clip.extend([
                t for t in metalora_tensors
                if t.requires_grad and t.grad is not None
                and t.device == my_device
            ])
            # Step 1: compute local gradient norm squared (0 if no params on this stage)
            _local_norm_sq = sum(
                p.grad.float().norm(2).item() ** 2 for p in params_to_clip
            )
            # Step 2: all_reduce across pipeline stages to get global norm
            # ALL stages must participate (collective op), even if local norm is 0
            node_group = parallel_cfg.get("node_process_group")
            if node_group is not None:
                _norm_sq_tensor = torch.tensor(
                    [_local_norm_sq], dtype=torch.float64, device=my_device)
                dist.all_reduce(_norm_sq_tensor, op=dist.ReduceOp.SUM, group=node_group)
                _global_norm = _norm_sq_tensor.item() ** 0.5
            else:
                _global_norm = _local_norm_sq ** 0.5
            # Step 3: compute clip coefficient and scale grads locally
            _clip_coef = gradient_clipping / max(_global_norm, gradient_clipping)
            if _clip_coef < 1.0:
                for p in params_to_clip:
                    p.grad.mul_(_clip_coef)

        # --- P0 Monitor: Post-clip gradient norm ---
        if _should_log_grad_norm:
            _local_post_clip = compute_post_clip_grad_norm(model, my_device)
            # Gather post-clip norms from all stages and sum squares
            node_group = parallel_cfg.get("node_process_group")
            if node_group is not None:
                _post_clip_tensor = torch.tensor(
                    [_local_post_clip ** 2], dtype=torch.float64, device=my_device)
                dist.all_reduce(_post_clip_tensor, op=dist.ReduceOp.SUM, group=node_group)
                _post_clip_total = _post_clip_tensor.item() ** 0.5
            else:
                _post_clip_total = _local_post_clip
            if parallel_cfg["stage"] == 0:
                _grad_norm_metrics["grad_norm/post_clip_total"] = _post_clip_total
                # _global_norm is computed during gradient clipping (all stages)
                _pre_clip_total = _global_norm if gradient_clipping > 0 else 0.0
                _grad_norm_metrics["grad_norm/pre_clip_total"] = _pre_clip_total
                _grad_norm_metrics["grad_norm/clip_ratio"] = (
                    _pre_clip_total / gradient_clipping if gradient_clipping > 0 else 0.0
                )

        # --- P0 Monitor: NaN/Inf skip ---
        if _skip_step:
            optimizer.zero_grad()
            # Force reset wdict so next step starts fresh (wdict may be polluted)
            if model.detach_state is not None:
                model.detach_state.reset()
            # Still increment step counter so training progresses
            model.invalidate_input_cache()
            global_step += 1
            epoch_steps += 1
            # Update loss running avg even on skip (use last known good value)
            # Log NaN/Inf metrics to wandb
            if is_main_process() and use_wandb and nan_inf_tracker is not None:
                _nan_metrics = nan_inf_tracker.get_wandb_metrics()
                wandb.log(_nan_metrics, step=global_step)
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
            return global_step, micro_step, running_loss, running_distill_loss, running_regu_sq_norm, running_reset_ratio, running_mean_update_step, False, epoch_loss_sum, epoch_steps, ema_time_per_step, step_start_time, step_context_tokens, step_conv_total_tokens, step_conv_valid_tokens, total_context_tokens, total_conv_total_tokens, total_conv_valid_tokens, loss_running_avg

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Clear model caches for next iteration
        model.invalidate_input_cache()

        global_step += 1
        epoch_steps += 1

        # --- [DEBUG] Cross-node parameter consistency check ---
        # Controlled by debug.check_consistency_across_nodes config.
        # This is a collective op: ALL DP ranks must participate.
        if check_consistency_sched is not None and check_consistency_sched.should_run(global_step):
            check_dp_param_consistency(
                model=model,
                my_device=my_device,
                parallel_cfg=parallel_cfg,
                global_step=global_step,
            )

        # --- [DEBUG] nograd_loradict integrity check ---
        # Controlled by debug.check_nograd_loradict config.
        # Only runs on main process per node (log output to per-node file).
        #
        # TODO: Currently nograd_loradict is not yet generated by the hypernetwork.
        #       Once you implement nograd_loradict splitting (separating no-grad rank
        #       tensors from grad rank tensors), replace the NotImplementedError below
        #       with the actual check logic:
        #
        #           _nograd_ld = getattr(model, '_last_nograd_loradict', None)
        #           log_nograd_loradict_check(_nograd_ld, global_step)
        #
        #       Also ensure model_hypernetwork.py saves the generated nograd_loradict
        #       to self._last_nograd_loradict after generation.
        if check_nograd_loradict_sched is not None and check_nograd_loradict_sched.should_run(global_step):
            raise NotImplementedError(
                "[DEBUG] check_nograd_loradict is scheduled to run at step "
                f"{global_step}, but nograd_loradict generation is not yet "
                "implemented. Please implement nograd_loradict splitting in "
                "model_hypernetwork.py and then replace this placeholder with "
                "the actual check logic in meta_train.py."
            )

        # --- Logging: gather loss from all DP replicas (actual log output deferred until after save) ---
        _should_log = False
        _gathered_metrics = None
        _global_avg_loss = 0.0
        _global_avg_distill_loss = 0.0
        _global_avg_regu_sq_norm = 0.0
        _global_avg_reset_ratio = 0.0
        _global_avg_repo_reset_ratio = 0.0
        _global_avg_mean_update_step = 0.0
        _global_total_loss = 0.0
        _global_total_ppl = 0.0
        _global_epoch_avg_loss = 0.0
        _global_ppl = 0.0
        _global_epoch_avg_ppl = 0.0
        _lr_now = 0.0
        # --- P0 Monitor: Param norm and generated LoRA norm ---
        # Computed after optimizer step so we see updated weights.
        # Uses same gather pattern as grad_norm (gather_object to stage 0).
        _param_norm_metrics = {}
        _gen_lora_norm_metrics = {}
        _should_log_param_norm = (
            param_norm_sched is not None
            and param_norm_sched.should_run(global_step)
        )
        _should_log_gen_lora_norm = (
            gen_lora_norm_sched is not None
            and gen_lora_norm_sched.should_run(global_step)
        )
        if _should_log_param_norm or _should_log_gen_lora_norm:
            # Compute local norms
            _local_pn = compute_param_norms(model, my_device) if _should_log_param_norm else {}
            _local_gln = compute_generated_lora_norms(model, my_device) if _should_log_gen_lora_norm else {}
            _local_combined = {**_local_pn, **_local_gln}
            # Gather from all stages to stage 0
            node_group = parallel_cfg.get("node_process_group")
            if node_group is not None:
                total_gpus = parallel_cfg.get("total_gpus", 8)
                dst_global = dist.get_global_rank(node_group, 0)
                if is_main_process_per_node():
                    _gathered_pn = [None] * total_gpus
                else:
                    _gathered_pn = None
                dist.gather_object(
                    _local_combined, _gathered_pn,
                    dst=dst_global, group=node_group,
                )
                if parallel_cfg["stage"] == 0 and _gathered_pn is not None:
                    # Collect all per-param norms from all stages
                    _all_pn = {}
                    for stage_norms in _gathered_pn:
                        if stage_norms is None:
                            continue
                        for k, v in stage_norms.items():
                            # Skip summary keys
                            if "/layer_avg/" in k or "/param_avg/" in k or k.endswith("_avg") or k.endswith("/avg"):
                                continue
                            _all_pn[k] = v
                    # Recompute averages for each prefix
                    for prefix in ("param_norm/hyper", "param_norm/meta", "gen_lora_norm"):
                        _layer_norms_pn = {}
                        _global_norms_pn = []
                        for k, v in _all_pn.items():
                            if not k.startswith(prefix + "/"):
                                continue
                            rest = k[len(prefix) + 1:]
                            if rest.startswith("layer"):
                                slash_pos = rest.index("/")
                                layer_str = rest[:slash_pos]
                                param_name = rest[slash_pos + 1:]
                                layer_idx = int(layer_str.replace("layer", ""))
                                if layer_idx not in _layer_norms_pn:
                                    _layer_norms_pn[layer_idx] = {}
                                _layer_norms_pn[layer_idx][param_name] = v
                                _global_norms_pn.append(v)
                            elif rest.startswith("global/"):
                                _global_norms_pn.append(v)
                        for li, pn in sorted(_layer_norms_pn.items()):
                            if pn:
                                _all_pn[f"{prefix}/layer_avg/layer{li}"] = sum(pn.values()) / len(pn)
                        _pt_norms = {}
                        for li, pn in _layer_norms_pn.items():
                            for pname, nv in pn.items():
                                if pname not in _pt_norms:
                                    _pt_norms[pname] = []
                                _pt_norms[pname].append(nv)
                        for pt, ns in _pt_norms.items():
                            _all_pn[f"{prefix}/param_avg/{pt}"] = sum(ns) / len(ns)
                        avg_key = prefix + "_avg" if "/" in prefix else prefix + "/avg"
                        # Determine correct avg key
                        if prefix == "gen_lora_norm":
                            avg_key = "gen_lora_norm/avg"
                        else:
                            avg_key = prefix + "_avg"
                        if _global_norms_pn:
                            _all_pn[avg_key] = sum(_global_norms_pn) / len(_global_norms_pn)
                        else:
                            _all_pn[avg_key] = 0.0
                    # Split into param_norm and gen_lora_norm
                    for k, v in _all_pn.items():
                        if k.startswith("param_norm/"):
                            _param_norm_metrics[k] = v
                        elif k.startswith("gen_lora_norm"):
                            _gen_lora_norm_metrics[k] = v
            else:
                # Single GPU fallback
                for k, v in _local_combined.items():
                    if k.startswith("param_norm/"):
                        _param_norm_metrics[k] = v
                    elif k.startswith("gen_lora_norm"):
                        _gen_lora_norm_metrics[k] = v

        if global_step % logging_steps == 0 and is_main_process_per_node():
            avg_loss = running_loss / (logging_steps * grad_accum_steps)
            avg_distill_loss = running_distill_loss / (logging_steps * grad_accum_steps)
            avg_regu_sq_norm = running_regu_sq_norm / (logging_steps * grad_accum_steps)
            avg_reset_ratio = running_reset_ratio / (logging_steps * grad_accum_steps)
            avg_mean_update_step = running_mean_update_step / (logging_steps * grad_accum_steps)
            # Epoch-level averages (epoch_loss_sum accumulates every micro-batch)
            epoch_micro_batches = epoch_steps * grad_accum_steps
            epoch_avg_loss = epoch_loss_sum / epoch_micro_batches if epoch_micro_batches > 0 else 0.0

            # Gather loss from all DP replicas (nodes) to compute global average
            dp_group = parallel_cfg.get("dp_process_group")
            dp_size = parallel_cfg["data_parallel_size"]
            if dp_group is not None and dp_size > 1:
                # Use gather_object (Gloo-backed) to collect per-node losses on node 0
                local_metrics = {
                    "avg_loss": avg_loss,
                    "avg_distill_loss": avg_distill_loss,
                    "epoch_avg_loss": epoch_avg_loss,
                    "avg_regu_sq_norm": avg_regu_sq_norm,
                    "avg_reset_ratio": avg_reset_ratio,
                    "avg_mean_update_step": avg_mean_update_step,
                    "avg_repo_reset_ratio": model._running_repo_reset_count / (logging_steps * grad_accum_steps) if logging_steps > 0 else 0.0,
                    "node_rank": node_rank,
                    "step_context_tokens": step_context_tokens,
                    "step_conv_total_tokens": step_conv_total_tokens,
                    "step_conv_valid_tokens": step_conv_valid_tokens,
                }
                # gather to DP rank 0 (global rank of first member in dp_group)
                dst_global = dist.get_global_rank(dp_group, 0)
                if is_main_process():
                    gathered_metrics = [None] * dp_size
                else:
                    gathered_metrics = None
                dist.gather_object(
                    local_metrics, gathered_metrics,
                    dst=dst_global,
                    group=dp_group,
                )
            else:
                # Single node: no gather needed
                gathered_metrics = [{"avg_loss": avg_loss, "avg_distill_loss": avg_distill_loss, "epoch_avg_loss": epoch_avg_loss, "avg_regu_sq_norm": avg_regu_sq_norm, "avg_reset_ratio": avg_reset_ratio, "avg_mean_update_step": avg_mean_update_step, "avg_repo_reset_ratio": model._running_repo_reset_count / (logging_steps * grad_accum_steps) if logging_steps > 0 else 0.0, "node_rank": 0, "step_context_tokens": step_context_tokens, "step_conv_total_tokens": step_conv_total_tokens, "step_conv_valid_tokens": step_conv_valid_tokens}]

            # Compute global averages (only on global main process)
            if is_main_process() and gathered_metrics is not None:
                _gathered_metrics = gathered_metrics
                all_losses = [m["avg_loss"] for m in gathered_metrics]
                all_distill_losses = [m["avg_distill_loss"] for m in gathered_metrics]
                all_epoch_losses = [m["epoch_avg_loss"] for m in gathered_metrics]
                _global_avg_loss = sum(all_losses) / len(all_losses)
                _global_avg_distill_loss = sum(all_distill_losses) / len(all_distill_losses)
                _global_epoch_avg_loss = sum(all_epoch_losses) / len(all_epoch_losses)
                all_regu_sq_norms = [m.get("avg_regu_sq_norm", 0.0) for m in gathered_metrics]
                _global_avg_regu_sq_norm = sum(all_regu_sq_norms) / len(all_regu_sq_norms)
                all_reset_ratios = [m.get("avg_reset_ratio", 0.0) for m in gathered_metrics]
                _global_avg_reset_ratio = sum(all_reset_ratios) / len(all_reset_ratios)
                all_mean_update_steps = [m.get("avg_mean_update_step", 0.0) for m in gathered_metrics]
                _global_avg_mean_update_step = sum(all_mean_update_steps) / len(all_mean_update_steps)
                all_repo_reset_ratios = [m.get("avg_repo_reset_ratio", 0.0) for m in gathered_metrics]
                _global_avg_repo_reset_ratio = sum(all_repo_reset_ratios) / len(all_repo_reset_ratios)
                _global_ppl = math.exp(_global_avg_loss) if _global_avg_loss < 20 else float("inf")
                _global_epoch_avg_ppl = math.exp(_global_epoch_avg_loss) if _global_epoch_avg_loss < 20 else float("inf")
                # Total loss = CE loss + coefficient * distill_loss (coefficient already applied)
                _global_total_loss = _global_avg_loss + _global_avg_distill_loss
                _global_total_ppl = math.exp(_global_total_loss) if _global_total_loss < 20 else float("inf")
                _lr_now = lr_scheduler.get_last_lr()[0]
                # Accumulate token counts from all nodes
                step_total_context = sum(m.get("step_context_tokens", 0) for m in gathered_metrics)
                step_total_conv_total = sum(m.get("step_conv_total_tokens", 0) for m in gathered_metrics)
                step_total_conv_valid = sum(m.get("step_conv_valid_tokens", 0) for m in gathered_metrics)
                total_context_tokens += step_total_context
                total_conv_total_tokens += step_total_conv_total
                total_conv_valid_tokens += step_total_conv_valid
                _should_log = True

                # --- P0 Monitor: Update loss running average ---
                if loss_running_avg <= 0:
                    loss_running_avg = _global_avg_loss
                else:
                    loss_running_avg = loss_ema_alpha * _global_avg_loss + (1 - loss_ema_alpha) * loss_running_avg

            running_loss = 0.0
            running_distill_loss = 0.0
            running_regu_sq_norm = 0.0
            running_reset_ratio = 0.0
            running_mean_update_step = 0.0
            model._running_repo_reset_count = 0
            # Reset per-step token counters after logging
            step_context_tokens = 0
            step_conv_total_tokens = 0
            step_conv_valid_tokens = 0

        # --- Detailed training log ---
        if (log_train_detail_sched is not None
                and log_train_detail_sched.should_run(global_step)
                and is_main_process_per_node()
                and detail_log_accumulator is not None
                and tokenizer is not None):
            log_training_detail(
                detail_log_accumulator=detail_log_accumulator,
                tokenizer=tokenizer,
                global_step=global_step,
                epoch=epoch,
                num_mem_token=num_mem_token,
                pad_token_id=pad_token_id,
            )
            detail_log_accumulator.clear()

        # --- Detailed training log triggered by ppl threshold ---
        if (log_train_detail_ppl_threshold > 0
                and is_main_process_per_node()
                and tokenizer is not None
                and _global_ppl > log_train_detail_ppl_threshold):
            _acc_to_use = detail_log_accumulator_ppl if detail_log_accumulator_ppl else detail_log_accumulator
            if _acc_to_use:
                log_training_detail(
                    detail_log_accumulator=_acc_to_use,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    epoch=epoch,
                    num_mem_token=num_mem_token,
                    pad_token_id=pad_token_id,
                    logger_name="debug.training_detail_ppl_threshold",
                )
        if detail_log_accumulator_ppl is not None:
            detail_log_accumulator_ppl.clear()

        # --- Peak memory reporting ---
        if log_peak_memory_sched is not None and log_peak_memory_sched.should_run(global_step):
            _report_peak_memory(f"Step {global_step}")

        # --- Checkpoint saving ---
        save_duration = 0.0
        if save_sched is not None and save_sched.should_run(global_step):
            save_duration = save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                global_step=global_step,
                epoch=epoch,
                micro_step=micro_step,
                run_name=run_name,
                forever_save_steps=forever_save_steps or set(),
                save_total_limit=save_total_limit,
                running_loss=running_loss,
                epoch_loss_sum=epoch_loss_sum,
                epoch_steps=epoch_steps,
                ema_time_per_step=ema_time_per_step,
                total_context_tokens=total_context_tokens,
                total_conv_total_tokens=total_conv_total_tokens,
                total_conv_valid_tokens=total_conv_valid_tokens,
                wandb_run_id=wandb_run_id,
                t_start=t_start,
                max_steps=max_steps,
                train_loader=train_loader,
                config_selections=config_selections,
                launch_cmd=launch_cmd,
            )
            barrier()  # Ensure node 0 finishes saving before other nodes continue
            # Save detach_state on ALL nodes (wdict differs across DP replicas)
            save_detach_state_all_nodes(model, run_name, global_step)

        # --- EMA time-per-step tracking (after save so step_time includes save overhead) ---
        now = time.time()
        last_step_duration = 0.0
        if step_start_time > 0:
            last_step_duration = now - step_start_time
            if ema_time_per_step <= 0:
                ema_time_per_step = last_step_duration  # first step: initialize
            else:
                ema_time_per_step = ema_alpha * last_step_duration + (1 - ema_alpha) * ema_time_per_step
        step_start_time = now

        # --- Step log + wandb (after save so step_time and ETA include save overhead) ---
        if _should_log and is_main_process() and _gathered_metrics is not None:
            steps_remaining = max_steps - global_step
            eta_seconds = ema_time_per_step * steps_remaining if ema_time_per_step > 0 else 0.0
            elapsed = time.time() - t_start
            save_suffix = f", save_time={format_duration(save_duration)}" if save_duration > 0 else ""
            # Log format: loss/ppl = CE loss (backward compatible with pre-distill runs)
            # distill_loss and total_loss are additional when distillation is enabled
            _distill_suffix = ""
            if _global_avg_distill_loss > 0:
                _distill_suffix = (
                    f", distill_loss={_global_avg_distill_loss:.4f}, "
                    f"total_loss={_global_total_loss:.4f}, total_ppl={_global_total_ppl:.2f}"
                )
            _regu_suffix = ""
            if model.detach_state is not None:
                _regu_suffix = (
                    f",\tregu_sq_norm={_global_avg_regu_sq_norm:.4f}"
                    f",\treset_ratio={_global_avg_reset_ratio:.4f}"
                    f",\trepo_reset_ratio={_global_avg_repo_reset_ratio:.4f}"
                    f",\tmean_upd_step={_global_avg_mean_update_step:.1f}"
                )
            logger.info(
                f"  [Step {global_step}/{max_steps}]\t"
                f"epoch={epoch},\tloss={_global_avg_loss:.4f},\tppl={_global_ppl:.2f}{_distill_suffix},\t"
                f"epoch_avg_loss={_global_epoch_avg_loss:.4f},\tepoch_avg_ppl={_global_epoch_avg_ppl:.2f}{_regu_suffix},\t"
                f"lr={_lr_now:.2e},\t"
                f"step_time={format_duration(last_step_duration)}{save_suffix},\t"
                f"elapsed={format_duration(elapsed)},\teta={format_duration(eta_seconds)}"
            )
            if use_wandb:
                wandb_metrics = {
                    "wall_time": elapsed,
                    "train/loss": _global_avg_loss,
                    "train/ppl": _global_ppl,
                    "train/distill_loss": _global_avg_distill_loss,
                    "train/total_loss": _global_total_loss,
                    "train/total_ppl": _global_total_ppl,
                    "train/epoch_avg_loss": _global_epoch_avg_loss,
                    "train/epoch_avg_ppl": _global_epoch_avg_ppl,
                    "train/regu_sq_norm": _global_avg_regu_sq_norm,
                    "train/reset_ratio": _global_avg_reset_ratio,
                    "train/mean_update_step": _global_avg_mean_update_step,
                    "train/repo_reset_ratio": _global_avg_repo_reset_ratio,
                    "train/lr": _lr_now,
                    "train/total_context_tokens": total_context_tokens,
                    "train/total_conv_total_tokens": total_conv_total_tokens,
                    "train/total_conv_valid_tokens": total_conv_valid_tokens,
                    "train/step_time": last_step_duration,
                }

                # --- P0 Monitor: Gradient norm metrics ---
                if _grad_norm_metrics:
                    wandb_metrics.update(_grad_norm_metrics)

                # --- P0 Monitor: Param norm metrics ---
                if _param_norm_metrics:
                    wandb_metrics.update(_param_norm_metrics)

                # --- P0 Monitor: Generated LoRA norm metrics ---
                if _gen_lora_norm_metrics:
                    wandb_metrics.update(_gen_lora_norm_metrics)

                # --- P0 Monitor: Update ratio (grad_norm * lr / param_norm) ---
                # Automatically computed when both grad_norm and param_norm
                # are available in the same step.
                if _grad_norm_metrics and _param_norm_metrics and _lr_now > 0:
                    for gn_key, gn_val in _grad_norm_metrics.items():
                        # Map grad_norm key to corresponding param_norm key
                        # e.g. "grad_norm/hyper/layer0/q_proj_weight" -> "param_norm/hyper/layer0/q_proj_weight"
                        if not gn_key.startswith("grad_norm/"):
                            continue
                        pn_key = "param_norm/" + gn_key[len("grad_norm/"):]
                        pn_val = _param_norm_metrics.get(pn_key)
                        if pn_val is not None and pn_val > 1e-12:
                            ur_key = "update_ratio/" + gn_key[len("grad_norm/"):]
                            wandb_metrics[ur_key] = gn_val * _lr_now / pn_val

                # --- P0 Monitor: Loss spike metrics ---
                if monitor_loss_spike and _gathered_metrics is not None:
                    _spike_metrics = compute_loss_spike_metrics(
                        _gathered_metrics, loss_running_avg,
                    )
                    wandb_metrics.update(_spike_metrics)

                # --- P0 Monitor: NaN/Inf metrics ---
                if monitor_nan_inf and nan_inf_tracker is not None:
                    wandb_metrics.update(nan_inf_tracker.get_wandb_metrics())

                wandb.log(wandb_metrics, step=global_step)

    return global_step, micro_step, running_loss, running_distill_loss, running_regu_sq_norm, running_reset_ratio, running_mean_update_step, False, epoch_loss_sum, epoch_steps, ema_time_per_step, step_start_time, step_context_tokens, step_conv_total_tokens, step_conv_valid_tokens, total_context_tokens, total_conv_total_tokens, total_conv_valid_tokens, loss_running_avg
# Main
# ---------------------------------------------------------------------------

from torch.distributed.elastic.multiprocessing.errors import record

@record
@hydra.main(version_base=None, config_path="configs", config_name="main_pretrain")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    # Dispatch: TP path runs its own self-contained training loop in
    # meta_train_tp.tp_main (forward -> loss -> backward -> step, no
    # pipeline scheduling). PP path below stays bit-for-bit unchanged.
    # Selected via cfg.parallel.mode ('pp' | 'tp'); default 'pp'.
    # ------------------------------------------------------------------
    if str(cfg.parallel.get("mode", "pp")).lower() == "tp":
        from meta_train_tp import tp_main
        tp_main(cfg)
        return

    # ==================================================================
    # 1. Distributed init
    # ==================================================================
    init_distributed()

    if is_main_process():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("training.log"),
            ],
            force=True,  # Override Hydra's logging configuration
        )
        logger.info("Starting meta training …")
    else:
        # Non-main processes: configure logging to output ERROR+ to stderr
        # so that crashes on any rank are visible in the log
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s - [rank %(process)d] %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
            force=True,
        )

    # --- Setup dedicated debug loggers (per-category, per-node) ---
    from hydra.utils import get_original_cwd
    _orig_cwd = get_original_cwd()
    _log_subdir = os.environ.get("LOG_SUBDIR", "")
    _log_dir = os.path.join(_orig_cwd, "logs", _log_subdir) if _log_subdir else os.path.join(_orig_cwd, "logs")
    _node_rank_for_log = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0")))
    # Generate a unique session_id (timestamp) so each run/resume produces
    # separate log files that won't overwrite each other on wandb.
    from datetime import datetime
    _session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _debug_categories, _debug_log_paths, _node0_only_categories = setup_debug_loggers(
        log_dir=_log_dir,
        node_rank=_node_rank_for_log,
        session_id=_session_id,
    )

    if is_main_process_per_node():
        logger.info(
            f"Debug logs directory: {_log_dir} "
            f"(categories: {list(_debug_categories.keys())})"
        )

    # ==================================================================
    # 2. Pipeline parallelism
    # ==================================================================
    pipeline_config = setup_pipeline_parallel(
        total_gpus=cfg.parallel.total_gpus,
        pipeline_parallel_size=cfg.parallel.pipeline_parallel_size,
    )
    parallel_cfg = get_pipeline_config()
    my_device = parallel_cfg["device"]

    if is_main_process_per_node():
        logger.info(f"Pipeline stage {parallel_cfg['stage']+1}/{parallel_cfg['total_stages']}, "
                     f"DP rank {parallel_cfg['data_parallel_rank']}, device {my_device}")

    all_gpu_stats("After pipeline setup")

    # ==================================================================
    # 2a. Load parallel-mode-specific functions (PP or TP)
    # ==================================================================
    parallel_mode = cfg.parallel.get("mode", "pp")  # Default to PP for backward compatibility
    _load_parallel_mode_functions(parallel_mode)
    if is_main_process_per_node():
        logger.info(f"Parallel mode: {parallel_mode} (train_step/run_evaluation loaded from meta_train_{parallel_mode}.py)")


    # ==================================================================
    # 3. Load model
    # ==================================================================
    # Inject detach_state config into model config (detach_state is a
    # top-level Hydra config group, but ModelHypernetwork reads it from
    # model_cfg.detach_state).
    if cfg.get("detach_state"):
        from omegaconf import OmegaConf, open_dict
        with open_dict(cfg.model):
            cfg.model.detach_state = cfg.detach_state

    if is_main_process_per_node():
        logger.info("Loading ModelHypernetwork …")

    try:
        compile_mode = cfg.get("torch_compile", {}).get("mode", None)
        activation_checkpointing = cfg.training.get("pp_knobs", {}).get("activation_checkpointing", False)
        model = ModelHypernetwork(
            model_cfg=cfg.model,
            m2p_transformer_cfg=cfg.m2p_transformer,
            training_cfg=cfg.training,
            debug_anchor=cfg.get("debug", {}).get("anchor_debug", False),
            compile_mode=compile_mode,
            activation_checkpointing=activation_checkpointing,
        )
    except Exception as e:
        if is_main_process_per_node():
            logger.error(f"Failed to load model: {e}")
        cleanup_distributed()
        sys.exit(1)

    all_gpu_stats("After model load")

    # ==================================================================
    # 3a. Optionally disable SDPA Flash backend (for A/B comparison)
    # ==================================================================
    no_flash = cfg.get("debug", {}).get("no_flash", False)
    if no_flash and is_main_process_per_node():
        logger.info(
            "⚠ debug.no_flash=true → SDPA Flash backend DISABLED "
            "(will use MATH / EFFICIENT_ATTENTION backends instead)"
        )

    # Build a factory that creates a context manager disabling Flash backend.
    # When no_flash=False, this is a no-op (nullcontext).
    def make_sdpa_ctx():
        if no_flash:
            return sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        return nullcontext()

    # ==================================================================
    # 3b. Verify SDPA Flash Attention
    # ==================================================================
    _verify_sdpa_flash_attention(model, my_device)

    # ==================================================================
    # 4. Resolve model path & pad_token_id
    # ==================================================================
    from hydra.utils import get_original_cwd
    _model_path = str(cfg.model.path)
    if not os.path.isabs(_model_path):
        _model_path = os.path.join(get_original_cwd(), _model_path)
    pad_token_id = resolve_pad_token_id(_model_path, tokenizer_cfg=cfg.tokenizer)
    vocab_size = model._text_config.vocab_size
    num_mem_token = getattr(model._text_config, 'num_mem_token', 0) or 0

    if is_main_process_per_node():
        logger.info(f"pad_token_id={pad_token_id}, vocab_size={vocab_size}")

    # ==================================================================
    # 5. Create dataset & dataloader
    # ==================================================================
    local_batch_size = cfg.training.pp_batchsize.local_batch_size
    local_micro_batch_size = cfg.training.pp_batchsize.local_micro_batch_size
    context_seq_len = cfg.data.context_seq_length
    conv_seq_len = cfg.data.conv_seq_length

    # Validate: local_batch_size must be an exact multiple of local_micro_batch_size
    if local_batch_size % local_micro_batch_size != 0:
        raise ValueError(
            f"local_batch_size ({local_batch_size}) must be an exact multiple of "
            f"local_micro_batch_size ({local_micro_batch_size})."
        )
    global_batch = local_batch_size * parallel_cfg["data_parallel_size"]

    # Use utility function to create dataset and collator (with optional val split)
    train_dataset, val_dataset, collator = create_dataset_from_config(
        cfg, _model_path, pad_token_id, num_mem_token
    )
    num_train_samples = len(train_dataset)
    data_name = cfg.data.get("name", None)
    if not data_name:
        raise ValueError(
            "cfg.data.name is required but not set. "
            "Every data config YAML must explicitly specify a 'name' field."
        )

    # Pre-compute batches_per_epoch for non-first stages so they can
    # synchronize epoch boundaries with the first stage.
    # For first stage with DistributedSampler + drop_last:
    #   samples_per_replica = ceil(num_samples / dp_size)
    #   batches = samples_per_replica // local_batch_size
    # We use the same formula here for consistency.
    _samples_per_replica = math.ceil(num_train_samples / parallel_cfg["data_parallel_size"])
    _estimated_batches_per_epoch = _samples_per_replica // local_batch_size

    dataset_seed = cfg.seed.dataset

    train_loader = PipelineDataLoader(
        dataset=train_dataset,
        batch_size=global_batch,
        micro_batch_size=local_micro_batch_size,
        data_parallel_rank=parallel_cfg["data_parallel_rank"],
        data_parallel_size=parallel_cfg["data_parallel_size"],
        pipeline_stage=parallel_cfg["stage"],
        total_pipeline_stages=parallel_cfg["total_stages"],
        num_workers=cfg.data.get("num_workers", 4),
        shuffle=cfg.data.shuffle,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        batches_per_epoch=_estimated_batches_per_epoch,
        seed=dataset_seed,
    )

    # --- Validation data loader ---
    val_loader = None
    eval_local_batch_size = None
    eval_local_micro_batch_size = None
    if val_dataset is not None:
        # Parse evaluation batch size config (PP mode)
        _eval_pp_cfg = cfg.training.get("evaluation_pp_batchsize", {})
        _eval_lbs_raw = _eval_pp_cfg.get("local_batch_size", -1) if _eval_pp_cfg else -1
        _eval_mbs_raw = _eval_pp_cfg.get("local_micro_batch_size", 1) if _eval_pp_cfg else 1

        _val_total = len(val_dataset)
        _val_samples_per_replica = math.ceil(_val_total / parallel_cfg["data_parallel_size"])

        if _eval_lbs_raw == -1:
            # -1 means use entire val set as one batch per node
            eval_local_batch_size = _val_samples_per_replica
        else:
            eval_local_batch_size = int(_eval_lbs_raw)

        eval_local_micro_batch_size = int(_eval_mbs_raw)

        # Ensure eval_local_batch_size is divisible by eval_local_micro_batch_size
        if eval_local_batch_size % eval_local_micro_batch_size != 0:
            # Round down to nearest multiple
            eval_local_batch_size = (eval_local_batch_size // eval_local_micro_batch_size) * eval_local_micro_batch_size
            if eval_local_batch_size == 0:
                eval_local_batch_size = eval_local_micro_batch_size

        eval_global_batch = eval_local_batch_size * parallel_cfg["data_parallel_size"]
        _val_batches = _val_samples_per_replica // eval_local_batch_size

        val_loader = PipelineDataLoader(
            dataset=val_dataset,
            batch_size=eval_global_batch,
            micro_batch_size=eval_local_micro_batch_size,
            data_parallel_rank=parallel_cfg["data_parallel_rank"],
            data_parallel_size=parallel_cfg["data_parallel_size"],
            pipeline_stage=parallel_cfg["stage"],
            total_pipeline_stages=parallel_cfg["total_stages"],
            num_workers=cfg.data.get("num_workers", 4),
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=collator,
            batches_per_epoch=_val_batches,
            seed=dataset_seed,
        )

    if is_main_process_per_node():
        logger.info(
            f"Dataset: {num_train_samples} train samples"
            f"{f', {len(val_dataset)} val samples' if val_dataset is not None else ''}"
            f" (name={data_name}), "
            f"global_batch={global_batch}, local_batch={local_batch_size}, "
            f"micro_batch={local_micro_batch_size}"
        )
        if val_dataset is not None:
            logger.info(
                f"  Eval config: eval_local_batch={eval_local_batch_size}, "
                f"eval_micro_batch={eval_local_micro_batch_size}"
            )

    # ==================================================================
    # 6. Create optimizer & scheduler
    # ==================================================================
    grad_accum_steps = cfg.training.pp_batchsize.gradient_accumulation_steps
    max_steps_raw = cfg.training.max_steps  # may be int or str like "2-epoch"

    # --- Step number variables explained ---
    # batches_per_epoch:  Number of DataLoader batches in one pass over the
    #                     dataset (= len(data_loader)).  Each batch has
    #                     local_batch_size samples.
    # grad_accum_steps:   Number of consecutive forward-backward passes
    #                     (micro-batches) whose gradients are accumulated
    #                     before one optimizer.step().
    # steps_per_epoch:    Number of optimizer steps per epoch
    #                     = batches_per_epoch // grad_accum_steps.
    # max_steps:          Hard upper bound on total optimizer steps.
    #                     The scheduler and training loop both use this.
    # num_epochs:         Number of full dataset passes needed to reach
    #                     max_steps.  Rounded UP so we never stop short.
    # total_training_steps: Always == max_steps.  Passed to the LR scheduler
    #                     so warmup + decay spans exactly max_steps.

    if train_loader.is_first_stage and train_loader.data_loader is not None:
        batches_per_epoch = len(train_loader.data_loader)
    else:
        # Non-first stages don't load data; use the same estimate as
        # _estimated_batches_per_epoch (consistent with DistributedSampler + drop_last).
        batches_per_epoch = _estimated_batches_per_epoch
    steps_per_epoch = max(batches_per_epoch // grad_accum_steps, 1)

    # --- Resolve max_steps: supports integer or "n-epoch" format ---
    # Examples: max_steps: 50000       → use 50000 directly
    #           max_steps: "2-epoch"   → 2 * steps_per_epoch
    #           max_steps: "0.5-epoch" → int(0.5 * steps_per_epoch)
    import re
    _epoch_pattern = re.compile(r"^([\d.]+)-epoch$", re.IGNORECASE)
    if isinstance(max_steps_raw, str):
        m = _epoch_pattern.match(max_steps_raw.strip())
        if m:
            n_epochs_requested = float(m.group(1))
            max_steps = max(int(n_epochs_requested * steps_per_epoch), 1)
            if is_main_process_per_node():
                logger.info(
                    f"max_steps resolved from '{max_steps_raw}': "
                    f"{n_epochs_requested} epochs × {steps_per_epoch} steps/epoch "
                    f"= {max_steps} steps"
                )
        else:
            raise ValueError(
                f"training.max_steps='{max_steps_raw}' is not a valid format. "
                f"Expected an integer or a string like '2-epoch' / '0.5-epoch'."
            )
    else:
        max_steps = int(max_steps_raw)

    # Guard: when max_steps is an integer (not n-epoch format) and exceeds
    # steps_per_epoch, training would cross an epoch boundary.  The n-epoch
    # format intentionally trains for multiple epochs, so it is always allowed
    # (PipelineDataLoader now synchronizes epoch boundaries across all stages).
    _is_epoch_format = isinstance(max_steps_raw, str) and _epoch_pattern.match(max_steps_raw.strip())
    if not _is_epoch_format and max_steps > steps_per_epoch:
        raise ValueError(
            f"max_steps ({max_steps}) > steps_per_epoch ({steps_per_epoch}). "
            f"When using an integer max_steps, it must not exceed steps_per_epoch "
            f"to avoid crossing epoch boundaries. Either reduce max_steps to "
            f"<= {steps_per_epoch}, increase the dataset size, or use the "
            f"'n-epoch' format (e.g. '10-epoch') to train for multiple epochs."
        )

    # Round UP so we have enough epochs to reach max_steps
    num_epochs = math.ceil(max_steps / steps_per_epoch)
    # Scheduler always uses max_steps so warmup + decay are exact
    total_training_steps = max_steps

    # Resolve warmup_steps: supports integer (exact steps) or float in (0, 1) (proportion of total steps)
    _raw_warmup = cfg.training.warmup_steps
    if isinstance(_raw_warmup, float) and 0 < _raw_warmup < 1:
        warmup_steps = int(_raw_warmup * total_training_steps)
    else:
        warmup_steps = int(_raw_warmup)

    if is_main_process_per_node():
        logger.info(
            f"Training plan: up to {num_epochs} epochs × {steps_per_epoch} steps/epoch, "
            f"total_training_steps={total_training_steps} (=max_steps={max_steps}), "
            f"grad_accum={grad_accum_steps}, warmup_steps={warmup_steps}"
        )

    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model=model,
        num_training_steps=total_training_steps,
        learning_rate=cfg.optimizer.learning_rate,
        min_learning_rate=cfg.optimizer.get("min_learning_rate", 0.0),
        weight_decay=cfg.optimizer.weight_decay,
        beta1=cfg.optimizer.beta1,
        beta2=cfg.optimizer.beta2,
        eps=cfg.optimizer.eps,
        warmup_steps=warmup_steps,
    )

    # ==================================================================
    # 7. Parameter audit
    # ==================================================================
    audit_parameters(model, optimizer)

    _report_peak_memory("Before training")

    # ==================================================================
    # 8. Training loop
    # ==================================================================
    if is_main_process_per_node():
        logger.info("=" * 70)
        logger.info("  Starting training")
        logger.info("=" * 70)

    gradient_clipping = cfg.training.gradient_clipping
    logging_steps = cfg.training.logging_steps

    # --- Distillation setup ---
    from utils.mydistill import create_distill_loss_fn
    distill_cfg = cfg.training.get("distill", None)
    distill_loss_fn = create_distill_loss_fn(distill_cfg)
    if distill_loss_fn is not None:
        # Validate distill batch size matches local_batch_size
        distill_local_batch_size = int(distill_cfg.get("distill_local_batch_size", local_batch_size))
        distill_micro_batch_size = int(distill_cfg.get("distill_micro_batch_size", local_micro_batch_size))
        if distill_local_batch_size != local_batch_size:
            raise ValueError(
                f"distill_local_batch_size ({distill_local_batch_size}) must equal "
                f"local_batch_size ({local_batch_size}). This is currently enforced."
            )
        if distill_local_batch_size % distill_micro_batch_size != 0:
            raise ValueError(
                f"distill_local_batch_size ({distill_local_batch_size}) must be divisible by "
                f"distill_micro_batch_size ({distill_micro_batch_size})."
            )
        if is_main_process_per_node():
            logger.info(
                f"  Distillation enabled: mode={distill_cfg.mode}, "
                f"loss_type={distill_cfg.loss_type}, "
                f"coefficient={distill_cfg.coefficient}, "
                f"temperature={distill_cfg.get('temperature', 'N/A')}, "
                f"distill_micro_batch_size={distill_micro_batch_size}"
            )

        # Distillation backward (Phase C') requires retain_graph=True because
        # the loradict computation graph is shared between Phase C and C' on
        # LLM stages. This is incompatible with torch.compile's donated_buffer
        # optimization which frees buffers eagerly.
        torch._functorch.config.donated_buffer = False
    else:
        distill_micro_batch_size = local_micro_batch_size
        if is_main_process_per_node():
            logger.info("  Distillation: disabled")

    save_sched = DebugSchedule(
        cfg.training.get("save_steps", -1), "save_steps")
    forever_save_steps = resolve_forever_save_steps(
        cfg.training.get("forever_save_steps", []))
    save_total_limit = int(cfg.training.get("save_total_limit", 5))
    # --- Resolve training mode and experiment identifiers ---
    training_mode = cfg.get("mode", "pretrain")
    exp_name = os.environ.get("EXP_NAME", "")
    annealing_name = os.environ.get("ANNEALING_NAME", "")
    sft_name = os.environ.get("SFT_NAME", "")
    # Build checkpoint run_name path:
    #   'name/pretrain', 'name/pretrain_annealing/annealing_name', or 'name/sft/annealing_name/sft_name'
    run_name = build_checkpoint_run_name(exp_name, training_mode, annealing_name, sft_name)
    # Build wandb run name:
    #   pretrain: 'name'
    #   pretrain_annealing: 'name_annealing_name'
    #   sft: 'name_annealing_name_sft_name'
    if training_mode == "sft":
        wandb_display_name = f"{exp_name}_{annealing_name}_{sft_name}"
    elif training_mode == "pretrain_annealing":
        wandb_display_name = f"{exp_name}_{annealing_name}"
    else:
        wandb_display_name = exp_name
    if is_main_process_per_node():
        logger.info(f"  Training mode: {training_mode}, exp_name: {exp_name}, "
                     f"annealing_name: {annealing_name or 'N/A'}, "
                     f"sft_name: {sft_name or 'N/A'}, "
                     f"checkpoint run_name: {run_name}")
    eval_sched = DebugSchedule(
        cfg.training.get("eval_steps", -1), "eval_steps")
    log_peak_memory_sched = DebugSchedule(
        cfg.debug.get("log_peak_memory_steps", -1), "log_peak_memory_steps")
    log_train_detail_sched = DebugSchedule(
        cfg.debug.get("log_train_detail_steps", -1), "log_train_detail_steps")
    log_train_detail_ppl_threshold = cfg.debug.get("log_train_detail_ppl_threshold", -1)
    check_consistency_sched = DebugSchedule(
        cfg.debug.get("check_consistency_across_nodes", -1), "check_consistency_across_nodes")
    check_nograd_loradict_sched = DebugSchedule(
        cfg.debug.get("check_nograd_loradict", -1), "check_nograd_loradict")

    # --- P0 Monitor schedules ---
    _monitor_cfg = cfg.debug.get("monitor", {})
    grad_norm_sched = DebugSchedule(
        _monitor_cfg.get("grad_norm_steps", -1), "grad_norm_steps")
    param_norm_sched = DebugSchedule(
        _monitor_cfg.get("param_norm_steps", -1), "param_norm_steps")
    gen_lora_norm_sched = DebugSchedule(
        _monitor_cfg.get("gen_lora_norm_steps", -1), "gen_lora_norm_steps")
    monitor_loss_spike = bool(_monitor_cfg.get("loss_spike", False))
    monitor_nan_inf = bool(_monitor_cfg.get("nan_inf_detect", False))
    nan_inf_tracker = NanInfTracker() if monitor_nan_inf else None
    nan_inf_stop_steps = int(_monitor_cfg.get("nan_inf_stop_steps", -1))
    loss_running_avg = 0.0

    if is_main_process_per_node():
        logger.info(
            f"  [Monitor] grad_norm_steps={grad_norm_sched}, "
            f"param_norm_steps={param_norm_sched}, "
            f"gen_lora_norm_steps={gen_lora_norm_sched}, "
            f"loss_spike={monitor_loss_spike}, nan_inf_detect={monitor_nan_inf}, "
            f"nan_inf_stop_steps={nan_inf_stop_steps}"
        )

    # Load tokenizer for detail logging (only needed on first stage / main process)
    _detail_tokenizer = None
    detail_log_accumulator: List[Dict[str, Any]] = []
    detail_log_accumulator_ppl: List[Dict[str, Any]] = []
    _need_tokenizer = log_train_detail_sched.enabled or log_train_detail_ppl_threshold > 0
    if _need_tokenizer and train_loader.is_first_stage:
        from utils.mytokenizer import create_tokenizer
        _detail_tokenizer = create_tokenizer(
            _model_path, tokenizer_cfg=cfg.tokenizer
        )
        if is_main_process_per_node():
            logger.info(
                f"debug.log_train_detail_steps={log_train_detail_sched}, "
                f"log_train_detail_ppl_threshold={log_train_detail_ppl_threshold} "
                f"(tokenizer loaded for detail logging)")

    # Wandb initialization (only on main process of node 0)
    # Priority: environment variables > yaml config > defaults
    # Force online mode — proxy is configured in run_training.sh
    # --- Resolve resume checkpoint ---
    resume_from_raw = cfg.training.get("resume_from", "latest")
    resume_checkpoint_dir = None
    load_model_only_flag = False  # Flag: loading model-only (no optimizer/scheduler) from a prior stage
    if resume_from_raw is not None and str(resume_from_raw).lower() != "null" and str(resume_from_raw).lower() != "none":
        if str(resume_from_raw).lower() == "latest":
            resume_checkpoint_dir = get_latest_checkpoint(run_name)
            if resume_checkpoint_dir is not None:
                if is_main_process_per_node():
                    logger.info(f"  [Resume] Found latest checkpoint: {resume_checkpoint_dir}")
            else:
                if is_main_process_per_node():
                    logger.info(f"  [Resume] No existing checkpoint found for run '{run_name}'.")
                # For SFT mode: auto-load pretrain_annealing final (or pretrain final if annealing_name=null)
                if training_mode == "sft":
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
                        # Default: load pretrain_annealing final checkpoint for the specified annealing_name
                        _annealing_final = get_pretrain_annealing_final_checkpoint(exp_name, annealing_name)
                        if _annealing_final is not None:
                            resume_checkpoint_dir = _annealing_final
                            load_model_only_flag = True
                            if is_main_process_per_node():
                                logger.info(f"  [Resume] SFT mode: loading pretrain_annealing final checkpoint: {_annealing_final}")
                        else:
                            raise RuntimeError(
                                f"SFT mode: no pretrain_annealing final checkpoint found for '{exp_name}' "
                                f"with annealing_name='{annealing_name}'. "
                                f"Expected at: checkpoint/{exp_name}/pretrain_annealing/{annealing_name}/final/. "
                                f"Use --annealing_name null to load pretrain final checkpoint directly."
                            )
                # For pretrain_annealing mode: auto-load pretrain final checkpoint
                elif training_mode == "pretrain_annealing":
                    _pretrain_final = get_pretrain_final_checkpoint(exp_name)
                    if _pretrain_final is not None:
                        resume_checkpoint_dir = _pretrain_final
                        load_model_only_flag = True
                        if is_main_process_per_node():
                            logger.info(f"  [Resume] pretrain_annealing mode: loading pretrain final checkpoint: {_pretrain_final}")
                    else:
                        raise RuntimeError(
                            f"pretrain_annealing mode: no pretrain final checkpoint found for '{exp_name}'. "
                            f"Expected at: checkpoint/{exp_name}/pretrain/final/"
                        )
                else:
                    if is_main_process_per_node():
                        logger.info(f"  [Resume] Starting fresh.")
        else:
            # Explicit path
            resume_checkpoint_dir = str(resume_from_raw)
            if not os.path.isabs(resume_checkpoint_dir):
                from hydra.utils import get_original_cwd
                resume_checkpoint_dir = os.path.join(get_original_cwd(), resume_checkpoint_dir)
            if not os.path.exists(resume_checkpoint_dir):
                if is_main_process_per_node():
                    logger.warning(f"  [Resume] Checkpoint path does not exist: {resume_checkpoint_dir}. Starting fresh.")
                resume_checkpoint_dir = None

    # Load resume metadata to get wandb_run_id before wandb.init
    # Note: when loading model-only from a prior stage, we do NOT use its wandb_run_id
    _resume_metadata = {}
    if resume_checkpoint_dir is not None and not load_model_only_flag:
        _meta_path = os.path.join(resume_checkpoint_dir, "training_state", "metadata.pt")
        if os.path.exists(_meta_path):
            _resume_metadata = torch.load(_meta_path, map_location="cpu")

    # --- Resolve current config selections (needed for consistency check + wandb) ---
    # Read defaults from the corresponding main_*.yaml file.
    # If an environment variable is set, it takes priority; otherwise use the yaml default.
    import re as _re_mod
    _configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    if training_mode == "sft":
        _main_yaml_path = os.path.join(_configs_dir, "main_sft.yaml")
    elif training_mode == "pretrain_annealing":
        _main_yaml_path = os.path.join(_configs_dir, "main_pretrain_annealing.yaml")
    else:
        _main_yaml_path = os.path.join(_configs_dir, "main_pretrain.yaml")

    _yaml_defaults = {}
    if os.path.exists(_main_yaml_path):
        with open(_main_yaml_path, "r") as _yf:
            for _line in _yf:
                _m = _re_mod.match(r'^\s*-\s+(\w+):\s*([^\s#]+)', _line)
                if _m:
                    _yaml_defaults[_m.group(1)] = _m.group(2)

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
    _missing_configs = []
    for _key, _env_var in _config_env_keys.items():
        _val = os.environ.get(_env_var, "")
        if not _val:
            _val = _yaml_defaults.get(_key, "")
        if not _val:
            _missing_configs.append(f"{_env_var} (yaml key: {_key})")
        _resolved_configs[_key] = _val
    if _missing_configs:
        raise RuntimeError(
            f"[FATAL] The following configs could not be resolved (not in env and not in {_main_yaml_path}):\n"
            f"    {', '.join(_missing_configs)}\n"
            f"Set them via environment variables or ensure they are defined in the main yaml file."
        )

    # --- Config consistency check on resume ---
    # If resuming from a checkpoint that has saved config_selections, compare with current.
    # Abort unless FORCE_OVERWRITE=1 is set.
    _force_overwrite = os.environ.get("FORCE_OVERWRITE", "") == "1"
    _config_changes = {}  # {key: (old_value, new_value)}
    if _resume_metadata and not load_model_only_flag:
        _saved_configs = _resume_metadata.get("config_selections", None)
        _saved_launch_cmd = _resume_metadata.get("launch_cmd", None)
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
                if _saved_launch_cmd:
                    _diff_msg += f"\n    [previous launch_cmd]: {_saved_launch_cmd}"
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

    use_wandb = False
    wandb_run_id = None
    # Honour WANDB_MODE / WANDB_DISABLED so we don't drag in wandb's
    # heavy import chain (sentry_sdk, etc.) on offline/CI boxes.
    _wandb_off = (
        os.environ.get("WANDB_MODE", "online").lower() in ("disabled", "offline")
        or os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
    )
    if is_main_process() and not _wandb_off:
        wandb_cfg = cfg.get("wandb", {})
        wandb_project = os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project", "SHINE_V2")
        wandb_run_name = wandb_display_name or os.environ.get("WANDB_NAME") or wandb_cfg.get("run_name", None)
        wandb_entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity", None)

        # _resolved_configs is already computed above (before wandb block)

        # Build wandb tags and experiment metadata for filtering
        # Note: wandb tags have a 64-character limit
        wandb_tags = [training_mode, f"parallel={cfg.parallel.get('mode', 'pp')}"]
        if exp_name:
            _tag_name = exp_name[:64] if len(exp_name) > 64 else exp_name
            wandb_tags.append(_tag_name)
        if annealing_name and annealing_name != "null":
            _ann_tag = annealing_name[:64] if len(annealing_name) > 64 else annealing_name
            wandb_tags.append(_ann_tag)
        if annealing_name == "null":
            wandb_tags.append("no_annealing")
        # Add key config module selections as tags for easy filtering in wandb UI
        # Format: "cfg:<key>=<value>" (truncated to 64 chars)
        for _cfg_key, _cfg_val in _resolved_configs.items():
            _cfg_tag = f"{_cfg_key}={_cfg_val}"
            if len(_cfg_tag) > 64:
                _cfg_tag = _cfg_tag[:64]
            wandb_tags.append(_cfg_tag)

        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config["experiment_name"] = exp_name
        wandb_config["training_mode"] = training_mode
        wandb_config["annealing_name"] = annealing_name if annealing_name else None
        wandb_config["sft_name"] = sft_name if training_mode == "sft" else None
        wandb_config["launch_cmd"] = os.environ.get("LAUNCH_CMD", "")
        # Store resolved config selections prominently in wandb config
        wandb_config["config_selections"] = _resolved_configs
        # Parallel mode fields for unified filtering across PP/TP runs
        wandb_config["parallel_mode"] = cfg.parallel.get("mode", "pp")
        wandb_config["pp_stages"] = cfg.parallel.get("pipeline_parallel_size", 8)
        wandb_config["tp_size"] = cfg.parallel.get("tensor_parallel_size", 2)
        wandb_config["total_gpus"] = cfg.parallel.get("total_gpus", 8)

        # Build wandb notes with config summary for quick identification in UI
        _notes_lines = [f"mode={training_mode}"]
        for _cfg_key, _cfg_val in _resolved_configs.items():
            _notes_lines.append(f"{_cfg_key}={_cfg_val}")
        wandb_notes = " | ".join(_notes_lines)

        # If resuming, use the saved wandb run_id to continue the same run
        _saved_wandb_id = _resume_metadata.get("wandb_run_id", None)
        if _saved_wandb_id is not None and resume_checkpoint_dir is not None:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                id=_saved_wandb_id,
                resume="must",
                tags=wandb_tags,
                config=wandb_config,
                notes=wandb_notes,
                reinit="finish_previous",
                mode="online",
            )
            logger.info(f"Wandb RESUMED: project={wandb_project}, run_id={_saved_wandb_id}")
        else:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                tags=wandb_tags,
                config=wandb_config,
                notes=wandb_notes,
                reinit="finish_previous",
                mode="online",
            )
        use_wandb = True
        wandb_run_id = wandb.run.id
        # Define wall_time as custom x-axis so that resume continues seamlessly
        wandb.define_metric("wall_time")
        wandb.define_metric("train/*", step_metric="wall_time")
        wandb.define_metric("eval/*", step_metric="wall_time")
        logger.info(f"Wandb logging enabled: project={wandb_project}, run={wandb.run.name}, id={wandb_run_id}")

        # Register all debug log files for real-time upload to wandb.
        # wandb.save with policy="live" monitors file changes and syncs
        # continuously. Each session has unique filenames (with session_id),
        # so previous sessions' logs are preserved on wandb.
        import glob as _glob
        # Save this session's debug log files
        for _dpath in _debug_log_paths.values():
            if os.path.exists(_dpath):
                wandb.save(_dpath, base_path=_orig_cwd, policy="live")
        # Register glob patterns for this session's logs from other nodes
        for cat_name in _debug_categories:
            if cat_name in _node0_only_categories:
                _pattern = os.path.join(_log_dir, f"{cat_name}_{_session_id}.log")
            else:
                _pattern = os.path.join(_log_dir, f"{cat_name}_node*_{_session_id}.log")
            wandb.save(_pattern, base_path=_orig_cwd, policy="live")
        # Also upload any existing logs from previous sessions (one-time upload)
        # so that resumed runs have complete history on wandb.
        # Exclude shell-managed files (node_*.log, launcher.log) which are always overwritten.
        import re as _re
        _shell_log_pattern = _re.compile(r'^(node_\d+|launcher)\.log$')
        _prev_logs = _glob.glob(os.path.join(_log_dir, "*.log"))
        for _prev_log in _prev_logs:
            _basename = os.path.basename(_prev_log)
            if _session_id not in _basename and not _shell_log_pattern.match(_basename):
                # Previous session's debug log — upload once (policy="now")
                wandb.save(_prev_log, base_path=_orig_cwd, policy="now")
        logger.info(f"Wandb live-syncing debug logs from: {_log_dir} (session={_session_id})")

        # Log config changes to wandb summary if resuming with --force_overwrite
        if _config_changes:
            wandb.run.summary["config_changes_on_resume"] = {
                k: {"old": old, "new": new} for k, (old, new) in _config_changes.items()
            }
            # Also log as a wandb alert for visibility
            _change_text = "\n".join(f"  {k}: '{old}' -> '{new}'" for k, (old, new) in sorted(_config_changes.items()))
            wandb.alert(
                title="Config changed on resume (--force_overwrite)",
                text=f"The following configs were changed:\n{_change_text}",
                level=wandb.AlertLevel.WARN,
            )

    # Log debug schedules
    if is_main_process_per_node():
        logger.info(f"  Eval schedule: {eval_sched}")
        logger.info(f"  Debug schedules: {log_peak_memory_sched}, {log_train_detail_sched}, {check_consistency_sched}, "
                    f"log_train_detail_ppl_threshold={log_train_detail_ppl_threshold}")

    global_step = 0
    running_loss = 0.0
    running_distill_loss = 0.0
    running_regu_sq_norm = 0.0
    running_reset_ratio = 0.0
    running_mean_update_step = 0.0
    # Store per-mb previous repo names on model for repo-change reset tracking
    if not hasattr(model, '_prev_repo_per_mb'):
        model._prev_repo_per_mb = [None] * (local_batch_size // local_micro_batch_size)
    model._running_repo_reset_count = 0  # Accessible from do_train_step
    t_start = time.time()  # Will be adjusted on resume to keep elapsed continuous
    ema_time_per_step = 0.0
    step_start_time = 0.0
    ema_alpha = 0.1  # EMA smoothing factor for time-per-step
    node_rank = get_rank() // parallel_cfg["total_gpus"]
    step_context_tokens = 0       # Context tokens accumulated in current logging window
    step_conv_total_tokens = 0    # Conversation total tokens (non-padding) in current logging window
    step_conv_valid_tokens = 0    # Conversation valid tokens (labels != -100) in current logging window
    total_context_tokens = 0      # Total context tokens trained so far (all nodes)
    total_conv_total_tokens = 0   # Total conversation tokens trained so far (all nodes)
    total_conv_valid_tokens = 0   # Total conversation valid tokens trained so far (all nodes)
    start_epoch = 0               # Epoch to start from (may be overridden by resume)

    # --- [INIT] Resume from checkpoint if available ---
    if resume_checkpoint_dir is not None:
        if load_model_only_flag:
            # Load only model weights from a prior stage's final checkpoint
            # (no optimizer/scheduler state — start fresh training state)
            if is_main_process_per_node():
                logger.info(f"  [Resume] Loading model-only from: {resume_checkpoint_dir}")
            resume_meta = load_model_only(
                model=model,
                checkpoint_dir=resume_checkpoint_dir,
            )
            barrier()
            # Start fresh training state — initialize epoch accumulators to zero
            epoch_loss_sum_resume = 0.0
            epoch_steps_resume = 0
            if is_main_process_per_node():
                _pt_step = resume_meta.get("global_step", "?")
                logger.info(f"  [Resume] Loaded model from step {_pt_step}. "
                             f"Optimizer/scheduler start fresh.")
        else:
            # Normal resume: load full checkpoint (model + optimizer + scheduler + metadata)
            if is_main_process_per_node():
                logger.info(f"  [Resume] Loading checkpoint from: {resume_checkpoint_dir}")
            resume_meta = load_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=resume_checkpoint_dir,
                my_device=my_device,
            )
            # Restore training state from metadata
            global_step = resume_meta.get("global_step", 0)
            start_epoch = resume_meta.get("epoch", 0)
            running_loss = resume_meta.get("running_loss", 0.0)
            epoch_loss_sum_resume = resume_meta.get("epoch_loss_sum", 0.0)
            epoch_steps_resume = resume_meta.get("epoch_steps", 0)
            ema_time_per_step = resume_meta.get("ema_time_per_step", 0.0)
            # Adjust t_start so that elapsed = time.time() - t_start continues
            # from where the previous run left off (seamless wall-clock time)
            saved_elapsed = resume_meta.get("elapsed_time", 0.0)
            if saved_elapsed > 0:
                t_start = time.time() - saved_elapsed
            total_context_tokens = resume_meta.get("total_context_tokens", 0)
            total_conv_total_tokens = resume_meta.get("total_conv_total_tokens", 0)
            total_conv_valid_tokens = resume_meta.get("total_conv_valid_tokens", 0)
            # Update wandb_run_id if not already set from metadata
            if wandb_run_id is None:
                wandb_run_id = resume_meta.get("wandb_run_id", None)
            # Restore dataloader state (generator state for reproducibility)
            dl_state = resume_meta.get("dataloader_state", None)
            if dl_state is not None:
                train_loader.load_state_dict(dl_state)
            barrier()
            if is_main_process_per_node():
                logger.info(
                f"  [Resume] Restored state: global_step={global_step}, "
                f"epoch={start_epoch}, ema_time_per_step={ema_time_per_step:.2f}s"
            )
    else:
        epoch_loss_sum_resume = 0.0
        epoch_steps_resume = 0

    # --- [INIT] Broadcast trainable params from DP rank 0 (first node) ---
    # Ensures all DP replicas start with identical parameters regardless of
    # random initialization differences across nodes.
    if is_main_process_per_node():
        logger.info("  [INIT] Broadcasting trainable params from DP rank 0 to all replicas...")
    broadcast_trainable_params_from_dp_rank0(
        model=model,
        my_device=my_device,
        parallel_cfg=parallel_cfg,
    )

    # --- [DEBUG] Initial parameter consistency check (after broadcast) ---
    if check_consistency_sched.should_run(0):
        if is_main_process_per_node():
            logger.info("  [DEBUG] Running initial DP parameter consistency check (step 0)...")
        check_dp_param_consistency(
            model=model,
            my_device=my_device,
            parallel_cfg=parallel_cfg,
            global_step=0,
        )

    # --- [INIT] NCCL P2P warmup (delegated to parallel-mode-specific module) ---
    _nccl_p2p_warmup(model, parallel_cfg, my_device)

    # ==================================================================
    # 8a. TP Forward Comparison — PP Save mode
    # ==================================================================
    _tp_fwd_compare_cfg = cfg.get("debug", {}).get("tp_forward_compare", None)
    _tp_fwd_compare_mode = _tp_fwd_compare_cfg.get("mode", None) if _tp_fwd_compare_cfg else None
    if _tp_fwd_compare_mode == "pp_save":
        from utils.tp_forward_compare import run_pp_save_mode
        run_pp_save_mode(
            model=model,
            train_loader=train_loader,
            cfg=cfg,
            parallel_cfg=parallel_cfg,
            my_device=my_device,
            make_sdpa_ctx=make_sdpa_ctx,
            local_micro_batch_size=local_micro_batch_size,
            pad_token_id=pad_token_id,
        )
        if is_main_process_per_node():
            logger.info("PP Save mode complete. Exiting.")
        cleanup_distributed()
        os._exit(0)

    # ==================================================================
    # 8b. Evaluation Baseline Mode — base LLM only, no hypernetwork
    # ==================================================================
    _evaluation_baseline = os.environ.get("EVALUATION_BASELINE", "") == "1"
    if _evaluation_baseline:
        if is_main_process_per_node():
            logger.info("=" * 60)
            logger.info("  EVALUATION BASELINE MODE (PP)")
            logger.info("  Running base LLM evaluation (no hypernetwork/LoRA/detach_state)")
            logger.info("=" * 60)

        if val_loader is None:
            if is_main_process_per_node():
                logger.error("  [Baseline] No validation set available. Exiting.")
            cleanup_distributed()
            os._exit(1)

        run_evaluation(
            val_loader=val_loader,
            model=model,
            my_device=my_device,
            parallel_cfg=parallel_cfg,
            make_sdpa_ctx=make_sdpa_ctx,
            global_step=0,
            local_batch_size=eval_local_batch_size,
            local_micro_batch_size=eval_local_micro_batch_size,
            vocab_size=vocab_size,
            context_seq_len=context_seq_len,
            conv_seq_len=conv_seq_len,
            num_mem_token=num_mem_token,
            use_wandb=use_wandb,
            node_rank=node_rank,
            t_start=t_start,
            max_steps=max_steps,
            ema_time_per_step=0.0,
            distill_loss_fn=None,
            distill_micro_batch_size=distill_micro_batch_size,
            baseline_mode=True,
        )

        if is_main_process() and use_wandb:
            import wandb
            wandb.finish()
        if is_main_process_per_node():
            logger.info("  [Baseline] Evaluation baseline complete. Exiting.")
        cleanup_distributed()
        os._exit(0)

    for epoch in range(start_epoch, num_epochs):
        train_loader.set_epoch(epoch)
        data_iter = iter(train_loader)

        micro_step = 0  # counts micro-batches within a grad-accum window
        # If resuming mid-epoch, restore epoch-level accumulators
        if epoch == start_epoch and resume_checkpoint_dir is not None:
            epoch_loss_sum = epoch_loss_sum_resume
            epoch_steps = epoch_steps_resume
            # Skip batches already processed in this epoch
            _batches_to_skip = epoch_steps_resume * grad_accum_steps
            if _batches_to_skip > 0 and is_main_process_per_node():
                logger.info(f"  [Resume] Skipping {_batches_to_skip} batches in epoch {epoch}...")
            for _skip_i in range(_batches_to_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    break
        else:
            epoch_loss_sum = 0.0  # cumulative loss for this epoch
            epoch_steps = 0       # number of optimizer steps in this epoch

        while global_step < max_steps:
            # Update collator training progress for dynamic mask scheduling
            if hasattr(collator, 'set_training_progress'):
                collator.set_training_progress(global_step, max_steps)

            global_step, micro_step, running_loss, running_distill_loss, running_regu_sq_norm, running_reset_ratio, running_mean_update_step, stop_epoch, \
                epoch_loss_sum, epoch_steps, ema_time_per_step, step_start_time, \
                step_context_tokens, step_conv_total_tokens, step_conv_valid_tokens, \
                total_context_tokens, total_conv_total_tokens, total_conv_valid_tokens, \
                loss_running_avg = do_train_step(
                data_iter=data_iter,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                my_device=my_device,
                parallel_cfg=parallel_cfg,
                make_sdpa_ctx=make_sdpa_ctx,
                epoch=epoch,
                global_step=global_step,
                micro_step=micro_step,
                running_loss=running_loss,
                running_distill_loss=running_distill_loss,
                running_regu_sq_norm=running_regu_sq_norm,
                running_reset_ratio=running_reset_ratio,
                running_mean_update_step=running_mean_update_step,
                t_start=t_start,
                grad_accum_steps=grad_accum_steps,
                gradient_clipping=gradient_clipping,
                logging_steps=logging_steps,
                save_sched=save_sched,
                forever_save_steps=forever_save_steps,
                save_total_limit=save_total_limit,
                run_name=run_name,
                max_steps=max_steps,
                log_peak_memory_sched=log_peak_memory_sched,
                log_train_detail_sched=log_train_detail_sched,
                log_train_detail_ppl_threshold=log_train_detail_ppl_threshold,
                local_batch_size=local_batch_size,
                local_micro_batch_size=local_micro_batch_size,
                vocab_size=vocab_size,
                context_seq_len=context_seq_len,
                conv_seq_len=conv_seq_len,
                num_mem_token=num_mem_token,
                use_wandb=use_wandb,
                node_rank=node_rank,
                epoch_loss_sum=epoch_loss_sum,
                epoch_steps=epoch_steps,
                detail_log_accumulator=detail_log_accumulator,
                detail_log_accumulator_ppl=detail_log_accumulator_ppl,
                tokenizer=_detail_tokenizer,
                pad_token_id=pad_token_id,
                check_consistency_sched=check_consistency_sched,
                check_nograd_loradict_sched=check_nograd_loradict_sched,
                ema_time_per_step=ema_time_per_step,
                step_start_time=step_start_time,
                ema_alpha=ema_alpha,
                step_context_tokens=step_context_tokens,
                step_conv_total_tokens=step_conv_total_tokens,
                step_conv_valid_tokens=step_conv_valid_tokens,
                total_context_tokens=total_context_tokens,
                total_conv_total_tokens=total_conv_total_tokens,
                total_conv_valid_tokens=total_conv_valid_tokens,
                wandb_run_id=wandb_run_id,
                # --- P0 Monitor parameters ---
                grad_norm_sched=grad_norm_sched,
                param_norm_sched=param_norm_sched,
                gen_lora_norm_sched=gen_lora_norm_sched,
                monitor_loss_spike=monitor_loss_spike,
                monitor_nan_inf=monitor_nan_inf,
                nan_inf_tracker=nan_inf_tracker,
                nan_inf_stop_steps=nan_inf_stop_steps,
                loss_running_avg=loss_running_avg,
                loss_ema_alpha=ema_alpha,
                # --- Distillation parameters ---
                distill_loss_fn=distill_loss_fn,
                distill_micro_batch_size=distill_micro_batch_size,
                # --- Config tracking for checkpoint ---
                config_selections=_resolved_configs,
                launch_cmd=os.environ.get("LAUNCH_CMD", ""),
            )

            # --- Evaluation ---
            if eval_sched.should_run(global_step) and val_loader is not None:
                # Disable masking during validation
                if hasattr(collator, 'set_eval_mode'):
                    collator.set_eval_mode(True)
                run_evaluation(
                    val_loader=val_loader,
                    model=model,
                    my_device=my_device,
                    parallel_cfg=parallel_cfg,
                    make_sdpa_ctx=make_sdpa_ctx,
                    global_step=global_step,
                    local_batch_size=eval_local_batch_size,
                    local_micro_batch_size=eval_local_micro_batch_size,
                    vocab_size=vocab_size,
                    context_seq_len=context_seq_len,
                    conv_seq_len=conv_seq_len,
                    num_mem_token=num_mem_token,
                    use_wandb=use_wandb,
                    node_rank=node_rank,
                    t_start=t_start,
                    max_steps=max_steps,
                    ema_time_per_step=ema_time_per_step,
                    distill_loss_fn=distill_loss_fn,
                    distill_micro_batch_size=distill_micro_batch_size,
                )
                # Re-enable masking after validation
                if hasattr(collator, 'set_eval_mode'):
                    collator.set_eval_mode(False)

            if stop_epoch:
                break  # end of epoch

        if is_main_process_per_node():
            epoch_micro_batches = epoch_steps * grad_accum_steps
            epoch_avg = epoch_loss_sum / epoch_micro_batches if epoch_micro_batches > 0 else 0.0
            epoch_ppl = math.exp(epoch_avg) if epoch_avg < 20 else float("inf")
            logger.info(
                f"  Epoch {epoch} finished (global_step={global_step}, "
                f"epoch_avg_loss={epoch_avg:.4f}, epoch_avg_ppl={epoch_ppl:.2f})"
            )

        if global_step >= max_steps:
            break

    # ==================================================================
    # 9. Finish — save final checkpoint and cleanup
    # ==================================================================
    # Final evaluation (always run regardless of eval schedule)
    if val_loader is not None:
        # Skip if the last training step already triggered an eval
        _already_evaled = eval_sched.should_run(global_step)
        if not _already_evaled:
            if is_main_process_per_node():
                logger.info(f"  [Final] Running final evaluation at step {global_step}...")
            if hasattr(collator, 'set_eval_mode'):
                collator.set_eval_mode(True)
            run_evaluation(
                val_loader=val_loader,
                model=model,
                my_device=my_device,
                parallel_cfg=parallel_cfg,
                make_sdpa_ctx=make_sdpa_ctx,
                global_step=global_step,
                local_batch_size=eval_local_batch_size,
                local_micro_batch_size=eval_local_micro_batch_size,
                vocab_size=vocab_size,
                context_seq_len=context_seq_len,
                conv_seq_len=conv_seq_len,
                num_mem_token=num_mem_token,
                use_wandb=use_wandb,
                node_rank=node_rank,
                t_start=t_start,
                max_steps=max_steps,
                ema_time_per_step=ema_time_per_step,
                distill_loss_fn=distill_loss_fn,
                distill_micro_batch_size=distill_micro_batch_size,
            )
            if hasattr(collator, 'set_eval_mode'):
                collator.set_eval_mode(False)
        else:
            if is_main_process_per_node():
                logger.info(f"  [Final] Skipping final evaluation (already evaluated at step {global_step})")

    # Save final checkpoint (model-only) for downstream use (e.g. SFT)
    if is_main_process_per_node():
        logger.info(f"  [Final] Saving final checkpoint for run '{run_name}'...")
    final_save_duration = save_final_checkpoint(
        model=model,
        run_name=run_name,
        global_step=global_step,
        epoch=epoch,
    )
    barrier()  # Ensure node 0 finishes saving before other nodes continue
    if is_main_process_per_node():
        logger.info(f"  [Final] Final checkpoint saved in {final_save_duration:.1f}s")

    _report_peak_memory("After training")

    # Flush all debug loggers so final writes are captured by wandb live sync
    flush_debug_loggers(_debug_categories)

    if use_wandb:
        wandb.finish()

    cleanup_distributed()
    if is_main_process_per_node():
        total_time = time.time() - t_start
        logger.info(f"Training complete: {global_step} steps in {format_duration(total_time)}")


if __name__ == "__main__":
    main()
