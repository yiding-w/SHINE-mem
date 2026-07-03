#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Detail Logging Utility

Provides per-token detail logging during training, similar to the dataset
debug output in mydatasets/, but augmented with per-token loss values.

Output format (per sample):
    Idx | Conv Token | Conv ID | Label Token | Label ID | Loss | PPL | Note

This is called from the training loop on the main GPU (stage 0 / local_rank 0)
of each node, after per-token loss has been gathered from the norm stage.
"""

import math
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified debug schedule: supports -1 (disabled), int N (every N steps),
# or list of specific steps.
# ---------------------------------------------------------------------------

class DebugSchedule:
    """
    Unified schedule for debug actions.

    Accepts three input formats (from YAML config):
      - ``-1``          → disabled (never fires)
      - positive int N  → fires every N optimizer steps
      - list of ints    → fires at exactly those steps (e.g. [0, 1, 3, 5])

    Usage::

        sched = DebugSchedule(cfg.debug.log_peak_memory_steps, "log_peak_memory_steps")
        if sched.should_run(global_step):
            ...

    The ``0`` step has special meaning: "before training starts".
    For interval mode, step 0 never fires (since ``0 % N == 0`` would always
    be true, but we treat 0 as a pre-training hook only when explicitly listed).
    """

    __slots__ = ("_mode", "_interval", "_steps", "_name")

    def __init__(self, raw_value, name: str = ""):
        """
        Parse a raw config value into a DebugSchedule.

        Args:
            raw_value: The value from YAML config. Can be int, list, or
                       an OmegaConf ListConfig.
            name: Human-readable name for error messages.
        """
        self._name = name

        # Convert OmegaConf containers to plain Python types
        from omegaconf import ListConfig
        if isinstance(raw_value, ListConfig):
            raw_value = list(raw_value)

        if isinstance(raw_value, list):
            # List mode: fire at specific steps
            if not all(isinstance(v, int) and v >= 0 for v in raw_value):
                raise ValueError(
                    f"debug.{name}: list values must be non-negative integers, "
                    f"got {raw_value}"
                )
            self._mode = "list"
            self._interval = 0
            self._steps = set(raw_value)
        elif isinstance(raw_value, int):
            if raw_value == -1:
                # Disabled
                self._mode = "disabled"
                self._interval = 0
                self._steps = set()
            elif raw_value > 0:
                # Interval mode
                self._mode = "interval"
                self._interval = raw_value
                self._steps = set()
            else:
                raise ValueError(
                    f"debug.{name}: integer value must be -1 (disabled) or "
                    f"a positive integer (interval), got {raw_value}"
                )
        else:
            raise ValueError(
                f"debug.{name}: expected int, list of ints, or -1, "
                f"got {type(raw_value).__name__}: {raw_value}"
            )

    @property
    def enabled(self) -> bool:
        """Return True if this schedule can ever fire."""
        return self._mode != "disabled"

    @property
    def mode(self) -> str:
        """Return the mode: 'disabled', 'interval', or 'list'."""
        return self._mode

    def should_run(self, step: int) -> bool:
        """
        Return True if the debug action should run at the given step.

        For interval mode, step 0 is excluded (use explicit list [0, ...]
        to include pre-training checks).
        """
        if self._mode == "disabled":
            return False
        elif self._mode == "interval":
            return step > 0 and step % self._interval == 0
        else:  # list
            return step in self._steps

    def __repr__(self) -> str:
        if self._mode == "disabled":
            return f"DebugSchedule({self._name}=disabled)"
        elif self._mode == "interval":
            return f"DebugSchedule({self._name}=every {self._interval} steps)"
        else:
            sorted_steps = sorted(self._steps)
            return f"DebugSchedule({self._name}=steps {sorted_steps})"


def _format_token(tokenizer, token_id: int, max_width: int = 20) -> str:
    """Decode a single token id and format for display."""
    tok_str = tokenizer.decode([token_id], skip_special_tokens=False)
    # Use repr for whitespace-only tokens
    if tok_str.strip() == "":
        tok_str = repr(tok_str)
    # Truncate if too long
    if len(tok_str) > max_width:
        tok_str = tok_str[:max_width - 1] + "…"
    return tok_str


def log_training_detail(
    detail_log_accumulator: List[Dict[str, Any]],
    tokenizer,
    global_step: int,
    epoch: int,
    num_mem_token: int = 0,
    pad_token_id: int = 0,
    logger_name: str = "debug.training_detail",
):
    """
    Log detailed per-token training information for all micro-batches
    accumulated during this optimizer step.

    Each micro-batch dict contains:
        - context_ids:      (micro_B, ctx_total_len) on CPU
        - conversation_ids: (micro_B, conv_len) on CPU
        - labels:           (micro_B, conv_len) on CPU
        - context_lengths:  (micro_B,) on CPU
        - per_token_loss:   (micro_B * (conv_len-1),) or (micro_B, conv_len-1) on CPU, or None

    Output is printed via logger.info on the main process per node.

    Args:
        detail_log_accumulator: List of dicts from accumulated micro-batches.
        tokenizer: HuggingFace tokenizer for decoding token ids.
        global_step: Current optimizer step number.
        epoch: Current epoch number.
        num_mem_token: Number of memory token placeholders.
        pad_token_id: Token id used for padding.
    """
    if not detail_log_accumulator:
        return

    sep = "=" * 130
    lines = []
    lines.append(f"\n{sep}")
    lines.append(f"  TRAINING DETAIL — Step {global_step}, Epoch {epoch}")
    lines.append(f"  Micro-batches: {len(detail_log_accumulator)}")
    lines.append(sep)

    sample_idx = 0  # global sample counter across micro-batches

    for mb_idx, mb_data in enumerate(detail_log_accumulator):
        context_ids = mb_data["context_ids"]        # (B, ctx_total_len)
        conversation_ids = mb_data["conversation_ids"]  # (B, conv_len)
        labels = mb_data["labels"]                  # (B, conv_len)
        context_lengths = mb_data["context_lengths"]  # (B,)
        per_token_loss = mb_data["per_token_loss"]  # (B*(conv_len-1),) or (B, conv_len-1) or None

        batch_size = conversation_ids.size(0)
        conv_len = conversation_ids.size(1)
        ctx_total_len = context_ids.size(0) if context_ids.dim() == 1 else context_ids.size(1)

        # Reshape per_token_loss to (B, conv_len-1) if available
        if per_token_loss is not None:
            per_token_loss_2d = per_token_loss.view(batch_size, conv_len - 1)
        else:
            per_token_loss_2d = None

        # Resolve <MASK> token id once per micro-batch (same tokenizer throughout)
        _mask_token_id = None
        try:
            _mask_enc = tokenizer.encode("<MASK>", add_special_tokens=False)
            if len(_mask_enc) == 1:
                _mask_token_id = _mask_enc[0]
        except Exception:
            pass

        def _fmt_ctx_tok(cid):
            if _mask_token_id is not None and cid == _mask_token_id:
                return f"[MASK]({cid})"
            return f"{_format_token(tokenizer, cid, 15)}({cid})"

        for bi in range(batch_size):
            ctx_ids = context_ids[bi]       # (ctx_total_len,)
            conv_ids = conversation_ids[bi]  # (conv_len,)
            lbl = labels[bi]                # (conv_len,)
            ctx_len = context_lengths[bi].item()

            lines.append(f"\n{'─' * 130}")
            lines.append(
                f"  Sample {sample_idx} (micro_batch={mb_idx}, batch_idx={bi}, "
                f"context_valid_tokens={ctx_len})"
            )
            lines.append(f"{'─' * 130}")

            # --- Context section (abbreviated) ---
            # Count <MASK> tokens in context
            ctx_valid = ctx_ids[:ctx_len]
            if _mask_token_id is not None:
                num_masks = int((ctx_valid == _mask_token_id).sum().item())
                mask_ratio = num_masks / max(ctx_len, 1)
                lines.append(
                    f"  [Context] length={ctx_len}, total_with_mem={ctx_ids.size(0)}, "
                    f"masked={num_masks}/{ctx_len} ({mask_ratio:.1%})"
                )
            else:
                lines.append(f"  [Context] length={ctx_len}, total_with_mem={ctx_ids.size(0)}")

            # Show first 5 and last 5 context tokens, highlight <MASK>
            ctx_preview_n = min(5, ctx_len)
            if ctx_len > 0:
                ctx_start_tokens = [_fmt_ctx_tok(ctx_ids[i].item()) for i in range(ctx_preview_n)]
                ctx_str = " | ".join(ctx_start_tokens)
                if ctx_len > 2 * ctx_preview_n:
                    ctx_end_tokens = [_fmt_ctx_tok(ctx_ids[i].item()) for i in range(ctx_len - ctx_preview_n, ctx_len)]
                    ctx_str += " | ... | " + " | ".join(ctx_end_tokens)
                lines.append(f"    {ctx_str}")

            # --- Conversation section with per-token loss ---
            lines.append(f"  [Conversation] length={conv_len}")

            # Column widths
            w_idx = 5
            w_tok = 20
            w_id = 8
            w_loss = 10
            w_ppl = 10
            w_note = 15

            # TP mode: per_token_loss is None, note this in header
            _tp_no_loss = (per_token_loss_2d is None)
            _loss_col_header = "Loss(N/A)" if _tp_no_loss else "Loss"
            _ppl_col_header  = "PPL(N/A)"  if _tp_no_loss else "PPL"

            header = (
                f"{'Idx':>{w_idx}} | "
                f"{'Ctx Token':<{w_tok}} | {'CtxID':>{w_id}} | "
                f"{'Conv Token':<{w_tok}} | {'ConvID':>{w_id}} | "
                f"{'Label Token':<{w_tok}} | {'LblID':>{w_id}} | "
                f"{_loss_col_header:>{w_loss}} | {_ppl_col_header:>{w_ppl}} | {'Note':<{w_note}}"
            )
            lines.append(f"    {header}")
            lines.append(f"    {'-' * len(header)}")

            # Track loss statistics for this sample
            valid_losses = []
            # Track valid label token count (for TP mode where loss is unavailable)
            num_valid_label_tokens = 0

            for pos in range(conv_len):
                notes = []
                # context_ids[:ctx_len] aligns 1-to-1 with conversation_ids
                ctx_id = ctx_ids[pos].item() if pos < ctx_ids.size(0) else -1
                conv_id = conv_ids[pos].item()
                label_id = lbl[pos].item()

                # Ctx token (masked input) — highlight <MASK> positions
                if _mask_token_id is not None and ctx_id == _mask_token_id:
                    ctx_tok = _format_token(tokenizer, ctx_id, w_tok)
                    notes.append("[MASK]")
                else:
                    ctx_tok = _format_token(tokenizer, ctx_id, w_tok) if ctx_id >= 0 else ""

                # Conv token (unmasked original)
                conv_tok = _format_token(tokenizer, conv_id, w_tok)

                # Label token
                if label_id == -100:
                    label_tok = ""
                    label_id_str = "-100"
                    notes.append("[MASKED]")
                else:
                    label_tok = _format_token(tokenizer, label_id, w_tok)
                    label_id_str = str(label_id)

                # Per-token loss (shifted: loss[pos] corresponds to predicting token at pos+1)
                # per_token_loss_2d[bi, pos] = loss for predicting label at position pos+1
                # So for display at position pos, we show the loss of predicting THIS position
                # which is per_token_loss_2d[bi, pos-1]
                loss_str = ""
                ppl_str = ""
                if label_id != -100:
                    num_valid_label_tokens += 1
                if per_token_loss_2d is not None and pos > 0:
                    # loss at index (pos-1) predicts token at position pos
                    token_loss = per_token_loss_2d[bi, pos - 1].item()
                    if label_id != -100:
                        loss_str = f"{token_loss:.4f}"
                        token_ppl = math.exp(token_loss) if token_loss < 20 else float("inf")
                        ppl_str = f"{token_ppl:.2f}" if token_ppl < 10000 else "inf"
                        valid_losses.append(token_loss)
                    elif token_loss > 0:
                        # Masked but has loss (shouldn't happen with ignore_index=-100)
                        loss_str = f"({token_loss:.4f})"

                # Note for special tokens
                if conv_id == pad_token_id and label_id == -100:
                    if "[MASKED]" in notes:
                        notes = ["[PAD]"]
                    else:
                        notes.append("[PAD]")

                note_str = " ".join(notes)

                lines.append(
                    f"    {pos:>{w_idx}} | "
                    f"{ctx_tok:<{w_tok}} | {ctx_id:>{w_id}} | "
                    f"{conv_tok:<{w_tok}} | {conv_id:>{w_id}} | "
                    f"{label_tok:<{w_tok}} | {label_id_str:>{w_id}} | "
                    f"{loss_str:>{w_loss}} | {ppl_str:>{w_ppl}} | {note_str:<{w_note}}"
                )

            # Sample-level statistics (always output)
            lines.append(f"    {'─' * 80}")
            if valid_losses:
                avg_loss = sum(valid_losses) / len(valid_losses)
                avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                max_loss = max(valid_losses)
                min_loss = min(valid_losses)
                lines.append(
                    f"    Sample stats: avg_loss={avg_loss:.4f}, avg_ppl={avg_ppl:.2f}, "
                    f"max_loss={max_loss:.4f}, min_loss={min_loss:.4f}, "
                    f"num_valid_tokens={len(valid_losses)}"
                )
            else:
                # per-token loss not available (e.g. TP mode or non-detail step)
                lines.append(
                    f"    Sample stats: avg_loss=N/A, avg_ppl=N/A, "
                    f"num_valid_label_tokens={num_valid_label_tokens}, "
                    f"num_total_tokens={conv_len}"
                )

            # --- Distillation section ---
            distill = mb_data.get("distill")
            if distill is not None and isinstance(distill, dict):
                distill_conv_ids = distill["conversation_ids"][bi]  # (conv_len,)
                distill_lbl = distill["labels"][bi]                 # (conv_len,)

                lines.append(f"\n  [Distill] conversation & labels for Sample {sample_idx - 1}")

                d_header = (
                    f"{'Idx':>{w_idx}} | "
                    f"{'Distill Conv':<{w_tok}} | {'ID':>{w_id}} | "
                    f"{'Distill Label':<{w_tok}} | {'LblID':>{w_id}} | {'Note':<{w_note}}"
                )
                lines.append(f"    {d_header}")
                lines.append(f"    {'-' * len(d_header)}")

                for pos in range(distill_conv_ids.size(0)):
                    d_notes = []
                    d_conv_id = distill_conv_ids[pos].item()
                    d_label_id = distill_lbl[pos].item()

                    d_conv_tok = _format_token(tokenizer, d_conv_id, w_tok)

                    if d_label_id == -100:
                        d_label_tok = ""
                        d_label_id_str = "-100"
                        d_notes.append("[MASKED]")
                    else:
                        d_label_tok = _format_token(tokenizer, d_label_id, w_tok)
                        d_label_id_str = str(d_label_id)

                    if d_conv_id == pad_token_id and d_label_id == -100:
                        d_notes = ["[PAD]"]

                    d_note_str = " ".join(d_notes)

                    lines.append(
                        f"    {pos:>{w_idx}} | "
                        f"{d_conv_tok:<{w_tok}} | {d_conv_id:>{w_id}} | "
                        f"{d_label_tok:<{w_tok}} | {d_label_id_str:>{w_id}} | {d_note_str:<{w_note}}"
                    )

            sample_idx += 1

    lines.append(f"\n{sep}")
    lines.append(f"  TRAINING DETAIL END — {sample_idx} samples logged")
    lines.append(sep)

    # Output all at once via training_detail debug logger
    full_output = "\n".join(lines)
    _detail_logger = logging.getLogger(logger_name)
    _detail_logger.info(full_output)


# ---------------------------------------------------------------------------
# Broadcast trainable parameters from DP rank 0 to all other DP replicas
# ---------------------------------------------------------------------------

@torch.no_grad()
def broadcast_trainable_params_from_dp_rank0(model, my_device, parallel_cfg):
    """
    Broadcast all trainable parameters (hypernetwork + metalora) from DP rank 0
    (first node) to all other DP replicas, ensuring identical initial parameters.

    This is a collective operation: ALL ranks in the DP group must call it.
    """
    from utils.myparallel import is_main_process_per_node
    from utils.myloradict import collect_loradict_tensors

    group = parallel_cfg.get("dp_process_group")
    if group is None:
        return  # DP size == 1, nothing to broadcast
    dp_size = dist.get_world_size(group)
    if dp_size <= 1:
        return

    # The global rank of DP rank 0 for this stage is the stage index itself.
    # In our topology: stage s on node 0 has global_rank = s.
    # (node 0's ranks are [0, 1, ..., total_gpus-1], and stage == local_rank)
    src_global_rank = parallel_cfg["stage"]

    # Broadcast hypernetwork parameters
    num_hyper_params = 0
    for name, param in model.hypernetwork.named_parameters():
        if param.requires_grad and param.device == my_device:
            dist.broadcast(param.data, src=src_global_rank, group=group)
            num_hyper_params += 1

    # Broadcast metalora tensors
    num_metalora_tensors = 0
    if hasattr(model, 'metalora') and model.metalora is not None:
        metalora_tensors = collect_loradict_tensors(model.metalora)
        for t in metalora_tensors:
            if t.requires_grad and t.device == my_device:
                dist.broadcast(t.data, src=src_global_rank, group=group)
                num_metalora_tensors += 1

    # Broadcast mem_tokens (lives on the embed stage only)
    broadcast_mem = False
    mem = getattr(model.llm.model, "mem_tokens", None)
    if mem is not None and mem.device == my_device:
        dist.broadcast(mem.data, src=src_global_rank, group=group)
        broadcast_mem = True

    if is_main_process_per_node():
        logger.info(
            f"  [INIT] Broadcast trainable params from DP rank 0 to all replicas: "
            f"stage {parallel_cfg['stage']}, "
            f"{num_hyper_params} hypernetwork params + "
            f"{num_metalora_tensors} metalora tensors + "
            f"mem_tokens={'yes' if broadcast_mem else 'no'}"
        )


# ---------------------------------------------------------------------------
# DEBUG: verify cross-node parameter consistency after optimizer step
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_dp_param_consistency(
    model,
    my_device: torch.device,
    parallel_cfg: dict,
    global_step: int,
):
    """
    [DEBUG] Check that trainable parameters are identical across all DP replicas.

    For each trainable parameter (hypernetwork + metalora) on this stage,
    compute a fingerprint (sum + L2 norm) and all_gather across the DP group.
    If any replica differs, collect the mismatch info.

    All stages send their results to stage 0 (via the node_process_group),
    and stage 0 prints a unified report.

    This function is a collective operation: ALL ranks in the DP group must
    call it simultaneously (otherwise it will deadlock).
    """
    from utils.myparallel import is_main_process_per_node
    from utils.myloradict import collect_loradict_tensors

    group = parallel_cfg.get("dp_process_group")
    if group is None:
        return  # DP size == 1, nothing to check
    dp_size = dist.get_world_size(group)
    if dp_size <= 1:
        return

    dp_rank = parallel_cfg["data_parallel_rank"]
    stage = parallel_cfg["stage"]
    total_stages = parallel_cfg["total_stages"]
    node_group = parallel_cfg.get("node_process_group")

    # Collect all trainable tensors on this device
    named_tensors = []

    # Hypernetwork parameters
    for name, param in model.hypernetwork.named_parameters():
        if param.requires_grad and param.device == my_device:
            named_tensors.append((f"hypernetwork.{name}", param.data))

    # Metalora tensors
    if hasattr(model, 'metalora') and model.metalora is not None:
        metalora_tensors = collect_loradict_tensors(model.metalora)
        for i, t in enumerate(metalora_tensors):
            if t.requires_grad and t.device == my_device:
                named_tensors.append((f"metalora.tensor_{i}", t.data))

    # W-transform parameters (replicated across TP ranks and DP replicas)
    for wt_name in ('w_transform_context', 'w_transform_conversation'):
        wt_module = getattr(model, wt_name, None)
        if wt_module is not None:
            for name, param in wt_module.named_parameters():
                if param.requires_grad and param.device == my_device:
                    named_tensors.append((f"{wt_name}.{name}", param.data))

    # For each tensor, compute (sum, l2_norm) as fingerprint
    # Then all_gather across DP group and compare
    mismatches = []
    num_params = len(named_tensors)
    for name, tensor in named_tensors:
        t_float = tensor.float()
        local_sum = t_float.sum()
        local_norm = t_float.norm()

        # Gather fingerprints from all DP replicas
        local_fp = torch.tensor([local_sum.item(), local_norm.item()],
                                dtype=torch.float64, device=my_device)
        gathered = [torch.zeros(2, dtype=torch.float64, device=my_device)
                    for _ in range(dp_size)]
        dist.all_gather(gathered, local_fp, group=group)

        # Compare: all should be identical (or very close due to fp precision)
        ref_sum, ref_norm = gathered[0][0].item(), gathered[0][1].item()
        for r in range(1, dp_size):
            r_sum, r_norm = gathered[r][0].item(), gathered[r][1].item()
            sum_diff = abs(r_sum - ref_sum) / (abs(ref_sum) + 1e-12)
            norm_diff = abs(r_norm - ref_norm) / (abs(ref_norm) + 1e-12)
            if sum_diff > 1e-5 or norm_diff > 1e-5:
                mismatches.append({
                    "name": name,
                    "dp_rank_0": (ref_sum, ref_norm),
                    "dp_rank": r,
                    "values": (r_sum, r_norm),
                    "sum_diff": sum_diff,
                    "norm_diff": norm_diff,
                })

    # --- Gather results from all stages to stage 0 via node_process_group ---
    # Encode local result as a tensor: [stage, num_params, num_mismatches]
    local_summary = torch.tensor(
        [float(stage), float(num_params), float(len(mismatches))],
        dtype=torch.float64, device=my_device
    )

    if node_group is not None:
        all_summaries = [torch.zeros(3, dtype=torch.float64, device=my_device)
                         for _ in range(total_stages)]
        dist.all_gather(all_summaries, local_summary, group=node_group)
    else:
        all_summaries = [local_summary]

    # --- Stage 0 prints unified report ---
    if stage == 0 and is_main_process_per_node():
        all_passed = True
        report_lines = []
        for s_summary in all_summaries:
            s_stage = int(s_summary[0].item())
            s_num = int(s_summary[1].item())
            s_mis = int(s_summary[2].item())
            if s_mis > 0:
                all_passed = False
                report_lines.append(
                    f"    Stage {s_stage}: {s_mis}/{s_num} params MISMATCH"
                )
            else:
                if s_num > 0:
                    report_lines.append(
                        f"    Stage {s_stage}: {s_num} params OK ✓"
                    )
                # stages with 0 params are skipped

        _dpc_logger = logging.getLogger("debug.dp_consistency")
        if all_passed:
            _dpc_logger.info(
                f"  [DEBUG] DP param consistency check PASSED at step {global_step} "
                f"(all stages, {dp_size} DP replicas):\n" +
                "\n".join(report_lines)
            )
        else:
            _dpc_logger.warning(
                f"\n{'!'*70}\n"
                f"  [DEBUG] DP PARAMETER MISMATCH DETECTED at step {global_step}!\n"
                f"  Summary across all stages ({dp_size} DP replicas):\n" +
                "\n".join(report_lines) + f"\n{'!'*70}"
            )

        # Print detailed mismatches from stage 0 (this stage)
        if mismatches:
            _dpc_logger.warning(
                f"  Stage {stage} detailed mismatches "
                f"({len(mismatches)}/{num_params}):"
            )
            for m in mismatches[:10]:
                _dpc_logger.warning(
                    f"    {m['name']}: "
                    f"dp_rank_0=(sum={m['dp_rank_0'][0]:.8e}, norm={m['dp_rank_0'][1]:.8e}), "
                    f"dp_rank_{m['dp_rank']}=(sum={m['values'][0]:.8e}, norm={m['values'][1]:.8e}), "
                    f"sum_rel_diff={m['sum_diff']:.2e}, norm_rel_diff={m['norm_diff']:.2e}"
                )
            if len(mismatches) > 10:
                _dpc_logger.warning(f"    ... and {len(mismatches) - 10} more mismatches")

    # --- Non-stage-0 processes with mismatches: print locally ---
    if stage != 0 and mismatches and dp_rank == 0:
        _dpc_logger = logging.getLogger("debug.dp_consistency")
        _dpc_logger.warning(
            f"  [DEBUG] Stage {stage} has {len(mismatches)}/{num_params} "
            f"param mismatches at step {global_step} (details below):"
        )
        for m in mismatches[:5]:
            _dpc_logger.warning(
                f"    {m['name']}: "
                f"dp_rank_0=(sum={m['dp_rank_0'][0]:.8e}, norm={m['dp_rank_0'][1]:.8e}), "
                f"dp_rank_{m['dp_rank']}=(sum={m['values'][0]:.8e}, norm={m['values'][1]:.8e}), "
                f"sum_rel_diff={m['sum_diff']:.2e}, norm_rel_diff={m['norm_diff']:.2e}"
            )
        if len(mismatches) > 5:
            _dpc_logger.warning(f"    ... and {len(mismatches) - 5} more mismatches")


@torch.no_grad()
def check_tp_param_consistency(
    model,
    my_device: torch.device,
    tp_cfg: dict,
    global_step: int,
):
    """
    [DEBUG] Check that replicated parameters are identical across all TP ranks.

    W-transform parameters (L, R, MLP weights, gate) are replicated across TP
    ranks and must be bit-identical for correct all-reduce semantics. This
    function verifies that by computing fingerprints and all_gathering across
    the TP group.

    This function is a collective operation: ALL ranks in the TP group must
    call it simultaneously (otherwise it will deadlock).
    """
    from utils.myparallel import is_main_process_per_node

    tp_group = tp_cfg.get("tp_process_group")
    if tp_group is None or dist.get_world_size(tp_group) <= 1:
        return
    tp_size = dist.get_world_size(tp_group)
    tp_rank = tp_cfg["tp_rank"]

    # Collect all replicated tensors that must be identical across TP ranks
    named_tensors = []

    # Hypernetwork parameters (replicated across TP)
    for name, param in model.hypernetwork.named_parameters():
        if param.requires_grad and param.device == my_device:
            named_tensors.append((f"hypernetwork.{name}", param.data))

    # W-transform parameters (replicated across TP)
    for wt_name in ('w_transform_context', 'w_transform_conversation'):
        wt_module = getattr(model, wt_name, None)
        if wt_module is not None:
            for name, param in wt_module.named_parameters():
                if param.requires_grad and param.device == my_device:
                    named_tensors.append((f"{wt_name}.{name}", param.data))

    # For each tensor, compute (sum, l2_norm) as fingerprint
    # Then all_gather across TP group and compare
    mismatches = []
    num_params = len(named_tensors)
    for name, tensor in named_tensors:
        t_float = tensor.float()
        local_sum = t_float.sum()
        local_norm = t_float.norm()

        # Gather fingerprints from all TP ranks
        local_fp = torch.tensor([local_sum.item(), local_norm.item()],
                                dtype=torch.float64, device=my_device)
        gathered = [torch.zeros(2, dtype=torch.float64, device=my_device)
                    for _ in range(tp_size)]
        dist.all_gather(gathered, local_fp, group=tp_group)

        # Compare: all should be identical
        ref_sum, ref_norm = gathered[0][0].item(), gathered[0][1].item()
        for r in range(1, tp_size):
            r_sum, r_norm = gathered[r][0].item(), gathered[r][1].item()
            sum_diff = abs(r_sum - ref_sum) / (abs(ref_sum) + 1e-12)
            norm_diff = abs(r_norm - ref_norm) / (abs(ref_norm) + 1e-12)
            if sum_diff > 1e-5 or norm_diff > 1e-5:
                mismatches.append({
                    "name": name,
                    "tp_rank_0": (ref_sum, ref_norm),
                    "tp_rank": r,
                    "values": (r_sum, r_norm),
                    "sum_diff": sum_diff,
                    "norm_diff": norm_diff,
                })

    # --- Report (only tp_rank 0 prints) ---
    _tpc_logger = logging.getLogger("debug.tp_consistency")
    if tp_rank == 0:
        if not mismatches:
            _tpc_logger.info(
                f"  [DEBUG] TP param consistency check PASSED at step {global_step} "
                f"({num_params} replicated params, {tp_size} TP ranks)"
            )
        else:
            _tpc_logger.warning(
                f"\n{'!'*70}\n"
                f"  [DEBUG] TP PARAMETER MISMATCH DETECTED at step {global_step}!\n"
                f"  {len(mismatches)}/{num_params} params differ across {tp_size} TP ranks:\n"
                f"{'!'*70}"
            )
            for m in mismatches[:10]:
                _tpc_logger.warning(
                    f"    {m['name']}: "
                    f"tp_rank_0=(sum={m['tp_rank_0'][0]:.8e}, norm={m['tp_rank_0'][1]:.8e}), "
                    f"tp_rank_{m['tp_rank']}=(sum={m['values'][0]:.8e}, norm={m['values'][1]:.8e}), "
                    f"sum_rel_diff={m['sum_diff']:.2e}, norm_rel_diff={m['norm_diff']:.2e}"
                )
            if len(mismatches) > 10:
                _tpc_logger.warning(f"    ... and {len(mismatches) - 10} more mismatches")


def check_sp_param_consistency(
    model,
    my_device: torch.device,
    tp_cfg: dict,
    global_step: int,
):
    """
    [DEBUG] Check that replicated parameters are identical across all SP ranks.

    When SP > 1, all SP ranks within a group must hold bit-identical trainable
    parameters (hypernetwork, metalora, w_transform). This function verifies
    that by computing fingerprints and all_gathering across the SP group.

    This function is a collective operation: ALL ranks in the SP group must
    call it simultaneously (otherwise it will deadlock).
    """
    from utils.myparallel import is_main_process_per_node

    sp_group = tp_cfg.get("sp_process_group")
    sp_size = tp_cfg.get("sequence_parallel_size", 1)
    if sp_group is None or sp_size <= 1:
        return
    sp_rank = tp_cfg.get("sp_rank", 0)

    # Collect all replicated tensors that must be identical across SP ranks
    named_tensors = []

    # Hypernetwork parameters (replicated across SP)
    for name, param in model.hypernetwork.named_parameters():
        if param.requires_grad and param.device == my_device:
            named_tensors.append((f"hypernetwork.{name}", param.data))

    # Metalora tensors (replicated across SP)
    if hasattr(model, 'metalora') and model.metalora is not None:
        from utils.myloradict import collect_loradict_tensors
        for i, t in enumerate(collect_loradict_tensors(model.metalora)):
            if t.requires_grad and t.device == my_device:
                named_tensors.append((f"metalora.tensor_{i}", t.data))

    # W-transform parameters (replicated across SP)
    for wt_name in ('w_transform_context', 'w_transform_conversation'):
        wt_module = getattr(model, wt_name, None)
        if wt_module is not None:
            for name, param in wt_module.named_parameters():
                if param.requires_grad and param.device == my_device:
                    named_tensors.append((f"{wt_name}.{name}", param.data))

    # For each tensor, compute (sum, l2_norm) as fingerprint
    # Then all_gather across SP group and compare
    mismatches = []
    num_params = len(named_tensors)
    for name, tensor in named_tensors:
        t_float = tensor.float()
        local_sum = t_float.sum()
        local_norm = t_float.norm()

        # Gather fingerprints from all SP ranks
        local_fp = torch.tensor([local_sum.item(), local_norm.item()],
                                dtype=torch.float64, device=my_device)
        gathered = [torch.zeros(2, dtype=torch.float64, device=my_device)
                    for _ in range(sp_size)]
        dist.all_gather(gathered, local_fp, group=sp_group)

        # Compare: all should be identical
        ref_sum, ref_norm = gathered[0][0].item(), gathered[0][1].item()
        for r in range(1, sp_size):
            r_sum, r_norm = gathered[r][0].item(), gathered[r][1].item()
            sum_diff = abs(r_sum - ref_sum) / (abs(ref_sum) + 1e-12)
            norm_diff = abs(r_norm - ref_norm) / (abs(ref_norm) + 1e-12)
            if sum_diff > 1e-5 or norm_diff > 1e-5:
                mismatches.append({
                    "name": name,
                    "sp_rank_0": (ref_sum, ref_norm),
                    "sp_rank": r,
                    "values": (r_sum, r_norm),
                    "sum_diff": sum_diff,
                    "norm_diff": norm_diff,
                })

    # --- Report (only sp_rank 0 prints) ---
    _spc_logger = logging.getLogger("debug.sp_consistency")
    if sp_rank == 0:
        if not mismatches:
            _spc_logger.info(
                f"  [DEBUG] SP param consistency check PASSED at step {global_step} "
                f"({num_params} replicated params, {sp_size} SP ranks)"
            )
        else:
            _spc_logger.warning(
                f"\n{'!'*70}\n"
                f"  [DEBUG] SP PARAMETER MISMATCH DETECTED at step {global_step}!\n"
                f"  {len(mismatches)}/{num_params} params differ across {sp_size} SP ranks:\n"
                f"{'!'*70}"
            )
            for m in mismatches[:10]:
                _spc_logger.warning(
                    f"    {m['name']}: "
                    f"sp_rank_0=(sum={m['sp_rank_0'][0]:.8e}, norm={m['sp_rank_0'][1]:.8e}), "
                    f"sp_rank_{m['sp_rank']}=(sum={m['values'][0]:.8e}, norm={m['values'][1]:.8e}), "
                    f"sum_rel_diff={m['sum_diff']:.2e}, norm_rel_diff={m['norm_diff']:.2e}"
                )
            if len(mismatches) > 10:
                _spc_logger.warning(f"    ... and {len(mismatches) - 10} more mismatches")


# ---------------------------------------------------------------------------
# P0 Monitor: Gradient Norm Computation
# ---------------------------------------------------------------------------

def _collect_metalora_named_tensors(metalora, my_device):
    """
    Recursively collect all tensors from metalora with structured names.

    Returns list of (name, tensor) tuples, e.g.:
        ("layer4_q_query_A", tensor), ("layer4_q_query_B", tensor),
        ("layer4_mlp_gate_A", tensor), ...

    Only returns tensors that require_grad, have grad, and are on my_device.
    """
    results = []
    if metalora is None:
        return results
    for layer_idx, layer_dict in sorted(metalora.items()):
        if layer_dict is None:
            continue
        for module_name, module_dict in layer_dict.items():
            # module_name: "attention" or "mlp"
            if module_dict is None:
                continue
            for param_name, leaf_dict in module_dict.items():
                # param_name: "q_query", "q_gate", "k", "v", "o", "gate", "up", "down"
                if leaf_dict is None:
                    continue
                for tensor_key, tensor in leaf_dict.items():
                    # tensor_key: "A", "B", "C"
                    if tensor is None:
                        continue
                    if not tensor.requires_grad or tensor.grad is None:
                        continue
                    if tensor.device != my_device:
                        continue
                    name = f"layer{layer_idx}_{param_name}_{tensor_key}"
                    results.append((name, tensor))
    return results


def _parse_hypernetwork_param_name(raw_name: str):
    """
    Parse a hypernetwork named_parameter name into (layer_idx_or_None, short_name).

    Examples:
        "m2p_transformer.layers.0.self_attn.q_proj.weight"
            -> (0, "q_proj_weight")
        "m2p_transformer.layers.2.mlp.gate_proj.weight"
            -> (2, "mlp_gate_proj_weight")
        "m2p_transformer.layers.1.input_layernorm.weight"
            -> (1, "input_layernorm_weight")
        "m2p_transformer.norm.weight"
            -> (None, "norm_weight")
        "m2p_transformer.norm_gate_proj.weight"
            -> (None, "norm_gate_proj_weight")
        "layer_pos_emb"
            -> (None, "layer_pos_emb")
        "token_pos_emb"
            -> (None, "token_pos_emb")
    """
    parts = raw_name.split(".")

    # m2p_transformer.layers.{i}.{rest}
    if len(parts) >= 4 and parts[0] == "m2p_transformer" and parts[1] == "layers":
        layer_idx = int(parts[2])
        rest = parts[3:]
        # Simplify: remove "self_attn" prefix for attention params
        if rest[0] == "self_attn":
            rest = rest[1:]
        short = "_".join(rest)
        return layer_idx, short

    # m2p_transformer.norm.weight, m2p_transformer.norm_gate_proj.weight, etc.
    if parts[0] == "m2p_transformer":
        short = "_".join(parts[1:])
        return None, short

    # layer_pos_emb, token_pos_emb
    return None, raw_name


@torch.no_grad()
def compute_grad_norms(
    model,
    my_device: torch.device,
) -> Dict[str, float]:
    """
    Compute structured gradient RMS (root mean square) for hypernetwork and
    metalora parameters on the current pipeline stage (pre-clipping).
    RMS = L2_norm / sqrt(numel), which is scale-invariant to parameter size.

    Naming convention:
        Hypernetwork:
            grad_norm/hyper/layer{i}/{short_name}       — per-param norm
            grad_norm/hyper/layer_avg/layer{i}          — per-layer average norm
            grad_norm/hyper/param_avg/{param_type}      — per-param-type average across layers
            grad_norm/hyper/global/{short_name}         — non-layer params (norm, pos_emb)
            grad_norm/hyper_avg                         — total average norm
        Metalora:
            grad_norm/meta/layer{i}/{param_name}        — per-param norm
            grad_norm/meta/layer_avg/layer{i}           — per-layer average norm
            grad_norm/meta/param_avg/{param_type}       — per-param-type average across layers
            grad_norm/meta_avg                          — total average norm

    Returns a dict of metric_name -> norm_value.
    """
    metrics: Dict[str, float] = {}

    # ===================== Hypernetwork =====================
    # Collect per-layer and global params
    hyper_layer_norms: Dict[int, Dict[str, float]] = {}  # layer_idx -> {short_name: norm}
    hyper_global_norms: Dict[str, float] = {}  # non-layer params
    hyper_all_norms: list = []  # all norms for total average

    # Batch norm computation: collect all norms as scalar tensors, then
    # transfer to CPU in one shot to minimize GPU→CPU sync overhead.
    _hyper_norm_tensors = []
    _hyper_meta_info = []  # (layer_idx_or_None, short_name, numel)
    for raw_name, param in model.hypernetwork.named_parameters():
        if not param.requires_grad or param.grad is None or param.device != my_device:
            continue
        _hyper_norm_tensors.append(param.grad.norm(2).float())
        layer_idx, short_name = _parse_hypernetwork_param_name(raw_name)
        _hyper_meta_info.append((layer_idx, short_name, param.grad.numel()))

    if _hyper_norm_tensors:
        _hyper_norms_cpu = torch.stack(_hyper_norm_tensors).cpu().tolist()
    else:
        _hyper_norms_cpu = []

    for i, (layer_idx, short_name, numel) in enumerate(_hyper_meta_info):
        grad_norm = _hyper_norms_cpu[i] / max(numel ** 0.5, 1.0)
        if layer_idx is not None:
            if layer_idx not in hyper_layer_norms:
                hyper_layer_norms[layer_idx] = {}
            hyper_layer_norms[layer_idx][short_name] = grad_norm
            metrics[f"grad_norm/hyper/layer{layer_idx}/{short_name}"] = grad_norm
        else:
            hyper_global_norms[short_name] = grad_norm
            metrics[f"grad_norm/hyper/global/{short_name}"] = grad_norm
        hyper_all_norms.append(grad_norm)

    # Per-layer average
    for layer_idx, param_norms in sorted(hyper_layer_norms.items()):
        if param_norms:
            avg = sum(param_norms.values()) / len(param_norms)
            metrics[f"grad_norm/hyper/layer_avg/layer{layer_idx}"] = avg

    # Per-param-type average across layers
    param_type_norms: Dict[str, list] = {}
    for layer_idx, param_norms in hyper_layer_norms.items():
        for short_name, norm_val in param_norms.items():
            if short_name not in param_type_norms:
                param_type_norms[short_name] = []
            param_type_norms[short_name].append(norm_val)
    for param_type, norms in param_type_norms.items():
        metrics[f"grad_norm/hyper/param_avg/{param_type}"] = sum(norms) / len(norms)

    # Total average
    if hyper_all_norms:
        metrics["grad_norm/hyper_avg"] = sum(hyper_all_norms) / len(hyper_all_norms)
    else:
        metrics["grad_norm/hyper_avg"] = 0.0

    # ===================== Metalora =====================
    meta_named = _collect_metalora_named_tensors(
        getattr(model, 'metalora', None), my_device)

    meta_layer_norms: Dict[int, Dict[str, float]] = {}  # layer_idx -> {param_name: norm}
    meta_all_norms: list = []

    # Batch norm computation for metalora (single GPU→CPU sync)
    _meta_norm_tensors = []
    _meta_meta_info = []  # (layer_idx, param_name, numel)
    for name, tensor in meta_named:
        _meta_norm_tensors.append(tensor.grad.norm(2).float())
        parts = name.split("_", 1)
        layer_idx = int(parts[0].replace("layer", ""))
        param_name = parts[1] if len(parts) > 1 else name
        _meta_meta_info.append((layer_idx, param_name, tensor.grad.numel()))

    if _meta_norm_tensors:
        _meta_norms_cpu = torch.stack(_meta_norm_tensors).cpu().tolist()
    else:
        _meta_norms_cpu = []

    for i, (layer_idx, param_name, numel) in enumerate(_meta_meta_info):
        grad_norm = _meta_norms_cpu[i] / max(numel ** 0.5, 1.0)
        if layer_idx not in meta_layer_norms:
            meta_layer_norms[layer_idx] = {}
        meta_layer_norms[layer_idx][param_name] = grad_norm
        metrics[f"grad_norm/meta/layer{layer_idx}/{param_name}"] = grad_norm
        meta_all_norms.append(grad_norm)

    # Per-layer average
    for layer_idx, param_norms in sorted(meta_layer_norms.items()):
        if param_norms:
            avg = sum(param_norms.values()) / len(param_norms)
            metrics[f"grad_norm/meta/layer_avg/layer{layer_idx}"] = avg

    # Per-param-type average across layers
    meta_param_type_norms: Dict[str, list] = {}
    for layer_idx, param_norms in meta_layer_norms.items():
        for param_name, norm_val in param_norms.items():
            if param_name not in meta_param_type_norms:
                meta_param_type_norms[param_name] = []
            meta_param_type_norms[param_name].append(norm_val)
    for param_type, norms in meta_param_type_norms.items():
        metrics[f"grad_norm/meta/param_avg/{param_type}"] = sum(norms) / len(norms)

    # Total average
    if meta_all_norms:
        metrics["grad_norm/meta_avg"] = sum(meta_all_norms) / len(meta_all_norms)
    else:
        metrics["grad_norm/meta_avg"] = 0.0

    # ===================== W-Transform =====================
    for wt_name in ('w_transform_context', 'w_transform_conversation'):
        wt_module = getattr(model, wt_name, None)
        if wt_module is None:
            continue
        _wt = wt_module._orig_mod if hasattr(wt_module, '_orig_mod') else wt_module
        wt_all_norms = []
        _wt_norm_tensors = []
        _wt_meta_info = []  # (param_name, numel)
        for pname, param in _wt.named_parameters():
            if not param.requires_grad or param.grad is None or param.device != my_device:
                continue
            _wt_norm_tensors.append(param.grad.norm(2).float())
            _wt_meta_info.append((pname, param.grad.numel()))

        if _wt_norm_tensors:
            _wt_norms_cpu = torch.stack(_wt_norm_tensors).cpu().tolist()
        else:
            _wt_norms_cpu = []

        short_wt = wt_name.replace('w_transform_', 'wt_')  # wt_context / wt_conversation
        for i, (pname, numel) in enumerate(_wt_meta_info):
            grad_norm = _wt_norms_cpu[i] / max(numel ** 0.5, 1.0)
            metrics[f"grad_norm/{short_wt}/{pname}"] = grad_norm
            wt_all_norms.append(grad_norm)

        if wt_all_norms:
            metrics[f"grad_norm/{short_wt}_avg"] = sum(wt_all_norms) / len(wt_all_norms)
        else:
            metrics[f"grad_norm/{short_wt}_avg"] = 0.0

    return metrics


@torch.no_grad()
def compute_post_clip_grad_norm(
    model,
    my_device: torch.device,
) -> float:
    """
    Compute the total gradient norm AFTER clipping for all trainable
    parameters (hypernetwork + metalora + w_transform) on the current
    pipeline stage.

    Returns:
        Post-clip total gradient norm (float).
    """
    from utils.myloradict import collect_loradict_tensors

    # Batch norm computation: collect all squared norms as scalar tensors,
    # then transfer to CPU in one shot to minimize GPU→CPU sync overhead.
    _norm_sq_tensors = []

    for param in model.hypernetwork.parameters():
        if param.requires_grad and param.grad is not None and param.device == my_device:
            _norm_sq_tensors.append(param.grad.norm(2).float() ** 2)

    if hasattr(model, 'metalora') and model.metalora is not None:
        metalora_tensors = collect_loradict_tensors(model.metalora)
        for t in metalora_tensors:
            if t.requires_grad and t.grad is not None and t.device == my_device:
                _norm_sq_tensors.append(t.grad.norm(2).float() ** 2)

    # Include w_transform parameters
    for wt_name in ('w_transform_context', 'w_transform_conversation'):
        wt_module = getattr(model, wt_name, None)
        if wt_module is not None:
            _wt = wt_module._orig_mod if hasattr(wt_module, '_orig_mod') else wt_module
            for param in _wt.parameters():
                if param.requires_grad and param.grad is not None and param.device == my_device:
                    _norm_sq_tensors.append(param.grad.norm(2).float() ** 2)

    if not _norm_sq_tensors:
        return 0.0
    total_norm_sq = torch.stack(_norm_sq_tensors).sum().cpu().item()
    return total_norm_sq ** 0.5


# ---------------------------------------------------------------------------
# P0 Monitor: Parameter (Weight) Norm Computation
# ---------------------------------------------------------------------------

def _collect_metalora_named_data(metalora, my_device):
    """
    Recursively collect all trainable tensors from metalora with structured
    names for weight norm computation.

    Returns list of (name, tensor) tuples, e.g.:
        ("layer4_q_query_A", tensor), ("layer4_q_query_B", tensor), ...

    Only returns tensors that require_grad and are on my_device.
    """
    results = []
    if metalora is None:
        return results
    for layer_idx, layer_dict in sorted(metalora.items()):
        if layer_dict is None:
            continue
        for module_name, module_dict in layer_dict.items():
            if module_dict is None:
                continue
            for param_name, leaf_dict in module_dict.items():
                if leaf_dict is None:
                    continue
                for tensor_key, tensor in leaf_dict.items():
                    if tensor is None:
                        continue
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if not tensor.requires_grad:
                        continue
                    if tensor.device != my_device:
                        continue
                    name = f"layer{layer_idx}_{param_name}_{tensor_key}"
                    results.append((name, tensor))
    return results


@torch.no_grad()
def compute_param_norms(
    model,
    my_device: torch.device,
) -> Dict[str, float]:
    """
    Compute structured weight RMS (root mean square) for hypernetwork and
    metalora parameters on the current pipeline stage.
    RMS = L2_norm / sqrt(numel), which is scale-invariant to parameter size.

    Same naming convention as compute_grad_norms but with prefix "param_norm/"
    instead of "grad_norm/":
        param_norm/hyper/layer{i}/{short_name}
        param_norm/hyper/layer_avg/layer{i}
        param_norm/hyper/param_avg/{param_type}
        param_norm/hyper/global/{short_name}
        param_norm/hyper_avg
        param_norm/meta/layer{i}/{param_name}
        param_norm/meta/layer_avg/layer{i}
        param_norm/meta/param_avg/{param_type}
        param_norm/meta_avg

    Returns a dict of metric_name -> norm_value.
    """
    metrics: Dict[str, float] = {}

    # ===================== Hypernetwork =====================
    hyper_layer_norms: Dict[int, Dict[str, float]] = {}
    hyper_global_norms: Dict[str, float] = {}
    hyper_all_norms: list = []

    # Batch norm computation: collect all norms as scalar tensors, then
    # transfer to CPU in one shot to minimize GPU→CPU sync overhead.
    _hyper_norm_tensors = []
    _hyper_meta_info = []  # (layer_idx_or_None, short_name, numel)
    for raw_name, param in model.hypernetwork.named_parameters():
        if not param.requires_grad or param.device != my_device:
            continue
        _hyper_norm_tensors.append(param.data.norm(2).float())
        layer_idx, short_name = _parse_hypernetwork_param_name(raw_name)
        _hyper_meta_info.append((layer_idx, short_name, param.data.numel()))

    if _hyper_norm_tensors:
        _hyper_norms_cpu = torch.stack(_hyper_norm_tensors).cpu().tolist()
    else:
        _hyper_norms_cpu = []

    for i, (layer_idx, short_name, numel) in enumerate(_hyper_meta_info):
        w_norm = _hyper_norms_cpu[i] / max(numel ** 0.5, 1.0)
        if layer_idx is not None:
            if layer_idx not in hyper_layer_norms:
                hyper_layer_norms[layer_idx] = {}
            hyper_layer_norms[layer_idx][short_name] = w_norm
            metrics[f"param_norm/hyper/layer{layer_idx}/{short_name}"] = w_norm
        else:
            hyper_global_norms[short_name] = w_norm
            metrics[f"param_norm/hyper/global/{short_name}"] = w_norm
        hyper_all_norms.append(w_norm)

    for layer_idx, param_norms in sorted(hyper_layer_norms.items()):
        if param_norms:
            metrics[f"param_norm/hyper/layer_avg/layer{layer_idx}"] = (
                sum(param_norms.values()) / len(param_norms))

    param_type_norms: Dict[str, list] = {}
    for layer_idx, param_norms in hyper_layer_norms.items():
        for short_name, norm_val in param_norms.items():
            if short_name not in param_type_norms:
                param_type_norms[short_name] = []
            param_type_norms[short_name].append(norm_val)
    for param_type, norms in param_type_norms.items():
        metrics[f"param_norm/hyper/param_avg/{param_type}"] = sum(norms) / len(norms)

    metrics["param_norm/hyper_avg"] = (
        sum(hyper_all_norms) / len(hyper_all_norms) if hyper_all_norms else 0.0)

    # ===================== Metalora =====================
    meta_named = _collect_metalora_named_data(
        getattr(model, 'metalora', None), my_device)

    meta_layer_norms: Dict[int, Dict[str, float]] = {}
    meta_all_norms: list = []

    # Batch norm computation for metalora (single GPU→CPU sync)
    _meta_norm_tensors = []
    _meta_meta_info = []  # (layer_idx, param_name, numel)
    for name, tensor in meta_named:
        _meta_norm_tensors.append(tensor.data.norm(2).float())
        parts = name.split("_", 1)
        layer_idx = int(parts[0].replace("layer", ""))
        param_name = parts[1] if len(parts) > 1 else name
        _meta_meta_info.append((layer_idx, param_name, tensor.data.numel()))

    if _meta_norm_tensors:
        _meta_norms_cpu = torch.stack(_meta_norm_tensors).cpu().tolist()
    else:
        _meta_norms_cpu = []

    for i, (layer_idx, param_name, numel) in enumerate(_meta_meta_info):
        w_norm = _meta_norms_cpu[i] / max(numel ** 0.5, 1.0)
        if layer_idx not in meta_layer_norms:
            meta_layer_norms[layer_idx] = {}
        meta_layer_norms[layer_idx][param_name] = w_norm
        metrics[f"param_norm/meta/layer{layer_idx}/{param_name}"] = w_norm
        meta_all_norms.append(w_norm)

    for layer_idx, param_norms in sorted(meta_layer_norms.items()):
        if param_norms:
            metrics[f"param_norm/meta/layer_avg/layer{layer_idx}"] = (
                sum(param_norms.values()) / len(param_norms))

    meta_param_type_norms: Dict[str, list] = {}
    for layer_idx, param_norms in meta_layer_norms.items():
        for param_name, norm_val in param_norms.items():
            if param_name not in meta_param_type_norms:
                meta_param_type_norms[param_name] = []
            meta_param_type_norms[param_name].append(norm_val)
    for param_type, norms in meta_param_type_norms.items():
        metrics[f"param_norm/meta/param_avg/{param_type}"] = sum(norms) / len(norms)

    metrics["param_norm/meta_avg"] = (
        sum(meta_all_norms) / len(meta_all_norms) if meta_all_norms else 0.0)

    return metrics


@torch.no_grad()
def compute_generated_lora_norms(
    model,
    my_device: torch.device,
) -> Dict[str, float]:
    """
    Compute weight RMS (root mean square) for the hypernetwork-generated
    LoRA parameters that are cached on the current pipeline stage after
    the last forward pass.
    RMS = L2_norm / sqrt(numel), which is scale-invariant to parameter size.

    The generated loradict is stored in model._last_generated_loradict
    (set during forward). If not available, returns empty dict.

    Naming convention:
        gen_lora_norm/layer{i}/{param_name}
        gen_lora_norm/layer_avg/layer{i}
        gen_lora_norm/param_avg/{param_type}
        gen_lora_norm/avg

    Returns a dict of metric_name -> norm_value.
    """
    loradict = getattr(model, '_last_generated_loradict', None)
    if loradict is None:
        return {}

    metrics: Dict[str, float] = {}
    layer_norms: Dict[int, Dict[str, float]] = {}
    all_norms: list = []

    # Batch norm computation: collect all norms as scalar tensors, then
    # transfer to CPU in one shot to minimize GPU→CPU sync overhead.
    _norm_tensors = []
    _norm_meta_info = []  # (layer_idx, full_name, numel)

    for layer_idx, layer_dict in sorted(loradict.items()):
        if layer_dict is None:
            continue
        # layer_dict has same structure as metalora: {"attention": {...}, "mlp": {...}}
        if not isinstance(layer_dict, dict):
            continue
        for module_name, module_dict in layer_dict.items():
            if module_dict is None or not isinstance(module_dict, dict):
                continue
            for param_name, leaf_dict in module_dict.items():
                if leaf_dict is None or not isinstance(leaf_dict, dict):
                    continue
                for tensor_key, tensor in leaf_dict.items():
                    if tensor is None or not isinstance(tensor, torch.Tensor):
                        continue
                    if tensor.device != my_device:
                        continue
                    _norm_tensors.append(tensor.detach().norm(2).float())
                    full_name = f"{param_name}_{tensor_key}"
                    _norm_meta_info.append((layer_idx, full_name, tensor.numel()))

    if _norm_tensors:
        _norms_cpu = torch.stack(_norm_tensors).cpu().tolist()
    else:
        _norms_cpu = []

    for i, (layer_idx, full_name, numel) in enumerate(_norm_meta_info):
        w_norm = _norms_cpu[i] / max(numel ** 0.5, 1.0)
        if layer_idx not in layer_norms:
            layer_norms[layer_idx] = {}
        layer_norms[layer_idx][full_name] = w_norm
        metrics[f"gen_lora_norm/layer{layer_idx}/{full_name}"] = w_norm
        all_norms.append(w_norm)

    # Per-layer average
    for layer_idx, param_norms in sorted(layer_norms.items()):
        if param_norms:
            metrics[f"gen_lora_norm/layer_avg/layer{layer_idx}"] = (
                sum(param_norms.values()) / len(param_norms))

    # Per-param-type average across layers
    param_type_norms: Dict[str, list] = {}
    for layer_idx, param_norms in layer_norms.items():
        for pname, norm_val in param_norms.items():
            if pname not in param_type_norms:
                param_type_norms[pname] = []
            param_type_norms[pname].append(norm_val)
    for ptype, norms in param_type_norms.items():
        metrics[f"gen_lora_norm/param_avg/{ptype}"] = sum(norms) / len(norms)

    # Total average
    metrics["gen_lora_norm/avg"] = (
        sum(all_norms) / len(all_norms) if all_norms else 0.0)

    return metrics


# ---------------------------------------------------------------------------
# P0 Monitor: Loss Spike Detection
# ---------------------------------------------------------------------------

def compute_loss_spike_metrics(
    gathered_metrics: List[Dict],
    running_avg_loss: float,
    spike_threshold: float = 3.0,
) -> Dict[str, float]:
    """
    Compute loss spike detection metrics from gathered per-node losses.

    Args:
        gathered_metrics: List of dicts from all DP replicas, each containing
            "avg_loss" key.
        running_avg_loss: Exponential moving average of loss so far.
        spike_threshold: A step is flagged as spike if
            loss > spike_threshold * running_avg_loss.

    Returns:
        Dict with keys:
            "monitor/loss_max": max loss across nodes
            "monitor/loss_min": min loss across nodes
            "monitor/loss_std": std of losses across nodes
            "monitor/loss_spike": 1.0 if spike detected, 0.0 otherwise
    """
    all_losses = [m["avg_loss"] for m in gathered_metrics]
    avg_loss = sum(all_losses) / len(all_losses)

    loss_max = max(all_losses)
    loss_min = min(all_losses)
    if len(all_losses) > 1:
        mean = sum(all_losses) / len(all_losses)
        variance = sum((x - mean) ** 2 for x in all_losses) / len(all_losses)
        loss_std = variance ** 0.5
    else:
        loss_std = 0.0

    # Spike detection: loss > threshold * running_avg (only if running_avg > 0)
    is_spike = 0.0
    if running_avg_loss > 0 and avg_loss > spike_threshold * running_avg_loss:
        is_spike = 1.0

    return {
        "monitor/loss_max": loss_max,
        "monitor/loss_min": loss_min,
        "monitor/loss_std": loss_std,
        "monitor/loss_spike": is_spike,
    }


# ---------------------------------------------------------------------------
# P0 Monitor: NaN/Inf Detection
# ---------------------------------------------------------------------------

class NanInfTracker:
    """
    Tracks NaN/Inf occurrences across training steps.

    Usage::

        tracker = NanInfTracker()

        # In training loop:
        is_bad = tracker.check_and_record(loss_value, global_step)
        if is_bad:
            optimizer.zero_grad()
            continue  # skip this step

        # When logging to wandb:
        metrics = tracker.get_wandb_metrics()
        wandb.log(metrics, step=global_step)
    """

    def __init__(self):
        self.nan_count: int = 0
        self.inf_count: int = 0
        self.total_steps: int = 0
        self.consecutive_bad: int = 0

    def check_and_record(self, loss_value: float, global_step: int) -> bool:
        """
        Check if loss is NaN or Inf, update counters.

        Args:
            loss_value: The loss value to check.
            global_step: Current training step (for logging).

        Returns:
            True if loss is NaN or Inf (caller should skip this step).
        """
        self.total_steps += 1
        is_nan = math.isnan(loss_value)
        is_inf = math.isinf(loss_value)

        if is_nan or is_inf:
            if is_nan:
                self.nan_count += 1
            if is_inf:
                self.inf_count += 1
            self.consecutive_bad += 1
            kind = "NaN" if is_nan else "Inf"
            logger.warning(
                f"  [MONITOR] {kind} loss detected at step {global_step} "
                f"(value={loss_value}, nan_count={self.nan_count}, "
                f"inf_count={self.inf_count}, consecutive={self.consecutive_bad}). "
                f"Skipping this step."
            )
            return True
        else:
            self.consecutive_bad = 0
            return False

    def get_wandb_metrics(self) -> Dict[str, float]:
        """
        Return wandb-loggable metrics for NaN/Inf tracking.

        Returns:
            Dict with keys:
                "monitor/nan_count": cumulative NaN count
                "monitor/inf_count": cumulative Inf count
                "monitor/nan_inf_rate": fraction of steps with NaN/Inf
                "monitor/consecutive_bad": current consecutive bad steps
        """
        rate = (self.nan_count + self.inf_count) / max(self.total_steps, 1)
        return {
            "monitor/nan_count": float(self.nan_count),
            "monitor/inf_count": float(self.inf_count),
            "monitor/nan_inf_rate": rate,
            "monitor/consecutive_bad": float(self.consecutive_bad),
        }


# ---------------------------------------------------------------------------
# nograd_loradict integrity check
# ---------------------------------------------------------------------------

def log_nograd_loradict_check(
    nograd_loradict,
    global_step: int,
    debug_logger_name: str = "debug.nograd_loradict",
) -> bool:
    """Check that all tensors in nograd_loradict have no gradient and no grad_fn.

    This function performs a strict check (method 2):
      - requires_grad must be False
      - grad_fn must be None (tensor must be properly detached from computation graph)

    Results are logged to the per-node debug log file.

    Args:
        nograd_loradict: The nograd_loradict to check. Can be None.
        global_step: Current optimizer step (for logging).
        debug_logger_name: Logger name for output.

    Returns:
        True if all checks passed (or nograd_loradict is None), False if any violations found.
    """
    import logging
    from utils.myloradict import check_nograd_loradict

    dbg_logger = logging.getLogger(debug_logger_name)

    if nograd_loradict is None:
        dbg_logger.warning(
            f"[Step {global_step}] nograd_loradict is None — nothing to check. "
            f"If you expect nograd_loradict to be active, this indicates a bug in loradict generation."
        )
        return True

    errors = check_nograd_loradict(nograd_loradict)

    if not errors:
        dbg_logger.info(f"[Step {global_step}] nograd_loradict check PASSED — all tensors have no grad.")
        return True
    else:
        dbg_logger.error(
            f"[Step {global_step}] nograd_loradict check FAILED — "
            f"{len(errors)} tensor(s) have unexpected gradient state:"
        )
        for err in errors:
            dbg_logger.error(err)
        return False
