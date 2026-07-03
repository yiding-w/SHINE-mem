#!/usr/bin/env python3
"""Merge local memory-stream / MEMORY_QA features into upstream meta_train_tp.py."""
from __future__ import annotations

import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZIP_TP = pathlib.Path("/tmp/shine_v2_zip_compare_13463/SHINE_V2-main/meta_train_tp.py")
LOCAL_BAK = ROOT / "meta_train_tp.py.local_bak"
OUT = ROOT / "meta_train_tp.py"


def extract_block(text: str, start_marker: str, end_marker: str) -> str:
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[i:j]


def main() -> None:
    if not ZIP_TP.is_file():
        print(f"Missing upstream file: {ZIP_TP}", file=sys.stderr)
        sys.exit(1)
    if not LOCAL_BAK.is_file():
        print(f"Missing local backup: {LOCAL_BAK}", file=sys.stderr)
        sys.exit(1)

    shutil.copy2(ZIP_TP, OUT)
    text = OUT.read_text(encoding="utf-8")
    local = LOCAL_BAK.read_text(encoding="utf-8")

    # 1) load_model_only_flag: include MEMORY_QA / SQUAD inference modes
    text = text.replace(
        "    load_model_only_flag = False\n",
        extract_block(
            local,
            "    # Inference modes never have/need optimizer state — load weights only.\n",
            "\n\n    # Read resume_from config",
        ),
        1,
    )

    # 2) MEMORY_QA_GEN + SQUAD_QA_GEN early-exit blocks (before section 5.6)
    memory_qa_block = extract_block(
        local,
        "    # ==================================================================\n    # Memory-QA free-form generation",
        "\n\n    # ------------------------------------------------------------------\n    # 5.6 Resolve config selections",
    )
    text = text.replace(
        "\n    # ------------------------------------------------------------------\n    # 5.6 Resolve config selections",
        "\n" + memory_qa_block + "\n    # ------------------------------------------------------------------\n    # 5.6 Resolve config selections",
        1,
    )

    # 3) gen_eval schedule after eval_sched
    gen_eval_block = extract_block(
        local,
        "    # Generation-eval schedule: per-type free-decode accuracy during training.\n",
        "\n\n    # Debug schedules (same as PP)\n",
    )
    text = text.replace(
        "    eval_sched = DebugSchedule(eval_steps_raw, \"eval_steps\")\n\n    # Debug schedules (same as PP)\n",
        "    eval_sched = DebugSchedule(eval_steps_raw, \"eval_steps\")\n\n" + gen_eval_block + "\n    # Debug schedules (same as PP)\n",
        1,
    )

    # 4) train acc counters in training loop init
    text = text.replace(
        "    running_distill_loss = 0.0\n    running_regu_sq_norm = 0.0\n",
        "    running_distill_loss = 0.0\n"
        "    # Train answer-token accuracy window (computed only on to-be-logged steps to\n"
        "    # keep overhead negligible; SUM-reduced over DP in the logging block).\n"
        "    _train_acc_on = os.environ.get(\"MEMORY_QA_TRAIN_ACC\", \"1\") != \"0\"\n"
        "    _train_acc_corr = 0\n"
        "    _train_acc_tot = 0\n"
        "    _train_ans_corr = 0\n"
        "    _train_ans_tot = 0\n"
        "    running_regu_sq_norm = 0.0\n",
        1,
    )

    # 5) save_best_only block before epoch loop
    save_best_block = extract_block(
        local,
        "    # --- save_best_only: keep ONLY the single best-by-val-ppl model (model-only,\n",
        "\n\n    for epoch in range(start_epoch, num_epochs):\n",
    )
    text = text.replace(
        "\n    for epoch in range(start_epoch, num_epochs):\n",
        "\n" + save_best_block + "\n    for epoch in range(start_epoch, num_epochs):\n",
        1,
    )

    # 6) train-step accuracy hooks
    text = text.replace(
        "                _need_detail = (\n"
        "                    (log_train_detail_sched is not None and log_train_detail_sched.should_run(global_step + 1))\n"
        "                    or log_train_detail_ppl_threshold > 0\n"
        "                )\n\n"
        "                accum_loss, stepped, _skipped, _grad_norm_metrics, _per_token_loss, _distill_loss_item, _regu_sq_norm_local = _train_step(\n",
        "                _need_detail = (\n"
        "                    (log_train_detail_sched is not None and log_train_detail_sched.should_run(global_step + 1))\n"
        "                    or log_train_detail_ppl_threshold > 0\n"
        "                )\n"
        "                # Compute train answer-token accuracy only on steps that will be\n"
        "                # logged (logging_steps cadence) -> negligible extra cost.\n"
        "                _train_acc_this = _train_acc_on and logging_steps > 0 and ((global_step + 1) % logging_steps == 0)\n\n"
        "                accum_loss, stepped, _skipped, _grad_norm_metrics, _per_token_loss, _distill_loss_item, _regu_sq_norm_local = _train_step(\n",
        1,
    )
    text = text.replace(
        "                    return_per_token_loss=_need_detail,\n"
        "                )\n"
        "                _accum_regu_sq_norm += _regu_sq_norm_local\n",
        "                    return_per_token_loss=_need_detail,\n"
        "                    return_acc=_train_acc_this,\n"
        "                )\n"
        "                _accum_regu_sq_norm += _regu_sq_norm_local\n",
        1,
    )
    text = text.replace(
        "                if _distill_loss_item is not None:\n"
        "                    _accum_distill_loss += _distill_loss_item\n"
        "                micro_step += 1\n",
        "                if _distill_loss_item is not None:\n"
        "                    _accum_distill_loss += _distill_loss_item\n"
        "                if _train_acc_this:\n"
        "                    _train_acc_corr += int(getattr(model, \"_last_eval_acc_correct\", 0))\n"
        "                    _train_acc_tot += int(getattr(model, \"_last_eval_acc_total\", 0))\n"
        "                    _train_ans_corr += int(getattr(model, \"_last_eval_ans_correct\", 0))\n"
        "                    _train_ans_tot += int(getattr(model, \"_last_eval_ans_total\", 0))\n"
        "                micro_step += 1\n",
        1,
    )

    # 7) save_best_only guard on interval checkpoints
    text = text.replace(
        "                    if save_sched.should_run(global_step):\n",
        "                    if save_sched.should_run(global_step) and not save_best_only:\n",
        1,
    )

    # 8) train acc logging block
    train_acc_log = extract_block(
        local,
        "                        # Train answer-token accuracy: SUM counts over DP only.\n",
        "\n\n                        # All-reduce detach_state metrics across DP replicas\n",
    )
    text = text.replace(
        "\n                        # All-reduce detach_state metrics across DP replicas\n",
        "\n" + train_acc_log + "\n                        # All-reduce detach_state metrics across DP replicas\n",
        1,
    )
    text = text.replace(
        "                            _regu_suffix = \"\"\n"
        "                            if model.detach_state is not None:\n",
        "                            _tr_acc_suffix = \"\"\n"
        "                            if _train_token_acc is not None:\n"
        "                                _tr_acc_suffix = f\",\\ttoken_acc={_train_token_acc:.4f},\\tanswer_acc={_train_answer_acc:.4f}\"\n"
        "                            _regu_suffix = \"\"\n"
        "                            if model.detach_state is not None:\n",
        1,
    )
    text = text.replace(
        "                                f\"epoch={epoch},\\tloss={_global_avg_ce_loss:.4f},\\tppl={_global_ce_ppl:.2f}{_distill_suffix},\\t\"\n",
        "                                f\"epoch={epoch},\\tloss={_global_avg_ce_loss:.4f},\\tppl={_global_ce_ppl:.2f}{_tr_acc_suffix}{_distill_suffix},\\t\"\n",
        1,
    )
    text = text.replace(
        "                        running_repo_reset_count = 0\n\n"
        "                        if is_main_process():\n",
        "                        running_repo_reset_count = 0\n"
        "                        _train_acc_corr = 0\n"
        "                        _train_acc_tot = 0\n"
        "                        _train_ans_corr = 0\n"
        "                        _train_ans_tot = 0\n\n"
        "                        if is_main_process():\n",
        1,
    )

    # 8b) _train_step: add return_acc passthrough (zip base keeps SP grad sync)
    text = text.replace(
        "    return_per_token_loss: bool = False,\n) -> tuple:",
        "    return_per_token_loss: bool = False,\n    return_acc: bool = False,\n) -> tuple:",
        1,
    )
    for old in (
        "                grad_accum_steps=grad_accum_steps,\n            )\n            loss, _per_token_loss, _distill_loss_val = _result",
        "                grad_accum_steps=grad_accum_steps,\n            )\n            loss, _per_token_loss = _result",
        "                grad_accum_steps=grad_accum_steps,\n            )\n            loss, _distill_loss_val = _result",
        "                grad_accum_steps=grad_accum_steps,\n            )\n            loss = _result",
    ):
        text = text.replace(
            old,
            old.replace(
                "                grad_accum_steps=grad_accum_steps,\n            )",
                "                grad_accum_steps=grad_accum_steps,\n                return_acc=return_acc,\n            )",
                1,
            ),
            1,
        )

    # 9) eval: capture output + save_best
    text = text.replace(
        "                        _tp_run_evaluation(\n"
        "                            model=model,\n"
        "                            val_loader=val_loader_for_eval,\n"
        "                            tp_cfg=tp_cfg,\n"
        "                            sdpa_ctx_factory=make_sdpa_ctx,\n"
        "                            global_step=global_step,\n"
        "                            use_wandb=use_wandb,\n"
        "                            t_start=t_start,\n"
        "                            max_steps=max_steps,\n"
        "                            ema_time_per_step=ema_time_per_step,\n"
        "                            distill_loss_fn=distill_loss_fn,\n"
        "                        )\n"
        "                        # Re-enable masking after validation\n"
        "                        if hasattr(collator, 'set_eval_mode'):\n"
        "                            collator.set_eval_mode(False)\n\n"
        "                    # --- Profiler step (no-op when disabled) ---\n",
        "                        _eval_out = _tp_run_evaluation(\n"
        "                            model=model,\n"
        "                            val_loader=val_loader_for_eval,\n"
        "                            tp_cfg=tp_cfg,\n"
        "                            sdpa_ctx_factory=make_sdpa_ctx,\n"
        "                            global_step=global_step,\n"
        "                            use_wandb=use_wandb,\n"
        "                            t_start=t_start,\n"
        "                            max_steps=max_steps,\n"
        "                            ema_time_per_step=ema_time_per_step,\n"
        "                            distill_loss_fn=distill_loss_fn,\n"
        "                        )\n"
        "                        # Re-enable masking after validation\n"
        "                        if hasattr(collator, 'set_eval_mode'):\n"
        "                            collator.set_eval_mode(False)\n"
        "                        if save_best_only and _eval_out is not None:\n"
        "                            _save_best_model(global_step, _eval_out[1])\n\n"
        "                    # Generation eval: per-type free-decode accuracy (+ optional\n"
        "                    # deferred-recall probe). Heavy (greedy decode), so gated on\n"
        "                    # its own schedule. Runs on main rank; others wait at barrier.\n"
        "                    # Main-only decode => only safe at TP=1 (TP>1 forward has\n"
        "                    # collectives that would deadlock); skip under TP>1.\n"
        "                    if (gen_eval_sched.should_run(global_step)\n"
        "                            and tp_cfg[\"tensor_parallel_size\"] == 1):\n"
        "                        from eval_memory_gen import run_memory_qa_gen_inloop\n"
        "                        for _ge_recall, _ge_prefix in ([(False, \"gen\")]\n"
        "                                + ([(True, \"gen_recall\")] if _gen_eval_recall else [])):\n"
        "                            _ge_hit, _ge_tot = run_memory_qa_gen_inloop(\n"
        "                                model, cfg, tp_cfg, my_device,\n"
        "                                recall=_ge_recall, n_hist=_gen_eval_nhist,\n"
        "                            )\n"
        "                            if _ge_tot is not None and is_main_process_per_node():\n"
        "                                logger.info(\n"
        "                                    f\"  [GenEval {_ge_prefix} step {global_step}] \"\n"
        "                                    + \", \".join(f\"{t}={_ge_hit[t]}/{_ge_tot[t]}\" for t in sorted(_ge_tot))\n"
        "                                    + f\", overall={sum(_ge_hit.values())}/{max(1, sum(_ge_tot.values()))}\"\n"
        "                                )\n"
        "                            if _ge_tot is not None and is_main_process() and use_wandb and wandb is not None:\n"
        "                                _ge_metrics = {f\"{_ge_prefix}/acc_{t}\": _ge_hit[t] / max(1, _ge_tot[t]) for t in _ge_tot}\n"
        "                                _ge_metrics[f\"{_ge_prefix}/acc_overall\"] = sum(_ge_hit.values()) / max(1, sum(_ge_tot.values()))\n"
        "                                wandb.log(_ge_metrics, step=global_step)\n\n"
        "                    # --- Profiler step (no-op when disabled) ---\n",
        1,
    )

    # 10) final eval + save_best_only final save
    text = text.replace(
        "            _tp_run_evaluation(\n"
        "                model=model,\n"
        "                val_loader=val_loader_for_eval,\n"
        "                tp_cfg=tp_cfg,\n"
        "                sdpa_ctx_factory=make_sdpa_ctx,\n"
        "                global_step=global_step,\n"
        "                use_wandb=use_wandb,\n"
        "                t_start=t_start,\n"
        "                max_steps=max_steps,\n"
        "                ema_time_per_step=ema_time_per_step,\n"
        "                distill_loss_fn=distill_loss_fn,\n"
        "            )\n"
        "            if hasattr(collator, 'set_eval_mode'):\n"
        "                collator.set_eval_mode(False)\n"
        "        else:\n",
        "            _eval_out = _tp_run_evaluation(\n"
        "                model=model,\n"
        "                val_loader=val_loader_for_eval,\n"
        "                tp_cfg=tp_cfg,\n"
        "                sdpa_ctx_factory=make_sdpa_ctx,\n"
        "                global_step=global_step,\n"
        "                use_wandb=use_wandb,\n"
        "                t_start=t_start,\n"
        "                max_steps=max_steps,\n"
        "                ema_time_per_step=ema_time_per_step,\n"
        "                distill_loss_fn=distill_loss_fn,\n"
        "            )\n"
        "            if hasattr(collator, 'set_eval_mode'):\n"
        "                collator.set_eval_mode(False)\n"
        "            if save_best_only and _eval_out is not None:\n"
        "                _save_best_model(global_step, _eval_out[1])\n"
        "        else:\n",
        1,
    )
    text = text.replace(
        "    # Save final checkpoint (model-only, for downstream annealing/SFT)\n"
        "    if is_main_process_per_node():\n"
        "        logger.info(f\"  [Final] Saving final checkpoint for run '{run_name}'...\")\n"
        "    if is_main_process():\n",
        "    # Save final checkpoint (model-only, for downstream annealing/SFT).\n"
        "    # In save_best_only mode the best model is ALREADY at final/model — don't\n"
        "    # overwrite it with the last-step model.\n"
        "    if save_best_only:\n"
        "        if is_main_process_per_node():\n"
        "            logger.info(f\"  [Final] save_best_only: keeping best model (val_ppl={best_val_ppl:.4f}) \"\n"
        "                        f\"at final/model; skipping last-step final save.\")\n"
        "    elif is_main_process_per_node():\n"
        "        logger.info(f\"  [Final] Saving final checkpoint for run '{run_name}'...\")\n"
        "    if not save_best_only and is_main_process():\n",
        1,
    )

    OUT.write_text(text, encoding="utf-8")
    print(f"Merged -> {OUT}")


if __name__ == "__main__":
    main()
