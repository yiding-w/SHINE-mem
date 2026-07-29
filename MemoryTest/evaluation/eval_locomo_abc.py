#!/usr/bin/env python
"""Run the LoCoMo A/B/C memory-path diagnostic on one or more conversations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from MemoryTest.case_test import (
    build_cfg,
    load_runtime,
    resolve_device,
    resolve_torch_dtype,
)
from MemoryTest.evaluation.locomo_probe import (
    CATEGORY_NAMES,
    select_probe_questions,
    select_probe_session_window,
    summarize_records,
)
from MemoryTest.evaluation.locomo_runtime import (
    LOCOMO_CONDITIONS,
    evaluate_locomo_probe,
)
from MemoryTest.training.lora_sft_utils import load_runtime_args, resolve_path
from MemoryTest.training.shine_train_utils import cast_floating_tensors


LOGGER = logging.getLogger("eval_locomo_abc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare A=direct Qwen context, B=one full-history SHINE write, "
            "C=session-wise recurrent writes, and optionally D=last session only."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--checkpoint-profile",
        choices=("auto", "pretrain", "ift"),
        default="auto",
    )
    parser.add_argument("--locomo-file", required=True)
    parser.add_argument(
        "--output-file",
        default="MemoryTest/checkpoints/locomo_abc/results.json",
    )
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of conversations; 0 evaluates all conversations after sample-start.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=LOCOMO_CONDITIONS,
        default=list(LOCOMO_CONDITIONS),
    )
    parser.add_argument(
        "--categories",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="LoCoMo categories. Category 3 needs external knowledge and is off by default.",
    )
    parser.add_argument("--questions-per-category", type=int, default=5)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=50,
        help="Maximum contiguous sessions per conversation.",
    )
    parser.add_argument(
        "--context-max-length",
        type=int,
        default=None,
        help="Selectable SHINE context budget (for example 4096, 8192, or 25000).",
    )
    parser.add_argument("--conversation-max-length", type=int, default=None)
    parser.add_argument(
        "--direct-context-max-length",
        type=int,
        default=0,
        help="0 automatically uses context budget + question budget.",
    )
    parser.add_argument("--question-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--memory-position-offset", type=int, default=None)
    parser.add_argument(
        "--recurrent-memory-policy",
        choices=("replace", "append"),
        default="replace",
    )
    parser.add_argument("--recurrent-memory-max-banks", type=int, default=1)
    parser.add_argument("--generated-lora-clamp", type=float, default=5.0)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"),
        default="bf16",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate(args: argparse.Namespace, sample_count: int) -> tuple[int, int]:
    if args.sample_start < 0 or args.sample_start >= sample_count:
        raise ValueError(
            f"--sample-start={args.sample_start} is outside {sample_count} conversations"
        )
    if args.num_samples < 0:
        raise ValueError("--num-samples must be non-negative")
    if args.max_sessions < 1:
        raise ValueError("--max-sessions must be at least 1")
    if args.questions_per_category < 0:
        raise ValueError("--questions-per-category must be non-negative")
    invalid_categories = sorted(set(args.categories) - set(CATEGORY_NAMES))
    if invalid_categories:
        raise ValueError(f"Unknown LoCoMo categories: {invalid_categories}")
    stop = (
        sample_count
        if args.num_samples == 0
        else min(sample_count, args.sample_start + args.num_samples)
    )
    return args.sample_start, stop


def _aggregate(per_sample: list[dict], condition_names: list[str]) -> dict:
    records = [
        record
        for sample_summary in per_sample
        for record in sample_summary["records"]
    ]
    conditions = summarize_records(records, condition_names)
    result = {
        "num_samples": len(per_sample),
        "num_questions": len(records),
        "conditions": conditions,
    }
    if "direct_context" in conditions and "single_write" in conditions:
        result["compression_gap"] = (
            conditions["direct_context"]["overall_score"]
            - conditions["single_write"]["overall_score"]
        )
    if "direct_context" in conditions and "evidence_write" in conditions:
        result["evidence_write_gap"] = (
            conditions["direct_context"]["overall_score"]
            - conditions["evidence_write"]["overall_score"]
        )
    if "evidence_write" in conditions and "single_write" in conditions:
        result["long_context_interference_gap"] = (
            conditions["evidence_write"]["overall_score"]
            - conditions["single_write"]["overall_score"]
        )
    if "evidence_write" in conditions and "evidence_session_write" in conditions:
        result["within_session_interference_gap"] = (
            conditions["evidence_write"]["overall_score"]
            - conditions["evidence_session_write"]["overall_score"]
        )
    if "evidence_session_write" in conditions and "single_write" in conditions:
        result["cross_session_interference_gap"] = (
            conditions["evidence_session_write"]["overall_score"]
            - conditions["single_write"]["overall_score"]
        )
    if "single_write" in conditions and "recurrent" in conditions:
        result["recurrent_gap"] = (
            conditions["single_write"]["overall_score"]
            - conditions["recurrent"]["overall_score"]
        )
    if "recurrent" in conditions and "last_session_only" in conditions:
        result["recurrent_gain_over_last_session"] = (
            conditions["recurrent"]["overall_score"]
            - conditions["last_session_only"]["overall_score"]
        )
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    locomo_path = resolve_path(args.locomo_file)
    samples = json.loads(locomo_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"LoCoMo file must contain a non-empty JSON list: {locomo_path}")
    start, stop = _validate(args, len(samples))

    runtime_args = load_runtime_args(resolve_path(args.config))
    runtime_args.checkpoint_dir = args.checkpoint_dir
    runtime_args.device = args.device
    runtime_args.gpu_id = args.gpu_id
    runtime_args.seed = args.seed
    if args.context_max_length is not None:
        runtime_args.context_max_length = args.context_max_length
    if args.conversation_max_length is not None:
        runtime_args.conversation_max_length = args.conversation_max_length
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    cfg.model.torch_dtype = args.torch_dtype
    metanetwork, metalora, tokenizer = load_runtime(
        cfg,
        args.checkpoint_dir,
        device,
        checkpoint_profile=args.checkpoint_profile,
    )
    dtype = resolve_torch_dtype(args.torch_dtype)
    if isinstance(dtype, torch.dtype):
        metanetwork.to(device=device, dtype=dtype)
        metalora = cast_floating_tensors(metalora, dtype)
    if hasattr(metanetwork.metamodel, "config"):
        # Context encoding uses SHINE recurrent K/V rather than Transformers'
        # ordinary autoregressive cache. Generation overrides this explicitly.
        metanetwork.metamodel.config.use_cache = False
    metanetwork.eval()

    eval_args = argparse.Namespace(
        seed=args.seed,
        memory_position_offset=args.memory_position_offset,
        recurrent_memory_policy=args.recurrent_memory_policy,
        recurrent_memory_max_banks=args.recurrent_memory_max_banks,
        generated_lora_clamp=args.generated_lora_clamp,
        locomo_eval_max_new_tokens=args.max_new_tokens,
        locomo_eval_question_max_length=args.question_max_length,
        locomo_eval_direct_context_max_length=args.direct_context_max_length,
        locomo_eval_conditions=args.conditions,
        locomo_eval_last_session_ablation=False,
        locomo_eval_categories=args.categories,
        locomo_eval_questions_per_category=args.questions_per_category,
    )

    per_sample = []
    for sample_index in range(start, stop):
        sample = samples[sample_index]
        sessions = select_probe_session_window(
            sample,
            args.categories,
            args.max_sessions,
        )
        questions = select_probe_questions(
            sample,
            args.categories,
            args.questions_per_category,
            args.seed,
            allowed_session_numbers=set(sessions),
        )
        if not questions:
            LOGGER.warning("Skipping sample %s: no supported questions", sample_index)
            continue
        LOGGER.info(
            "Evaluating sample=%s id=%s sessions=%s questions=%s context_budget=%s",
            sample_index,
            sample.get("sample_id"),
            sessions,
            len(questions),
            cfg.test.context_max_length,
        )
        summary = evaluate_locomo_probe(
            metanetwork,
            metalora,
            tokenizer,
            cfg,
            eval_args,
            device,
            sample,
            questions,
            sessions,
        )
        summary["sample_index"] = sample_index
        per_sample.append(summary)
        LOGGER.info(
            "Sample %s scores: %s",
            sample_index,
            ", ".join(
                f"{name}={value['overall_score']:.4f}"
                for name, value in summary["conditions"].items()
            ),
        )

    if not per_sample:
        raise RuntimeError("No LoCoMo conversations produced evaluable questions")
    aggregate = _aggregate(per_sample, args.conditions)
    result = {
        "config": vars(args),
        "resolved_context_max_length": int(cfg.test.context_max_length),
        "aggregate": aggregate,
        "samples": per_sample,
    }
    output_path = resolve_path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Aggregate scores: %s",
        ", ".join(
            f"{name}={value['overall_score']:.4f}"
            for name, value in aggregate["conditions"].items()
        ),
    )
    LOGGER.info("Saved LoCoMo A/B/C diagnostic to %s", output_path)


if __name__ == "__main__":
    main()
