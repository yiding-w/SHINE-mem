"""Standalone Doc-to-LoRA MemoryAgentBench runner (no benchmark_compare / deltamem.core)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from deltamem.eval.d2l_memory_agent_bench import evaluate_d2l_memory_agent_bench
from deltamem.eval.memory_agent_bench_protocol_light import (
    finalize_distributed,
    init_distributed,
    load_memory_agent_bench,
    local_memory_agent_bench_row_tasks,
    set_all_seeds,
    summarize_memory_agent_bench,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Doc-to-LoRA MemoryAgentBench (δ-mem protocol, light entry).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets-cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets")),
    )
    parser.add_argument(
        "--hub-cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")),
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--memory-agent-bench-max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--memory-agent-bench-use-official-generation-lengths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--memory-agent-bench-use-official-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-temperature", type=float, default=0.4)
    parser.add_argument("--eval-top-p", type=float, default=0.9)
    parser.add_argument("--eval-top-k", type=int, default=10)
    parser.add_argument(
        "--memory-agent-bench-splits",
        nargs="+",
        default=[
            "Accurate_Retrieval",
            "Test_Time_Learning",
            "Long_Range_Understanding",
            "Conflict_Resolution",
        ],
    )
    parser.add_argument("--memory-agent-bench-sources", nargs="*", default=None)
    parser.add_argument("--external-memory-agent-bench-root", type=Path, required=True)
    parser.add_argument("--memory-agent-bench-max-context-chars", type=int, default=120000)
    parser.add_argument("--memory-agent-bench-max-questions-per-row-task", type=int, default=0)
    parser.add_argument("--d2l-root", type=Path, default=None)
    parser.add_argument("--d2l-agent-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.d2l_root is None:
        env_d2l_root = os.environ.get("D2L_ROOT")
        if env_d2l_root:
            args.d2l_root = Path(env_d2l_root)
        else:
            args.d2l_root = args.external_memory_agent_bench_root.parent / "doc-to-lora"
    return args


def main() -> None:
    args = parse_args()
    attn_impl = os.environ.get("D2L_ATTN_IMPLEMENTATION") or os.environ.get("ATTN_IMPLEMENTATION") or "sdpa"
    try:
        from methods.d2l_attn_patch import apply_d2l_attn_patch, flash_attn_available, resolve_d2l_attn

        if flash_attn_available() and os.environ.get("D2L_FORCE_SDPA", "").strip() not in ("1", "true", "yes"):
            attn_impl = os.environ.get("D2L_ATTN_IMPLEMENTATION") or "flash_attention_2"
        apply_d2l_attn_patch(attn_impl)
    except ImportError:
        pass
    context = init_distributed(args.device)
    try:
        set_all_seeds(args.seed)
        args.datasets_cache_dir.mkdir(parents=True, exist_ok=True)
        args.hub_cache_dir.mkdir(parents=True, exist_ok=True)

        items = load_memory_agent_bench(
            cache_dir=args.datasets_cache_dir,
            hub_cache_dir=args.hub_cache_dir,
            splits=args.memory_agent_bench_splits,
            sources=args.memory_agent_bench_sources,
            max_samples=args.max_samples,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        row_tasks = local_memory_agent_bench_row_tasks(
            items,
            context,
            default_max_new_tokens=args.memory_agent_bench_max_new_tokens,
            use_official_generation_lengths=args.memory_agent_bench_use_official_generation_lengths,
            max_questions_per_row_task=args.memory_agent_bench_max_questions_per_row_task,
        )
        progress_total = sum(row_task.question_count for row_task in row_tasks)
        progress_bar = None
        if context.rank == 0:
            progress_bar = tqdm(total=progress_total, desc="d2l_memory_agent_bench", dynamic_ncols=True)

        records = evaluate_d2l_memory_agent_bench(
            row_tasks=row_tasks,
            args=args,
            context=context,
            progress_bar=progress_bar,
        )
        if progress_bar is not None:
            progress_bar.close()

        if context.rank == 0 and records is not None:
            payload: dict[str, object] = {
                "d2l_root": str(args.d2l_root),
                "d2l_agent_config": str(args.d2l_agent_config),
                "seed": args.seed,
                "memory_agent_bench_max_context_chars": args.memory_agent_bench_max_context_chars,
                "eval_temperature": args.eval_temperature,
                "eval_top_p": args.eval_top_p,
                "eval_top_k": args.eval_top_k,
                "d2l": {
                    "d2l_root": str(args.d2l_root),
                    "d2l_agent_config": str(args.d2l_agent_config),
                    "memory_agent_bench": {
                        "records": records,
                        "summary": summarize_memory_agent_bench(records),
                    },
                },
            }
            write_json_atomic(args.output_json, payload)
            print(json.dumps(payload["d2l"], indent=2))
    finally:
        finalize_distributed(context)


if __name__ == "__main__":
    main()
