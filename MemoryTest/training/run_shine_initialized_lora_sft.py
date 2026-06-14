#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from pathlib import Path

import torch

from MemoryTest.evaluation.eval_shine_memory import answer_with_lora, load_shine
from MemoryTest.evaluation.forced_choice import unavailable_forced_choice_result
from MemoryTest.evaluation.metrics import make_eval_row, relation_breakdown, summarize_examples, wrong_examples
from MemoryTest.prepare_data.prompt_templates import build_context
from MemoryTest.training.lora_sft_utils import (
    make_lora_trainable,
    move_lora_to_cpu,
    read_facts,
    resolve_mixed_context_format,
    resolve_path,
    select_fact_subset,
    summarize_mean_std,
    train_lora_dict,
)
from MemoryTest.training.shine_train_utils import cast_floating_tensors


LOGGER = logging.getLogger("run_shine_initialized_lora_sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SHINE generated LoRA as initialization, then do a small amount of direct LoRA SFT/NTP."
    )
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="SHINE checkpoint used to generate the initial LoRA.")
    parser.add_argument("--facts-path", "--train-file", dest="facts_path", type=str, default="MemoryTest/json_data/semantic_facts.json")
    parser.add_argument("--test-file", type=str, default="")
    parser.add_argument("--output", type=str, default="MemoryTest/results/shine_initialized_lora_sft.json")
    parser.add_argument("--num-facts-list", type=int, nargs="+", default=[4, 8, 20])
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--selection-mode", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--training-objective", choices=["qa_sft", "ntp"], default="qa_sft")
    parser.add_argument("--variants-per-fact", type=int, default=5)
    parser.add_argument("--ntp-context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--ntp-context-variants", type=int, default=5)
    parser.add_argument("--ntp-record-mode", choices=["packed", "per_fact", "both"], default="both")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="bf16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--save-loras", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_lora_dtype(metanetwork) -> torch.dtype:
    return metanetwork.metamodel.get_input_embeddings().weight.dtype


def generate_shine_lora(context: str, metanetwork, tokenizer, metalora, cfg, device):
    from MemoryTest.case_test import generate_context_lora

    metanetwork.eval()
    lora_dict = generate_context_lora(context, metanetwork, tokenizer, metalora, cfg, device)
    return cast_floating_tensors(lora_dict, model_lora_dtype(metanetwork))


def evaluate_facts(label: str, facts: list[dict], metanetwork, tokenizer, lora_dict, device, max_new_tokens: int, max_length: int) -> dict:
    rows = []
    metanetwork.metamodel.eval()
    for idx, fact in enumerate(facts, start=1):
        result = answer_with_lora(metanetwork, tokenizer, lora_dict, fact["question"], device, max_new_tokens, max_length)
        rows.append(make_eval_row(idx, fact, result["answer"], raw=result["raw"]))
    summary = summarize_examples(rows)
    return {
        "label": label,
        **summary,
        "relation_breakdown": relation_breakdown(rows),
        "wrong_examples": wrong_examples(rows),
        "forced_choice": unavailable_forced_choice_result("not_enabled"),
        "rows": rows,
    }


def print_summary_table(rows: list[dict]) -> None:
    header = "num_facts | shine_init_acc | adapted_train_acc | adapted_test_acc"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['num_facts']} | "
            f"{row['shine_init_acc_mean']:.4f} | "
            f"{row['adapted_train_acc_mean']:.4f} | "
            f"{row['adapted_test_acc_mean']:.4f}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    all_facts = read_facts(args.facts_path)
    test_facts = read_facts(args.test_file) if args.test_file else []
    output_path = resolve_path(args.output)
    max_new_tokens = args.max_new_tokens

    runtime_args, cfg, device, metanetwork, metalora, tokenizer = load_shine(
        args.runtime_config,
        args.checkpoint_dir,
        args.device,
        args.gpu_id,
        args.torch_dtype,
    )
    if max_new_tokens is None:
        max_new_tokens = runtime_args.max_new_tokens

    trials = []
    summary_rows = []
    rng = random.Random(args.seed)

    for num_facts in args.num_facts_list:
        shine_init_accs = []
        adapted_train_accs = []
        adapted_test_accs = []
        for trial in range(args.num_trials):
            LOGGER.info("num_facts=%s trial=%s", num_facts, trial)
            train_facts = select_fact_subset(all_facts, num_facts, args.selection_mode, args.seed, trial)
            context_format = resolve_mixed_context_format(trial) if args.context_format == "mixed" else args.context_format
            context_rows = list(train_facts)
            if args.selection_mode == "random":
                rng.shuffle(context_rows)
            context = build_context(context_rows, context_format=context_format)

            shine_lora_dict = generate_shine_lora(context, metanetwork, tokenizer, metalora, cfg, device)
            shine_init_result = evaluate_facts(
                "shine_initialized_lora_before_sft",
                train_facts,
                metanetwork,
                tokenizer,
                shine_lora_dict,
                device,
                max_new_tokens,
                runtime_args.conversation_max_length,
            )

            trainable_lora_dict = make_lora_trainable(shine_lora_dict)
            metanetwork.metamodel.train()

            def epoch_eval_callback(epoch: int, current_lora_dict):
                result = evaluate_facts(
                    f"epoch_{epoch}_train_memorization",
                    train_facts,
                    metanetwork,
                    tokenizer,
                    current_lora_dict,
                    device,
                    max_new_tokens,
                    runtime_args.conversation_max_length,
                )
                metanetwork.metamodel.train()
                return {
                    "epoch": epoch,
                    "correct": result["correct"],
                    "total": result["total"],
                    "accuracy": result["accuracy"],
                    "wrong_examples": result["wrong_examples"],
                }

            train_stats = train_lora_dict(
                model=metanetwork.metamodel,
                tokenizer=tokenizer,
                lora_dict=trainable_lora_dict,
                facts=train_facts,
                device=device,
                seed=args.seed + trial,
                variants_per_fact=args.variants_per_fact,
                max_length=args.max_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                training_objective=args.training_objective,
                ntp_context_format=args.ntp_context_format,
                ntp_context_variants=args.ntp_context_variants,
                ntp_record_mode=args.ntp_record_mode,
                progress_label=f"SHINE-init facts={num_facts} trial={trial}",
                epoch_eval_callback=epoch_eval_callback,
            )
            best_lora_dict = train_stats.pop("best_lora_dict")
            adapted_train_result = evaluate_facts(
                "shine_initialized_lora_after_sft_best",
                train_facts,
                metanetwork,
                tokenizer,
                best_lora_dict,
                device,
                max_new_tokens,
                runtime_args.conversation_max_length,
            )
            adapted_test_result = None
            if test_facts:
                adapted_test_result = evaluate_facts(
                    "heldout_test_after_sft_best",
                    test_facts,
                    metanetwork,
                    tokenizer,
                    best_lora_dict,
                    device,
                    max_new_tokens,
                    runtime_args.conversation_max_length,
                )

            shine_init_accs.append(shine_init_result["accuracy"])
            adapted_train_accs.append(adapted_train_result["accuracy"])
            adapted_test_accs.append(adapted_test_result["accuracy"] if adapted_test_result else 0.0)
            trial_record = {
                "num_facts": num_facts,
                "trial": trial,
                "selection_mode": args.selection_mode,
                "context_format": context_format,
                "training_objective": args.training_objective,
                "train_fact_ids": [row["id"] for row in train_facts],
                "context": context,
                "shine_init_result": shine_init_result,
                "train_stats": train_stats,
                "adapted_train_result": adapted_train_result,
                "adapted_test_result": adapted_test_result,
            }
            if args.save_loras:
                lora_path = output_path.with_name(
                    f"{output_path.stem}_{args.training_objective}_facts{num_facts}_trial{trial}_best_lora.pt"
                )
                torch.save(move_lora_to_cpu(best_lora_dict), lora_path)
                trial_record["lora_path"] = str(lora_path)
            trials.append(trial_record)

            del shine_lora_dict, trainable_lora_dict, best_lora_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        shine_mean, shine_std = summarize_mean_std(shine_init_accs)
        train_mean, train_std = summarize_mean_std(adapted_train_accs)
        test_mean, test_std = summarize_mean_std(adapted_test_accs)
        summary_rows.append(
            {
                "num_facts": num_facts,
                "shine_init_acc_mean": shine_mean,
                "shine_init_acc_std": shine_std,
                "adapted_train_acc_mean": train_mean,
                "adapted_train_acc_std": train_std,
                "adapted_test_acc_mean": test_mean,
                "adapted_test_acc_std": test_std,
            }
        )

    payload = {
        "config": vars(args),
        "note": (
            "This experiment freezes SHINE and the base Qwen model, initializes a LoRA dictionary from "
            "SHINE(context), then directly updates only that LoRA dictionary with a small QA-SFT or NTP run."
        ),
        "data": {
            "facts_path": str(resolve_path(args.facts_path)),
            "test_file": str(resolve_path(args.test_file)) if args.test_file else None,
        },
        "summary": summary_rows,
        "trials": trials,
    }
    save_json(output_path, payload)
    print_summary_table(summary_rows)
    print(f"Wrote {output_path}")

    del metanetwork, metalora, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
