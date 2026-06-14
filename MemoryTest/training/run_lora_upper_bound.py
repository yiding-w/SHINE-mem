#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from MemoryTest.evaluation.forced_choice import unavailable_forced_choice_result
from MemoryTest.evaluation.metrics import make_eval_row, relation_breakdown, summarize_examples, wrong_examples
from MemoryTest.training.lora_sft_utils import (
    load_frozen_lora_model,
    move_lora_to_cpu,
    print_summary_table,
    read_facts,
    resolve_path,
    select_fact_subset,
    summarize_mean_std,
    train_lora_dict,
)


LOGGER = logging.getLogger("run_lora_upper_bound")
TARGET_MODULES = [
    "model.layers[*].self_attn.q_proj",
    "model.layers[*].self_attn.k_proj",
    "model.layers[*].self_attn.v_proj",
    "model.layers[*].self_attn.o_proj",
    "model.layers[*].mlp.gate_proj",
    "model.layers[*].mlp.up_proj",
    "model.layers[*].mlp.down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ordinary LoRA as a MemoryTest memorization upper bound.")
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--facts-path", "--train-file", dest="facts_path", type=str, default="MemoryTest/json_data/semantic_facts.json")
    parser.add_argument("--test-file", type=str, default="")
    parser.add_argument("--output", "--output-path", dest="output", type=str, default="MemoryTest/results/lora_upper_bound.json")
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--num-facts-list", type=int, nargs="+", default=[4, 8, 20])
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--selection-mode", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--training-objective", choices=["qa_sft", "ntp"], default="qa_sft")
    parser.add_argument("--variants-per-fact", type=int, default=5)
    parser.add_argument("--ntp-context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--ntp-context-variants", type=int, default=5)
    parser.add_argument("--ntp-record-mode", choices=["packed", "per_fact", "both"], default="both")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--save-loras", action="store_true")
    return parser.parse_args()


def generate_answer(model, tokenizer, lora_dict, question: str, device, max_new_tokens: int, max_length: int) -> dict:
    from MemoryTest.case_test import extract_think_and_answer
    from MemoryTest.prepare_data.prompt_templates import question_prompt

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_prompt(question)}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
    new_tokens = outputs[0, input_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    think, answer = extract_think_and_answer(raw)
    return {"think": think, "answer": answer, "raw": raw}


def evaluate_facts(label: str, facts: list[dict], model, tokenizer, lora_dict, device, max_new_tokens: int, max_length: int) -> dict:
    rows = []
    for idx, fact in enumerate(facts, start=1):
        result = generate_answer(model, tokenizer, lora_dict, fact["question"], device, max_new_tokens, max_length)
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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    all_facts = read_facts(args.facts_path)
    test_facts = read_facts(args.test_file) if args.test_file else []
    output_path = resolve_path(args.output)
    trials = []
    summary_rows = []

    for rank in args.ranks:
        for num_facts in args.num_facts_list:
            setting_train_acc = []
            setting_test_acc = []
            for trial in range(args.num_trials):
                LOGGER.info("rank=%s num_facts=%s trial=%s", rank, num_facts, trial)
                train_facts = select_fact_subset(all_facts, num_facts, args.selection_mode, args.seed, trial)
                runtime_args, cfg, device, model, tokenizer, lora_dict = load_frozen_lora_model(
                    args.runtime_config,
                    rank=rank,
                    device_name=args.device,
                    gpu_id=args.gpu_id,
                )

                def epoch_eval_callback(epoch: int, current_lora_dict):
                    model.eval()
                    result = evaluate_facts(
                        f"epoch_{epoch}_train_memorization",
                        train_facts,
                        model,
                        tokenizer,
                        current_lora_dict,
                        device,
                        args.max_new_tokens,
                        runtime_args.conversation_max_length,
                    )
                    model.train()
                    return {
                        "epoch": epoch,
                        "correct": result["correct"],
                        "total": result["total"],
                        "accuracy": result["accuracy"],
                        "wrong_examples": result["wrong_examples"],
                    }

                train_stats = train_lora_dict(
                    model=model,
                    tokenizer=tokenizer,
                    lora_dict=lora_dict,
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
                    progress_label=f"rank={rank} facts={num_facts} trial={trial}",
                    epoch_eval_callback=epoch_eval_callback,
                )
                best_lora_dict = train_stats.pop("best_lora_dict")
                model.eval()
                train_result = evaluate_facts(
                    "train_memorization_best_lora",
                    train_facts,
                    model,
                    tokenizer,
                    best_lora_dict,
                    device,
                    args.max_new_tokens,
                    runtime_args.conversation_max_length,
                )
                test_result = None
                if test_facts:
                    test_result = evaluate_facts(
                        "heldout_test",
                        test_facts,
                        model,
                        tokenizer,
                        best_lora_dict,
                        device,
                        args.max_new_tokens,
                        runtime_args.conversation_max_length,
                    )

                setting_train_acc.append(train_result["accuracy"])
                setting_test_acc.append(test_result["accuracy"] if test_result else 0.0)
                trial_record = {
                    "rank": rank,
                    "num_facts": num_facts,
                    "trial": trial,
                    "selection_mode": args.selection_mode,
                    "training_objective": args.training_objective,
                    "train_fact_ids": [row["id"] for row in train_facts],
                    "train_stats": train_stats,
                    "train_result": train_result,
                    "test_result": test_result,
                }
                if args.save_loras:
                    lora_path = output_path.with_name(
                        f"{output_path.stem}_{args.training_objective}_rank{rank}_facts{num_facts}_trial{trial}_best_lora.pt"
                    )
                    torch.save(move_lora_to_cpu(best_lora_dict), lora_path)
                    trial_record["lora_path"] = str(lora_path)
                trials.append(trial_record)

                del model, tokenizer, lora_dict, best_lora_dict
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            train_mean, train_std = summarize_mean_std(setting_train_acc)
            test_mean, test_std = summarize_mean_std(setting_test_acc)
            summary_rows.append(
                {
                    "rank": rank,
                    "num_facts": num_facts,
                    "train_acc_mean": train_mean,
                    "train_acc_std": train_std,
                    "test_acc_mean": test_mean,
                    "test_acc_std": test_std,
                }
            )

    payload = {
        "config": vars(args),
        "target_modules": TARGET_MODULES,
        "note": "Ordinary LoRA uses the same LoraQwen injection surface as SHINE generated LoRA: every decoder layer q/k/v/o/gate/up/down. qa_sft trains on question-answer records; ntp trains only on fact/context text and is evaluated with QA generation.",
        "data": {
            "facts_path": str(resolve_path(args.facts_path)),
            "test_file": str(resolve_path(args.test_file)) if args.test_file else None,
        },
        "summary": summary_rows,
        "trials": trials,
    }
    save_json(output_path, payload)
    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
