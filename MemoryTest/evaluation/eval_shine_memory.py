#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from pathlib import Path

import torch

from MemoryTest.prepare_data.prompt_templates import build_context, question_prompt
from MemoryTest.evaluation.metrics import make_eval_row, relation_breakdown, summarize_examples, wrong_examples
from MemoryTest.training.lora_sft_utils import load_runtime_args, read_facts, resolve_path


LOGGER = logging.getLogger("eval_shine_memory")

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SHINE memory behavior on MemoryTest semantic facts.")
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--baseline-checkpoint-dir", type=str, default="")
    parser.add_argument("--test-file", "--eval-file", dest="test_file", type=str, default="MemoryTest/json_data/splits/semantic_test.json")
    parser.add_argument("--output", type=str, default="MemoryTest/results/shine_memory_eval.json")
    parser.add_argument("--num-facts-list", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20])
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--include-baselines", action="store_true")
    return parser.parse_args()


def load_shine(runtime_config_path: str, checkpoint_dir: str, device_name: str | None, gpu_id: int | None):
    from MemoryTest.case_test import build_cfg, load_runtime, resolve_device

    runtime_args = load_runtime_args(runtime_config_path)
    if checkpoint_dir:
        runtime_args.checkpoint_dir = checkpoint_dir
    if device_name is not None:
        runtime_args.device = device_name
    if gpu_id is not None:
        runtime_args.gpu_id = gpu_id
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    metanetwork, metalora, tokenizer = load_runtime(cfg, runtime_args.checkpoint_dir, device)
    metanetwork.eval()
    return runtime_args, cfg, device, metanetwork, metalora, tokenizer


def answer_with_lora(metanetwork, tokenizer, lora_dict, question: str, device, max_new_tokens: int, max_length: int):
    from MemoryTest.case_test import extract_think_and_answer

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
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
    raw = tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True)
    think, answer = extract_think_and_answer(raw)
    return {"think": think, "answer": answer, "raw": raw}


def answer_in_context(metanetwork, tokenizer, context: str, question: str, device, max_new_tokens: int, max_length: int):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer the question directly."
    return answer_with_lora(metanetwork, tokenizer, None, prompt, device, max_new_tokens, max_length)


def generate_lora(context: str, metanetwork, tokenizer, metalora, cfg, device):
    from MemoryTest.case_test import generate_context_lora

    return generate_context_lora(context, metanetwork, tokenizer, metalora, cfg, device)


def evaluate_rows(label: str, rows: list[dict], answer_fn) -> dict:
    eval_rows = []
    for idx, fact in enumerate(rows, start=1):
        result = answer_fn(fact)
        eval_rows.append(make_eval_row(idx, fact, result["answer"], raw=result["raw"]))
    summary = summarize_examples(eval_rows)
    return {
        "label": label,
        **summary,
        "relation_breakdown": relation_breakdown(eval_rows),
        "wrong_examples": wrong_examples(eval_rows),
        "rows": eval_rows,
    }


def evaluate_checkpoint(label: str, args: argparse.Namespace, checkpoint_dir: str, facts: list[dict]) -> dict:
    runtime_args, cfg, device, metanetwork, metalora, tokenizer = load_shine(
        args.runtime_config,
        checkpoint_dir,
        args.device,
        args.gpu_id,
    )
    max_new_tokens = args.max_new_tokens or runtime_args.max_new_tokens
    rng = random.Random(args.seed)
    curves = []

    for num_facts in args.num_facts_list:
        trial_results = []
        trial_iter = tqdm(
            range(args.num_trials),
            desc=f"{label} facts={num_facts}",
            dynamic_ncols=True,
            leave=False,
        ) if tqdm is not None else range(args.num_trials)
        for trial in trial_iter:
            if len(facts) < num_facts:
                raise ValueError(f"Need {num_facts} facts, found {len(facts)}")
            context_rows = rng.sample(facts, num_facts)
            context = build_context(context_rows, context_format=args.context_format)
            lora_dict = generate_lora(context, metanetwork, tokenizer, metalora, cfg, device)
            shine_result = evaluate_rows(
                "shine_generated_lora",
                context_rows,
                lambda fact: answer_with_lora(
                    metanetwork,
                    tokenizer,
                    lora_dict,
                    fact["question"],
                    device,
                    max_new_tokens,
                    runtime_args.conversation_max_length,
                ),
            )
            trial_payload = {
                "trial": trial,
                "num_facts": num_facts,
                "context_fact_ids": [row["id"] for row in context_rows],
                "context": context,
                "shine": shine_result,
            }
            if args.include_baselines:
                trial_payload["no_lora_no_context"] = evaluate_rows(
                    "no_lora_no_context",
                    context_rows,
                    lambda fact: answer_with_lora(
                        metanetwork,
                        tokenizer,
                        None,
                        fact["question"],
                        device,
                        max_new_tokens,
                        runtime_args.conversation_max_length,
                    ),
                )
                trial_payload["no_lora_in_context"] = evaluate_rows(
                    "no_lora_in_context",
                    context_rows,
                    lambda fact: answer_in_context(
                        metanetwork,
                        tokenizer,
                        context,
                        fact["question"],
                        device,
                        max_new_tokens,
                        runtime_args.conversation_max_length,
                    ),
                )
            trial_results.append(trial_payload)
            if tqdm is not None:
                trial_iter.set_postfix({"shine_acc": f"{shine_result['accuracy']:.4f}"})
        shine_acc = [trial["shine"]["accuracy"] for trial in trial_results]
        curves.append(
            {
                "num_facts": num_facts,
                "accuracy_mean": sum(shine_acc) / len(shine_acc) if shine_acc else 0.0,
                "trials": trial_results,
            }
        )

    del metanetwork, metalora, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "label": label,
        "checkpoint_dir": checkpoint_dir,
        "curves": curves,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    facts = read_facts(args.test_file)
    checkpoints = []
    if args.baseline_checkpoint_dir:
        checkpoints.append(("original_shine", args.baseline_checkpoint_dir))
    checkpoints.append(("shine", args.checkpoint_dir))

    results = [evaluate_checkpoint(label, args, checkpoint_dir, facts) for label, checkpoint_dir in checkpoints]
    payload = {
        "config": vars(args),
        "data": {
            "test_file": str(resolve_path(args.test_file)),
            "num_rows": len(facts),
        },
        "results": results,
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
