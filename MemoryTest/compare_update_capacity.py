#!/usr/bin/env python
import argparse
import gc
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("compare_update_capacity")
MEMORY_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MEMORY_TEST_ROOT.parent
DEFAULT_RUNTIME_CONFIG_PATH = MEMORY_TEST_ROOT / "config" / "case_test.yaml"
torch = None


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (REPO_ROOT / resolved).resolve()


def load_yaml_defaults(path: str) -> dict:
    from omegaconf import OmegaConf

    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Runtime config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config must be a YAML mapping: {cfg_path}")
    return data


def make_runtime_args(defaults: dict) -> Namespace:
    return Namespace(
        model_path=defaults["model_path"],
        checkpoint_dir=defaults["checkpoint_dir"],
        device=defaults.get("device", "cuda"),
        gpu_id=int(defaults.get("gpu_id", 0)),
        seed=int(defaults.get("seed", 42)),
        context_max_length=int(defaults.get("context_max_length", 1550)),
        conversation_max_length=int(defaults.get("conversation_max_length", 5000)),
        max_new_tokens=int(defaults.get("max_new_tokens", 128)),
        lora_r=int(defaults.get("lora_r", 8)),
        metalora_r=int(defaults.get("metalora_r", 128)),
        metanetwork_layers=int(defaults.get("metanetwork_layers", 4)),
        use_system_prompt=bool(defaults.get("use_system_prompt", False)),
    )


def weighted_average_lora(old_lora: Any, new_lora: Any, new_count: int):
    if old_lora is None or new_lora is None:
        if old_lora is not None or new_lora is not None:
            raise ValueError("LoRA structures do not match: one side is None and the other is not.")
        return None
    old_weight = float(new_count - 1) / float(new_count)
    new_weight = 1.0 / float(new_count)
    if torch.is_tensor(old_lora):
        return old_lora * old_weight + new_lora * new_weight
    if isinstance(old_lora, dict):
        return {key: weighted_average_lora(old_lora[key], new_lora[key], new_count) for key in old_lora}
    if isinstance(old_lora, list):
        return [weighted_average_lora(old_item, new_item, new_count) for old_item, new_item in zip(old_lora, new_lora)]
    if isinstance(old_lora, tuple):
        return tuple(weighted_average_lora(old_item, new_item, new_count) for old_item, new_item in zip(old_lora, new_lora))
    raise TypeError(f"Unsupported LoRA value type for averaging: {type(old_lora)!r}")


def build_context(rows: list[dict]) -> str:
    return "\n".join(row["text"] for row in rows)


def generate_average_lora(fact_chunks, metanetwork, tokenizer, metalora, cfg, device, log_context: bool = False):
    averaged_lora = None
    context_records = []
    for update_idx, chunk in enumerate(fact_chunks, start=1):
        context = build_context(chunk)
        context_records.append(
            {
                "update_index": update_idx,
                "num_facts": len(chunk),
                "fact_ids": [row["id"] for row in chunk],
                "context": context,
            }
        )
        LOGGER.info("A/update %s: generating LoRA from %s facts", update_idx, len(chunk))
        if log_context:
            LOGGER.info("A/update %s context:\n%s", update_idx, context)
        new_lora = generate_context_lora(context, metanetwork, tokenizer, metalora, cfg, device)
        if averaged_lora is None:
            averaged_lora = new_lora
        else:
            averaged_lora = weighted_average_lora(averaged_lora, new_lora, update_idx)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return averaged_lora, context_records


def evaluate_lora(label, facts, lora_dict, metanetwork, tokenizer, runtime_args, device):
    LOGGER.info("Evaluating %s on %s questions", label, len(facts))
    rows = []
    correct = 0
    for idx, fact in enumerate(facts, start=1):
        result = answer_question(
            question=fact["question"],
            metanetwork=metanetwork,
            tokenizer=tokenizer,
            lora_dict=lora_dict,
            device=device,
            max_new_tokens=runtime_args.max_new_tokens,
            max_conversation_length=runtime_args.conversation_max_length,
            use_system_prompt=runtime_args.use_system_prompt,
        )
        is_correct = fact["answer"] in result["answer"] or fact["answer"] in result["raw"]
        correct += int(is_correct)
        rows.append(
            {
                "index": idx,
                "id": fact["id"],
                "person": fact["person"],
                "question": fact["question"],
                "expected_answer": fact["answer"],
                "model_answer": result["answer"],
                "raw": result["raw"],
                "correct": is_correct,
            }
        )
    return {
        "label": label,
        "correct": correct,
        "total": len(facts),
        "accuracy": correct / len(facts) if facts else 0.0,
        "rows": rows,
    }


def save_lora_snapshot(path: Path, lora_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(move_lora_to_cpu(lora_dict), path)
    LOGGER.info("Saved LoRA snapshot to %s", path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare repeated averaged LoRA updates with one-shot LoRA memory.")
    parser.add_argument("--runtime-config", type=str, default=str(DEFAULT_RUNTIME_CONFIG_PATH), help="MemoryTest runtime config with model/checkpoint paths.")
    parser.add_argument("--phonebook-path", type=str, default="MemoryTest/json_data/phonebook.json", help="Phonebook JSON generated by generate_capacity_data.py.")
    parser.add_argument("--output-path", type=str, default="MemoryTest/results/update4_avg_vs_single40.json", help="Where to save comparison results.")
    parser.add_argument("--num-updates", type=int, default=2, help="Number of averaged LoRA updates for condition A.")
    parser.add_argument("--facts-per-update", type=int, default=1, help="Facts per update for condition A.")
    parser.add_argument("--save-loras", action="store_true", help="Save A and B generated LoRA dictionaries next to the result JSON.")
    parser.add_argument("--log-context", action="store_true", help="Print the exact A/B contexts before generating LoRA.")
    return parser.parse_args()


def main():
    global torch
    global answer_question, build_cfg, generate_context_lora, load_runtime, move_lora_to_cpu, resolve_device

    args = parse_args()

    import torch as torch_module
    from case_test import (
        answer_question,
        build_cfg,
        generate_context_lora,
        load_runtime,
        move_lora_to_cpu,
        resolve_device,
    )

    torch = torch_module
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    total_facts = args.num_updates * args.facts_per_update
    phonebook_path = resolve_path(args.phonebook_path)
    output_path = resolve_path(args.output_path)

    facts = json.loads(phonebook_path.read_text(encoding="utf-8"))
    if len(facts) < total_facts:
        raise ValueError(f"Need at least {total_facts} phone facts, found {len(facts)} in {phonebook_path}")
    facts = facts[:total_facts]
    fact_chunks = [facts[i:i + args.facts_per_update] for i in range(0, total_facts, args.facts_per_update)]

    runtime_args = make_runtime_args(load_yaml_defaults(args.runtime_config))
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    metanetwork, metalora, tokenizer = load_runtime(cfg, runtime_args.checkpoint_dir, device)

    lora_a, contexts_a = generate_average_lora(
        fact_chunks,
        metanetwork,
        tokenizer,
        metalora,
        cfg,
        device,
        log_context=args.log_context,
    )
    LOGGER.info("B/single: generating LoRA from all %s facts at once", total_facts)
    context_b = build_context(facts)
    if args.log_context:
        LOGGER.info("B/single context:\n%s", context_b)
    lora_b = generate_context_lora(context_b, metanetwork, tokenizer, metalora, cfg, device)

    if args.save_loras:
        save_lora_snapshot(output_path.with_name(output_path.stem + "_A_avg_lora.pt"), lora_a)
        save_lora_snapshot(output_path.with_name(output_path.stem + "_B_single_lora.pt"), lora_b)

    result_a = evaluate_lora(f"A_{args.num_updates}x{args.facts_per_update}_average_lora", facts, lora_a, metanetwork, tokenizer, runtime_args, device)
    result_b = evaluate_lora(f"B_1x{total_facts}_single_lora", facts, lora_b, metanetwork, tokenizer, runtime_args, device)

    payload = {
        "experiment": {
            "description": "A: repeated LoRA generation averaged across updates; B: one-shot LoRA from the same facts.",
            "num_updates": args.num_updates,
            "facts_per_update": args.facts_per_update,
            "total_facts": total_facts,
            "phonebook_path": str(phonebook_path),
            "runtime_config": str(resolve_path(args.runtime_config)),
        },
        "summary": {
            result_a["label"]: {"correct": result_a["correct"], "total": result_a["total"], "accuracy": result_a["accuracy"]},
            result_b["label"]: {"correct": result_b["correct"], "total": result_b["total"], "accuracy": result_b["accuracy"]},
        },
        "contexts": {
            result_a["label"]: contexts_a,
            result_b["label"]: [
                {
                    "update_index": 1,
                    "num_facts": len(facts),
                    "fact_ids": [row["id"] for row in facts],
                    "context": context_b,
                }
            ],
        },
        "results": [result_a, result_b],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
