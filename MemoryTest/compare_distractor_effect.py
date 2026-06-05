#!/usr/bin/env python
import argparse
import json
import logging

import experiment_utils as exp


LOGGER = logging.getLogger("compare_distractor_effect")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 1: compare same target facts/update count, with vs without distractor context."
    )
    parser.add_argument("--runtime-config", type=str, default=str(exp.DEFAULT_RUNTIME_CONFIG_PATH))
    parser.add_argument("--facts-path", type=str, default="MemoryTest/json_data/semantic_facts.json")
    parser.add_argument("--distractors-path", type=str, default="MemoryTest/json_data/distractors.json")
    parser.add_argument("--output-path", type=str, default="MemoryTest/results/distractor_effect_4x5_vs_4x5plus5.json")
    parser.add_argument("--num-updates", type=int, default=4)
    parser.add_argument("--facts-per-update", type=int, default=5)
    parser.add_argument("--distractors-per-update", type=int, default=5)
    parser.add_argument("--save-loras", action="store_true")
    parser.add_argument("--log-context", action="store_true")
    parser.add_argument("--merge-method", choices=["average", "concat"], default="average")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    total_facts = args.num_updates * args.facts_per_update
    total_distractors = args.num_updates * args.distractors_per_update
    output_path = exp.resolve_path(args.output_path)

    facts, facts_path = exp.read_json_rows(args.facts_path, total_facts)
    distractors, distractors_path = exp.read_json_rows(args.distractors_path, total_distractors)
    target_facts = facts[:total_facts]
    fact_chunks = exp.chunk_rows(target_facts, args.facts_per_update, args.num_updates)
    distractor_chunks = exp.chunk_rows(distractors[:total_distractors], args.distractors_per_update, args.num_updates)
    mixed_chunks = [exp.interleave_rows(fact_chunk, distractor_chunk) for fact_chunk, distractor_chunk in zip(fact_chunks, distractor_chunks)]

    runtime_args, device, cfg, metanetwork, metalora, tokenizer = exp.bootstrap_runtime(args.runtime_config)

    lora_c, contexts_c = exp.generate_merged_lora(
        fact_chunks,
        metanetwork,
        tokenizer,
        metalora,
        cfg,
        device,
        log_context=args.log_context,
        condition_label="C_clean_4x5",
        merge_method=args.merge_method,
    )
    lora_d, contexts_d = exp.generate_merged_lora(
        mixed_chunks,
        metanetwork,
        tokenizer,
        metalora,
        cfg,
        device,
        log_context=args.log_context,
        condition_label="D_distractor_4x5plus5",
        merge_method=args.merge_method,
    )

    if args.save_loras:
        exp.save_lora_snapshot(output_path.with_name(output_path.stem + f"_C_clean_{args.merge_method}_lora.pt"), lora_c)
        exp.save_lora_snapshot(output_path.with_name(output_path.stem + f"_D_distractor_{args.merge_method}_lora.pt"), lora_d)

    result_c = exp.evaluate_lora(f"C_4x5_fact_only_{args.merge_method}_lora", target_facts, lora_c, metanetwork, tokenizer, runtime_args, device)
    result_d = exp.evaluate_lora(f"D_4x5_fact_plus_5_distractor_{args.merge_method}_lora", target_facts, lora_d, metanetwork, tokenizer, runtime_args, device)

    payload = {
        "experiment": {
            "description": "Experiment 1: same target fact count and same update count; D adds filler distractors to each update.",
            "facts_path": str(facts_path),
            "distractors_path": str(distractors_path),
            "runtime_config": str(exp.resolve_path(args.runtime_config)),
            "num_updates": args.num_updates,
            "facts_per_update": args.facts_per_update,
            "distractors_per_update": args.distractors_per_update,
            "total_eval_facts": total_facts,
            "distractor_placement": "interleaved",
            "merge_method": args.merge_method,
        },
        "summary": {
            result_c["label"]: exp.summarize_result(result_c),
            result_d["label"]: exp.summarize_result(result_d),
        },
        "contexts": {
            result_c["label"]: contexts_c,
            result_d["label"]: contexts_d,
        },
        "results": [result_c, result_d],
    }
    exp.write_payload(output_path, payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
