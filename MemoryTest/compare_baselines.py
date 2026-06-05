#!/usr/bin/env python
import argparse
import json
import logging

import experiment_utils as exp


def parse_args():
    parser = argparse.ArgumentParser(description="Run no-LoRA baselines with and without prompt context.")
    parser.add_argument("--runtime-config", type=str, default=str(exp.DEFAULT_RUNTIME_CONFIG_PATH))
    parser.add_argument("--facts-path", type=str, default="MemoryTest/json_data/semantic_facts.json")
    parser.add_argument("--output-path", type=str, default="MemoryTest/results/baselines_no_lora_20.json")
    parser.add_argument("--num-facts", type=int, default=20)
    parser.add_argument("--log-context", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    output_path = exp.resolve_path(args.output_path)
    facts, facts_path = exp.read_json_rows(args.facts_path, args.num_facts)
    facts = facts[:args.num_facts]
    context = exp.build_context(facts)
    if args.log_context:
        logging.getLogger("compare_baselines").info("In-context baseline context:\n%s", context)

    runtime_args, device, cfg, metanetwork, metalora, tokenizer = exp.bootstrap_runtime(args.runtime_config)

    result_no_context = exp.evaluate_lora(
        "baseline_no_lora_no_context",
        facts,
        None,
        metanetwork,
        tokenizer,
        runtime_args,
        device,
    )
    result_in_context = exp.evaluate_in_context_baseline(
        "baseline_no_lora_in_context",
        facts,
        context,
        metanetwork,
        tokenizer,
        runtime_args,
        device,
    )

    payload = {
        "experiment": {
            "description": "No-LoRA baselines: direct question answering without context vs prompt in-context answering.",
            "facts_path": str(facts_path),
            "runtime_config": str(exp.resolve_path(args.runtime_config)),
            "num_facts": args.num_facts,
        },
        "summary": {
            result_no_context["label"]: exp.summarize_result(result_no_context),
            result_in_context["label"]: exp.summarize_result(result_in_context),
        },
        "contexts": {
            result_no_context["label"]: [
                {
                    "num_rows": 0,
                    "num_facts": 0,
                    "fact_ids": [],
                    "context": "",
                }
            ],
            result_in_context["label"]: [
                {
                    "num_rows": len(facts),
                    "num_facts": len(facts),
                    "fact_ids": [row["id"] for row in facts],
                    "context": context,
                }
            ],
        },
        "results": [result_no_context, result_in_context],
    }
    exp.write_payload(output_path, payload)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
