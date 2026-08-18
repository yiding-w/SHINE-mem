#!/usr/bin/env python
"""Evaluate fixed Memory logit scale across CounterFact paraphrases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ControlSHINE.controlshine.metrics import answer_match  # noqa: E402
from ControlSHINE.controlshine.runtime import load_pretrain_runtime  # noqa: E402
from ControlSHINE.scripts.run_memory_logit_sweep import (  # noqa: E402
    _decode_residual, _read_jsonl, _select,
)
from ControlSHINE.scripts.run_source_only import _memory_lora, _prompt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-only", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--paraphrases-per-sample", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--rescue-limit", type=int, default=999999)
    parser.add_argument("--retention-limit", type=int, default=999999)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    cfg, model, metalora, tokenizer = load_pretrain_runtime(
        args.runtime_config, args.checkpoint_dir, device
    )
    selected = _select(
        _read_jsonl(args.input), _read_jsonl(args.source_only),
        args.rescue_limit, args.retention_limit,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    correct = defaultdict(Counter)
    totals = Counter()
    all_correct = defaultdict(Counter)

    with (output_dir / "paraphrase_consistency.jsonl").open("w", encoding="utf-8") as handle:
        for index, (group, row) in enumerate(selected, 1):
            paraphrases = list(row.get("prompts", {}).get("paraphrases") or [])
            paraphrases = paraphrases[: args.paraphrases_per_sample]
            if not paraphrases:
                continue
            lora = _memory_lora(model, tokenizer, metalora, cfg, device, row["memory"]["text"])
            target = row["memory"]["answer"]
            aliases = row["memory"].get("aliases", [])
            records = []
            sample_matches = {"1.0": [], str(float(args.alpha)): []}
            for prompt_index, paraphrase in enumerate(paraphrases):
                outputs = {}
                for alpha in (1.0, float(args.alpha)):
                    prediction = _decode_residual(
                        model, tokenizer, device, _prompt(paraphrase), lora,
                        alpha, args.max_new_tokens,
                    )
                    matched = answer_match(prediction, target, aliases)
                    key = str(float(alpha))
                    outputs[key] = {"prediction": prediction, "match": matched}
                    correct[group][key] += int(matched)
                    totals[(group, key)] += 1
                    sample_matches[key].append(matched)
                records.append({"index": prompt_index, "prompt": paraphrase, "outputs": outputs})
            for key, matches in sample_matches.items():
                all_correct[group][key] += int(bool(matches) and all(matches))
            handle.write(json.dumps({
                "sample_id": row["sample_id"], "group": group,
                "target": target, "paraphrases": records,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[Paraphrase] {index}/{len(selected)} group={group}", flush=True)

    summary = {"alpha": args.alpha, "groups": {}}
    for group in ("rescue", "retention"):
        summary["groups"][group] = {}
        for alpha in (1.0, float(args.alpha)):
            key = str(float(alpha))
            denom = totals[(group, key)]
            summary["groups"][group][key] = {
                "prompt_correct": correct[group][key],
                "prompt_total": denom,
                "prompt_accuracy": correct[group][key] / denom if denom else None,
                "all_paraphrases_correct_samples": all_correct[group][key],
            }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

