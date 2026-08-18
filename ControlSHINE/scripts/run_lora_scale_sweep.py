#!/usr/bin/env python
"""Baseline: scale the generated LoRA output xAB+C before decoding."""

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
from ControlSHINE.scripts.run_memory_logit_sweep import _read_jsonl, _select  # noqa: E402
from ControlSHINE.scripts.run_source_only import _generate, _memory_lora, _prompt  # noqa: E402


def _scale_lora(node, scale: float):
    if isinstance(node, dict) and "A" in node and "B" in node:
        result = dict(node)
        result["B"] = node["B"] * scale
        if node.get("C") is not None:
            result["C"] = node["C"] * scale
        return result
    if isinstance(node, dict):
        return {key: _scale_lora(value, scale) for key, value in node.items()}
    if isinstance(node, list):
        return [_scale_lora(value, scale) for value in node]
    if isinstance(node, tuple):
        return tuple(_scale_lora(value, scale) for value in node)
    return node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-only", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
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

    with (output_dir / "lora_scale_sweep.jsonl").open("w", encoding="utf-8") as handle:
        for index, (group, row) in enumerate(selected, 1):
            question = row.get("prompts", {}).get("rewrite") or row["question"]
            original = _memory_lora(model, tokenizer, metalora, cfg, device, row["memory"]["text"])
            target = row["memory"]["answer"]
            aliases = row["memory"].get("aliases", [])
            sweep = {}
            totals[group] += 1
            for scale in args.scales:
                prediction = _generate(
                    model, tokenizer, device, _prompt(question),
                    _scale_lora(original, float(scale)),
                    cfg.test.conversation_max_length, args.max_new_tokens, "force-empty",
                )
                matched = answer_match(prediction, target, aliases)
                key = str(float(scale))
                sweep[key] = {"prediction": prediction, "match": matched}
                correct[group][key] += int(matched)
            handle.write(json.dumps({
                "sample_id": row["sample_id"], "group": group,
                "target": target, "sweep": sweep,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[LoRAScale] {index}/{len(selected)} group={group}", flush=True)

    summary = {"totals": dict(totals), "scales": {}}
    for scale in args.scales:
        key = str(float(scale))
        summary["scales"][key] = {
            "rescue_correct": correct["rescue"][key],
            "rescue_rate": correct["rescue"][key] / totals["rescue"] if totals["rescue"] else None,
            "retention_correct": correct["retention"][key],
            "retention_rate": correct["retention"][key] / totals["retention"] if totals["retention"] else None,
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

