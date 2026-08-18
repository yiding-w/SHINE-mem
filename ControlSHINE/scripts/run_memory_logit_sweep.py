#!/usr/bin/env python
"""Training-free sweep of zB + alpha * (zM - zB) for Memory policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ControlSHINE.controlshine.metrics import answer_match  # noqa: E402
from ControlSHINE.controlshine.runtime import load_pretrain_runtime  # noqa: E402
from ControlSHINE.scripts.run_source_only import _memory_lora, _prompt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True, help="Converted three-source JSONL")
    parser.add_argument("--source-only", required=True, help="source_only.jsonl from the same data/checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2, 3, 4, 6, 8])
    parser.add_argument("--rescue-limit", type=int, default=20)
    parser.add_argument("--retention-limit", type=int, default=20)
    return parser.parse_args()


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select(data_rows, source_rows, rescue_limit, retention_limit):
    data = {row["sample_id"]: row for row in data_rows}
    selected = []
    used = Counter()
    limits = {"rescue": rescue_limit, "retention": retention_limit}
    for result in source_rows:
        label = result["recoverability"]
        group = "rescue" if label == "unrecoverable_memory" else (
            "retention" if label == "fully_recoverable" else None
        )
        if group is None or used[group] >= limits[group]:
            continue
        sample_id = result["sample_id"]
        if sample_id in data:
            selected.append((group, data[sample_id]))
            used[group] += 1
    return selected


def _encoded_prompt(tokenizer, prompt, device):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    no_think = tokenizer(
        "<think>\n\n</think>\n\n", add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(device)
    return (
        torch.cat((input_ids, no_think), dim=1),
        torch.cat((attention_mask, torch.ones_like(no_think)), dim=1),
    )


@torch.inference_mode()
def _decode_residual(model, tokenizer, device, prompt, lora, alpha, max_new_tokens):
    initial_ids, initial_mask = _encoded_prompt(tokenizer, prompt, device)
    base_ids = initial_ids
    memory_ids = initial_ids
    base_cache = None
    memory_cache = None
    full_mask = initial_mask
    generated = []
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    for _ in range(max_new_tokens):
        base_out = model.metamodel(
            input_ids=base_ids,
            attention_mask=full_mask,
            past_key_values=base_cache,
            use_cache=True,
            logits_to_keep=1,
            loradict=None,
            ignore_mem_token=True,
        )
        memory_out = model.metamodel(
            input_ids=memory_ids,
            attention_mask=full_mask,
            past_key_values=memory_cache,
            use_cache=True,
            logits_to_keep=1,
            loradict=lora,
            ignore_mem_token=True,
        )
        base_cache = base_out.past_key_values
        memory_cache = memory_out.past_key_values
        z_base = base_out.logits[:, -1, :].float()
        z_memory = memory_out.logits[:, -1, :].float()
        token = int((z_base + alpha * (z_memory - z_base)).argmax(dim=-1).item())
        if token in stop_ids:
            break
        generated.append(token)
        next_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        base_ids = next_ids
        memory_ids = next_ids
        full_mask = torch.cat(
            (full_mask, torch.ones((1, 1), dtype=full_mask.dtype, device=device)), dim=1
        )
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    cfg, model, metalora, tokenizer = load_pretrain_runtime(
        args.runtime_config, args.checkpoint_dir, device
    )
    selected = _select(
        _read_jsonl(args.input),
        _read_jsonl(args.source_only),
        args.rescue_limit,
        args.retention_limit,
    )
    if not selected:
        raise ValueError("No rescue or retention samples selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "memory_logit_sweep.jsonl"
    correct = defaultdict(Counter)
    totals = Counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for index, (group, row) in enumerate(selected, 1):
            question = row.get("prompts", {}).get("rewrite") or row["question"]
            lora = _memory_lora(model, tokenizer, metalora, cfg, device, row["memory"]["text"])
            target = row["memory"]["answer"]
            aliases = row["memory"].get("aliases", [])
            sweep = {}
            totals[group] += 1
            for alpha in args.scales:
                prediction = _decode_residual(
                    model, tokenizer, device, _prompt(question), lora,
                    float(alpha), args.max_new_tokens,
                )
                matched = answer_match(prediction, target, aliases)
                key = str(float(alpha))
                sweep[key] = {"prediction": prediction, "match": matched}
                correct[group][key] += int(matched)
            handle.write(json.dumps({
                "sample_id": row["sample_id"],
                "group": group,
                "target": target,
                "sweep": sweep,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[MemorySweep] {index}/{len(selected)} group={group} target={target!r}", flush=True)

    summary = {"totals": dict(totals), "scales": {}}
    for alpha in args.scales:
        key = str(float(alpha))
        summary["scales"][key] = {
            "rescue_correct": correct["rescue"][key],
            "rescue_rate": correct["rescue"][key] / totals["rescue"] if totals["rescue"] else None,
            "retention_correct": correct["retention"][key],
            "retention_rate": correct["retention"][key] / totals["retention"] if totals["retention"] else None,
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

