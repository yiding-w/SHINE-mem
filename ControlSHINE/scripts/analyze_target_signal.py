#!/usr/bin/env python
"""Teacher-forced target log-probability and token-rank analysis at alpha=1."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ControlSHINE.controlshine.runtime import load_pretrain_runtime  # noqa: E402
from ControlSHINE.scripts.run_memory_logit_sweep import _encoded_prompt, _read_jsonl, _select  # noqa: E402
from ControlSHINE.scripts.run_source_only import _memory_lora, _prompt  # noqa: E402


@torch.inference_mode()
def _target_stats(model, tokenizer, device, prompt, target, loradict):
    prompt_ids, prompt_mask = _encoded_prompt(tokenizer, prompt, device)
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if target_ids.shape[1] == 0:
        raise ValueError(f"empty target tokenization: {target!r}")
    full_ids = torch.cat((prompt_ids, target_ids), dim=1)
    full_mask = torch.cat((prompt_mask, torch.ones_like(target_ids)), dim=1)
    output = model.metamodel(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=False,
        loradict=loradict,
        ignore_mem_token=True,
    )
    start = prompt_ids.shape[1] - 1
    logits = output.logits[0, start : start + target_ids.shape[1], :].float()
    targets = target_ids[0]
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    log_probs = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    ranks = (logits > target_logits[:, None]).sum(dim=-1) + 1
    top_logits = logits.max(dim=-1).values
    return {
        "num_tokens": int(targets.numel()),
        "total_logprob": float(log_probs.sum().cpu()),
        "mean_logprob": float(log_probs.mean().cpu()),
        "first_token_rank": int(ranks[0].cpu()),
        "mean_token_rank": float(ranks.float().mean().cpu()),
        "mean_top1_margin": float((target_logits - top_logits).mean().cpu()),
        "token_ids": targets.detach().cpu().tolist(),
        "token_ranks": ranks.detach().cpu().tolist(),
    }


def _mean(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-only", required=True)
    parser.add_argument("--sweep-results", required=True)
    parser.add_argument("--controlled-alpha", type=float, default=1.2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
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
    sweep = {row["sample_id"]: row for row in _read_jsonl(args.sweep_results)}
    alpha_key = str(float(args.controlled_alpha))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)

    with (output_dir / "target_signal.jsonl").open("w", encoding="utf-8") as handle:
        for index, (group, row) in enumerate(selected, 1):
            if row["sample_id"] not in sweep or alpha_key not in sweep[row["sample_id"]]["sweep"]:
                continue
            question = row.get("prompts", {}).get("rewrite") or row["question"]
            prompt = _prompt(question)
            target = row["memory"]["answer"]
            lora = _memory_lora(model, tokenizer, metalora, cfg, device, row["memory"]["text"])
            base = _target_stats(model, tokenizer, device, prompt, target, None)
            memory = _target_stats(model, tokenizer, device, prompt, target, lora)
            controlled_success = bool(sweep[row["sample_id"]]["sweep"][alpha_key]["match"])
            status = "rescued" if group == "rescue" and controlled_success else (
                "not_rescued" if group == "rescue" else "retention"
            )
            result = {
                "sample_id": row["sample_id"], "group": group, "status": status,
                "target": target, "controlled_success": controlled_success,
                "base": base, "memory": memory,
                "delta_mean_logprob": memory["mean_logprob"] - base["mean_logprob"],
                "first_rank_improvement": base["first_token_rank"] - memory["first_token_rank"],
                "mean_rank_improvement": base["mean_token_rank"] - memory["mean_token_rank"],
            }
            grouped[status].append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[TargetSignal] {index}/{len(selected)} status={status}", flush=True)

    summary = {"controlled_alpha": args.controlled_alpha, "groups": {}}
    for status, rows in grouped.items():
        summary["groups"][status] = {
            "count": len(rows),
            "mean_delta_logprob": _mean(rows, "delta_mean_logprob"),
            "fraction_positive_delta_logprob": (
                sum(row["delta_mean_logprob"] > 0 for row in rows) / len(rows) if rows else None
            ),
            "mean_first_rank_improvement": _mean(rows, "first_rank_improvement"),
            "mean_rank_improvement": _mean(rows, "mean_rank_improvement"),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

