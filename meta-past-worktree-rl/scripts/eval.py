"""Score a hypernetwork checkpoint on a slice of SQuAD contexts.

Usage:
    python scripts/eval.py \
        --ckpt ~/huggingfacemodels/SHINE-ift_mqa_1qa \
        --context-start 200 --context-count 50 \
        --questions-per-context 2

Prints mean F1 and writes a per-context breakdown to --out (JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.data.squad_contexts import load_squad_contexts
from meta_past.reward.f1_reward import f1_reward
from meta_past.rollout.squad_rollout import SquadRollout, SquadRolloutConfig
from meta_past.shine_adapter import ShineHypernet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    home = Path.home()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backbone", default=str(home / "huggingfacemodels" / "Qwen3-8B"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--context-start", type=int, default=0)
    p.add_argument("--context-count", type=int, default=50)
    p.add_argument("--questions-per-context", type=int, default=2)
    p.add_argument("--context-max-length", type=int, default=1024)
    p.add_argument("--question-max-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    p.add_argument("--label", default=None, help="Label included in summary JSON.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[eval] Loading hypernet from {args.ckpt}", flush=True)
    net = ShineHypernet(
        ckpt_dir=args.ckpt,
        device=args.device,
        backbone=args.backbone,
        lora_r=8,
        metalora_r=128,
    )
    net.assert_only_hypernet_trainable()

    rollout = SquadRollout(
        hypernet=net,
        reward_fn=f1_reward,
        cfg=SquadRolloutConfig(
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
            max_new_tokens=args.max_new_tokens,
            questions_per_context=args.questions_per_context,
        ),
    )

    all_ctx = load_squad_contexts(split="validation")
    end = args.context_start + args.context_count
    if end > len(all_ctx):
        raise ValueError(
            f"Requested contexts [{args.context_start}:{end}] "
            f"but SQuAD validation has only {len(all_ctx)} grouped contexts."
        )
    ctxs = all_ctx[args.context_start:end]
    print(f"[eval] Scoring {len(ctxs)} contexts, "
          f"{args.questions_per_context} questions each ...", flush=True)

    per_ctx: list[dict] = []
    all_rewards: list[float] = []
    for i, ctx in enumerate(ctxs):
        rs = rollout.score_context(ctx)
        m = float(mean(rs)) if rs else 0.0
        per_ctx.append({
            "context_id": ctx.context_id,
            "n_questions_scored": len(rs),
            "mean_f1": m,
            "per_question_f1": rs,
        })
        all_rewards.extend(rs)
        if (i + 1) % 10 == 0:
            print(f"[eval] {i + 1}/{len(ctxs)} contexts, running mean F1 = "
                  f"{mean(all_rewards):.4f}", flush=True)

    overall = float(mean(all_rewards)) if all_rewards else 0.0
    summary = {
        "label": args.label,
        "ckpt": args.ckpt,
        "context_start": args.context_start,
        "context_count": len(ctxs),
        "questions_per_context": args.questions_per_context,
        "n_questions_scored": len(all_rewards),
        "mean_f1": overall,
    }
    print("[eval] summary =", json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_context": per_ctx}, f, indent=2)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
