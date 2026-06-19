"""Phase 0 sanity: load SHINE-ift_mqa_1qa, run zero-shot SQuAD, probe perturbation.

Per implementation_plan.md §5:
  1. Load SHINE-ift_mqa_1qa.
  2. Zero-shot eval on N SQuAD contexts (F1).
  3. Apply one random perturbation (sigma=0.005) to all of m_h, re-eval.
  4. Restore. Re-eval — should match step 2 exactly (fp32) or closely (bf16).

Run with:
    python scripts/phase0_sanity.py --contexts 10 --questions-per-context 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

import torch

# Make the package importable when running as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.data.squad_contexts import load_squad_contexts
from meta_past.es.perturb import InPlacePerturber
from meta_past.reward.f1_reward import f1_reward
from meta_past.rollout.squad_rollout import SquadRollout, SquadRolloutConfig
from meta_past.shine_adapter import ShineHypernet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    home = Path.home()
    p.add_argument("--ckpt", default=str(home / "huggingfacemodels" / "SHINE-ift_mqa_1qa"))
    p.add_argument("--backbone", default=str(home / "huggingfacemodels" / "Qwen3-8B"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--contexts", type=int, default=10)
    p.add_argument("--questions-per-context", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.005)
    p.add_argument("--perturb-seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"[phase0] Loading hypernet from {args.ckpt} ...", flush=True)
    net = ShineHypernet(
        ckpt_dir=args.ckpt,
        device=args.device,
        backbone=args.backbone,
        lora_r=8,
        metalora_r=128,
    )
    net.assert_only_hypernet_trainable()
    print(f"[phase0] num_mem_token = {net.num_mem_token}", flush=True)

    rollout = SquadRollout(
        hypernet=net,
        reward_fn=f1_reward,
        cfg=SquadRolloutConfig(
            questions_per_context=args.questions_per_context,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    ctxs = load_squad_contexts(split="validation", max_contexts=args.contexts)
    print(f"[phase0] Running zero-shot on {len(ctxs)} contexts...", flush=True)
    baseline = rollout(ctxs)
    print(f"[phase0] baseline F1 = {baseline:.4f}")

    print(f"[phase0] Applying random perturbation sigma={args.sigma} "
          f"seed={args.perturb_seed} to all of m_h ...", flush=True)
    perturber = InPlacePerturber(net.all_perturbable_params(), sigma=args.sigma)
    perturber.apply(args.perturb_seed, +1)
    perturbed = rollout(ctxs)
    print(f"[phase0] perturbed F1 = {perturbed:.4f}")

    perturber.restore(args.perturb_seed, +1)
    restored = rollout(ctxs)
    print(f"[phase0] restored F1 = {restored:.4f}")

    summary = {
        "baseline": baseline,
        "perturbed": perturbed,
        "restored": restored,
        "contexts": len(ctxs),
        "questions_per_context": args.questions_per_context,
        "sigma": args.sigma,
    }
    print("[phase0] summary =", json.dumps(summary, indent=2))

    assert abs(baseline - restored) < 1e-3, (
        f"[phase0] FAIL: restored F1 deviates from baseline by "
        f"{abs(baseline - restored):.4f}"
    )
    print("[phase0] OK: baseline and restored match within tolerance.")


if __name__ == "__main__":
    main()
