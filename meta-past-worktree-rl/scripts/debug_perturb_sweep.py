"""Diagnose why ES rewards collapsed: measure the F1 effect of sigma on the
exact rollout path the trainer uses, across multiple seeds and sigmas, on the
exact first training batch (the 4 contexts ES step 1 would see).

Also logs the pre-perturbation F1 on that same batch, to separate 'unlucky
batch' from 'perturbation is too destructive'.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.data.squad_contexts import iter_train_val
from meta_past.es.perturb import InPlacePerturber
from meta_past.reward.f1_reward import f1_reward
from meta_past.rollout.squad_rollout import SquadRollout, SquadRolloutConfig
from meta_past.shine_adapter import ShineHypernet


def main() -> None:
    home = Path.home()
    net = ShineHypernet(
        ckpt_dir=str(home / "huggingfacemodels" / "SHINE-ift_mqa_1qa"),
        device="cuda:0",
        backbone=str(home / "huggingfacemodels" / "Qwen3-8B"),
        lora_r=8,
        metalora_r=128,
    )

    rollout = SquadRollout(
        hypernet=net,
        reward_fn=f1_reward,
        cfg=SquadRolloutConfig(
            context_max_length=1024,
            question_max_length=256,
            max_new_tokens=64,
            questions_per_context=2,
        ),
    )

    train_ctx, _ = iter_train_val(train_size=200, val_size=50)

    # Reproduce what ESTrainer._sample_train_batch returns on step 1 with seed=42
    rng = np.random.default_rng(42)
    idx = rng.integers(0, 200, size=4)
    batch = [train_ctx[i] for i in idx]
    print(f"[debug] training batch indices: {list(idx)}")

    # ---- 1. Baseline (no perturbation) on this exact batch ------------------
    base_f1 = rollout(batch)
    print(f"[debug] UN-perturbed F1 on batch = {base_f1:.4f}")

    # ---- 2. Inspect parameter magnitudes ------------------------------------
    params = net.all_perturbable_params()
    print(f"\n[debug] {len(params)} perturbable tensors. Magnitude summary:")
    bins = {"m2p": [], "mem_tokens": [], "metalora": []}
    for name, t in params:
        rms = float(t.detach().float().pow(2).mean().sqrt().item())
        if name.startswith("m2p."):
            bins["m2p"].append(rms)
        elif name.startswith("mem_tokens"):
            bins["mem_tokens"].append(rms)
        else:
            bins["metalora"].append(rms)
    for k, v in bins.items():
        if v:
            print(f"  {k:10s}  n={len(v):4d}  rms_range=[{min(v):.2e}, {max(v):.2e}]  "
                  f"rms_median={sorted(v)[len(v)//2]:.2e}")

    # ---- 3. Sigma sweep -----------------------------------------------------
    print("\n[debug] Sigma sweep (F1 under single +1 perturbation):")
    for sigma in (0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3):
        if sigma == 0.0:
            # Already measured; show it again for alignment
            print(f"  sigma={sigma:<8}  F1={base_f1:.4f}  (no perturb)")
            continue
        perturber = InPlacePerturber(params, sigma=sigma)
        # Try 2 seeds per sigma so we see variance.
        f1s = []
        for seed in (7, 13):
            perturber.apply(seed, +1)
            f1s.append(rollout(batch))
            perturber.restore(seed, +1)
        print(f"  sigma={sigma:<8}  F1 per seed = {[round(x, 4) for x in f1s]}  "
              f"mean={mean(f1s):.4f}")


if __name__ == "__main__":
    main()
