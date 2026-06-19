"""Test rms_relative perturbation: single perturb + restore bit-exact,
F1 stays in a productive band across σ_rel values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev

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
        lora_r=8, metalora_r=128,
    )
    rollout = SquadRollout(
        hypernet=net, reward_fn=f1_reward,
        cfg=SquadRolloutConfig(
            context_max_length=1024, question_max_length=256,
            max_new_tokens=64, questions_per_context=2,
        ),
    )
    train_ctx, _ = iter_train_val(train_size=200, val_size=50)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, 200, size=4)
    batch = [train_ctx[i] for i in idx]

    base_f1 = rollout(batch)
    print(f"[rms] un-perturbed F1 = {base_f1:.4f}")

    params = net.all_perturbable_params()

    # Restore bit-exactness check
    snap = {id(p): p.detach().clone() for _, p in params}

    seeds = (7, 13, 23, 31)
    for sigma_rel in (0.01, 0.03, 0.05, 0.08, 0.1, 0.2):
        perturber = InPlacePerturber(
            params, sigma=sigma_rel, sigma_mode="rms_relative", rms_floor=1e-3,
        )
        f1s = []
        for s in seeds:
            perturber.apply(s, +1)
            f1s.append(rollout(batch))
            perturber.restore(s, +1)
        # Check restore bit-exactness
        max_drift = max((p - snap[id(p)]).abs().max().item() for _, p in params)
        m, sd = mean(f1s), stdev(f1s)
        print(f"  σ_rel={sigma_rel:<6}  F1s={[round(x, 3) for x in f1s]}  "
              f"mean={m:.4f}  std={sd:.4f}  |Δ|={abs(m - base_f1):.4f}  "
              f"max_drift={max_drift:.2e}")


if __name__ == "__main__":
    main()
