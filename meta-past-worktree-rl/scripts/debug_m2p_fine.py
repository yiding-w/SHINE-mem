"""Fine-grained σ sweep on m2p+metalora (excluding mem_tokens), across 4 seeds.

Goal: find a σ band where F1 actually varies measurably (not pinned at 0.67
baseline, not collapsed to 0). That band is our working ES sigma.
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
    rng = np.random.default_rng(42)
    idx = rng.integers(0, 200, size=4)
    batch = [train_ctx[i] for i in idx]

    base_f1 = rollout(batch)
    print(f"[debug] un-perturbed F1 = {base_f1:.4f}\n")

    all_params = net.all_perturbable_params()
    m2p_metalora = [(n, p) for n, p in all_params if not n.startswith("mem_tokens")]
    m2p_only = [(n, p) for n, p in all_params if n.startswith("m2p.")]
    print(f"m2p+metalora tensors: {len(m2p_metalora)}")
    print(f"m2p only tensors:     {len(m2p_only)}\n")

    seeds = (7, 13, 23, 31)

    def _sweep(name, params, sigmas):
        print(f"\n[{name}] sweep across {len(seeds)} seeds:")
        for sigma in sigmas:
            perturber = InPlacePerturber(params, sigma=sigma)
            f1s = []
            for s in seeds:
                perturber.apply(s, +1)
                f1s.append(rollout(batch))
                perturber.restore(s, +1)
            m, sd = mean(f1s), stdev(f1s)
            print(f"  sigma={sigma:<8}  F1s={[round(x, 3) for x in f1s]}  "
                  f"mean={m:.4f}  std={sd:.4f}  |Δ|={abs(m - base_f1):.4f}")

    _sweep("m2p+metalora", m2p_metalora,
           (1e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2))
    _sweep("m2p_only", m2p_only,
           (3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2))


if __name__ == "__main__":
    main()
