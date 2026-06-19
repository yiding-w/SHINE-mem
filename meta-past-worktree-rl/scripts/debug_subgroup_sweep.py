"""Isolate which parameter subgroup (m2p / mem_tokens / metalora) is
responsible for the sharp F1 collapse at σ above the 1e-5 noise floor.

Also tries a per-tensor RMS-scaled σ ('relative' perturbation) to see if
rescaling closes the signal band.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.data.squad_contexts import iter_train_val
from meta_past.es.noise import make_noise
from meta_past.es.perturb import InPlacePerturber, tensor_seed
from meta_past.reward.f1_reward import f1_reward
from meta_past.rollout.squad_rollout import SquadRollout, SquadRolloutConfig
from meta_past.shine_adapter import ShineHypernet


def _subset(params, kind):
    if kind == "m2p":
        return [(n, p) for n, p in params if n.startswith("m2p.")]
    if kind == "mem_tokens":
        return [(n, p) for n, p in params if n.startswith("mem_tokens")]
    if kind == "metalora":
        return [(n, p) for n, p in params if n.startswith("metalora.")]
    if kind == "all":
        return list(params)
    raise ValueError(kind)


def _apply_relative(params, base_seed, sign, sigma_rel):
    """Perturb each tensor by sign * sigma_rel * RMS(tensor) * eps."""
    for k, (_, p) in enumerate(params):
        rms = float(p.detach().float().pow(2).mean().sqrt().item())
        if rms == 0.0:
            # fall back to absolute sigma_rel for zero-init tensors (mem_tokens)
            rms = 1.0
        eps = make_noise(p.shape, tensor_seed(base_seed, k), p.device,
                         torch.float32)
        p.data.add_((sign * sigma_rel * rms) * eps.to(p.dtype))


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

    params = net.all_perturbable_params()

    # --- 1. Per-subgroup sweep (absolute sigma) ------------------------------
    print("[debug] Absolute-sigma sweep per subgroup (single seed +1):")
    for kind in ("m2p", "mem_tokens", "metalora"):
        sub = _subset(params, kind)
        print(f"\n  subgroup = {kind} ({len(sub)} tensors):")
        for sigma in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            perturber = InPlacePerturber(sub, sigma=sigma)
            perturber.apply(7, +1)
            f1 = rollout(batch)
            perturber.restore(7, +1)
            print(f"    sigma={sigma:<8}  F1={f1:.4f}")

    # --- 2. Relative-sigma sweep (all params) --------------------------------
    print("\n[debug] Relative-sigma sweep (σ_k = σ_rel · RMS(p_k)), all params:")
    snap = {id(p): p.detach().clone() for _, p in params}
    for sigma_rel in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        # apply
        _apply_relative(params, 7, +1, sigma_rel)
        f1 = rollout(batch)
        # restore via +/- symmetry
        _apply_relative(params, 7, -1, sigma_rel)
        # sanity: tensors back to snapshot?
        drift = max((p - snap[id(p)]).abs().max().item() for _, p in params)
        print(f"  sigma_rel={sigma_rel:<8}  F1={f1:.4f}  max_drift_after_restore={drift:.2e}")


if __name__ == "__main__":
    main()
