"""Compute the parameter-space displacement between two checkpoints.

For every perturbable tensor in the m_h scope, prints:
  - RMS of the init tensor
  - RMS of (final - init)
  - Relative displacement (RMS(δ)/RMS(init))

Hypothesis being tested: ES at the smoke-run scale produces tiny net updates
in parameter space (the cumulative ĝ averages out to ~0), which is why 43/50
heldout F1 values are unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.shine_adapter import ShineHypernet


def _flatten(params, include_metalora=True):
    """Return list of (name, RMS_init, displacement_RMS, rel_displacement)."""
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    home = Path.home()
    ap.add_argument("--init-ckpt",
                    default=str(home / "huggingfacemodels" / "SHINE-ift_mqa_1qa"))
    ap.add_argument("--final-ckpt",
                    default="runs/smoke_run_01/checkpoint-final")
    ap.add_argument("--backbone",
                    default=str(home / "huggingfacemodels" / "Qwen3-8B"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[disp] loading init   = {args.init_ckpt}")
    init = ShineHypernet(
        ckpt_dir=args.init_ckpt, device=args.device, backbone=args.backbone,
        lora_r=8, metalora_r=128,
    )
    print(f"[disp] loading final  = {args.final_ckpt}")
    final = ShineHypernet(
        ckpt_dir=args.final_ckpt, device=args.device, backbone=args.backbone,
        lora_r=8, metalora_r=128,
    )

    # Compare ALL tensors that ES might have touched (include_metalora=True).
    init_params = init.all_perturbable_params(include_metalora=True)
    final_params = final.all_perturbable_params(include_metalora=True)
    assert len(init_params) == len(final_params), \
        f"shape count mismatch: init={len(init_params)} vs final={len(final_params)}"

    rows = []
    for (n_i, p_i), (n_f, p_f) in zip(init_params, final_params):
        if n_i != n_f:
            print(f"[disp] WARN name mismatch: {n_i!r} vs {n_f!r}")
        delta = (p_f.detach().float() - p_i.detach().float())
        rms_init = float(p_i.detach().float().pow(2).mean().sqrt().item())
        rms_delta = float(delta.pow(2).mean().sqrt().item())
        rel = rms_delta / rms_init if rms_init > 0 else float("inf")
        max_delta = float(delta.abs().max().item())
        rows.append((n_i, rms_init, rms_delta, rel, max_delta, p_i.numel()))

    # Print bucketed summary first
    by_group = {"m2p": [], "metalora": [], "other": []}
    for r in rows:
        name = r[0]
        if name.startswith("m2p."):
            by_group["m2p"].append(r)
        elif name.startswith("metalora."):
            by_group["metalora"].append(r)
        else:
            by_group["other"].append(r)

    print("\n[disp] per-group summary (relative displacement RMS(δ)/RMS(init)):")
    for grp, items in by_group.items():
        if not items:
            continue
        rels = [it[3] for it in items]
        max_deltas = [it[4] for it in items]
        zero_displace = sum(1 for d in [it[2] for it in items] if d == 0)
        print(f"  {grp:10s} n={len(items):4d}  "
              f"rel_disp: mean={mean(rels):.4e}  "
              f"max={max(rels):.4e}  "
              f"min={min(rels):.4e}  "
              f"zero-displaced={zero_displace}/{len(items)}  "
              f"max_abs_δ={max(max_deltas):.4e}")

    # Top 10 movers per group
    print("\n[disp] top 5 movers (relative) per group:")
    for grp, items in by_group.items():
        if not items:
            continue
        sorted_items = sorted(items, key=lambda r: -r[3])
        print(f"\n  -- {grp} --")
        for n, rms_i, rms_d, rel, max_d, numel in sorted_items[:5]:
            print(f"    {n:60s}  numel={numel:>10d}  "
                  f"RMS_init={rms_i:.3e}  rel_disp={rel:.3e}  max|δ|={max_d:.3e}")


if __name__ == "__main__":
    main()
