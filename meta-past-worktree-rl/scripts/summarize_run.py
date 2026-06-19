"""Print a compact summary of an ES training run.

Reads <run_dir>/train_log.jsonl and emits:
  * init / final heldout
  * per-eval heldout trajectory
  * per-step R_diff (sign indicates gradient signal quality)
  * anchor bump events

Usage:
    python scripts/summarize_run.py runs/smoke_run_01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _load(log_path: Path) -> list[dict]:
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(run_dir: Path) -> None:
    log = run_dir / "train_log.jsonl"
    if not log.is_file():
        raise FileNotFoundError(f"No train_log.jsonl at {log}")
    records = _load(log)

    train_steps = [r for r in records if r.get("event") == "train_step"]
    heldouts = [r for r in records
                if r.get("event") in {"init_heldout", "heldout_eval", "final_heldout"}]
    bumps = [r for r in records if r.get("event") == "anchor_bump"]

    print(f"# {run_dir}")
    print(f"total log lines: {len(records)}")
    print(f"train steps:     {len(train_steps)}")
    print(f"heldout evals:   {len(heldouts)}")
    print(f"anchor bumps:    {len(bumps)}")
    print()

    if heldouts:
        print("Heldout trajectory (step, reward):")
        for r in heldouts:
            print(f"  {r['step']:4d}  {r.get('heldout_reward', float('nan')):.4f}  "
                  f"[{r['event']}]")
        init = next((r for r in heldouts if r["event"] == "init_heldout"), None)
        final = heldouts[-1] if heldouts[-1]["event"] != "init_heldout" else None
        if init and final:
            delta = final["heldout_reward"] - init["heldout_reward"]
            print(f"\n  Δ (final - init) = {delta:+.4f}  ({delta * 100:+.2f} F1 points)")
        print()

    if train_steps:
        r_diffs = [s["R_diff_mean"] for s in train_steps if "R_diff_mean" in s]
        if r_diffs:
            wins = sum(1 for d in r_diffs if d > 0)
            print(f"Antithetic R_diff stats across {len(r_diffs)} steps:")
            print(f"  mean:     {mean(r_diffs):+.4f}")
            print(f"  positive: {wins}/{len(r_diffs)} ({wins / max(len(r_diffs), 1):.0%})")
            print(f"  first 5:  {[round(d, 4) for d in r_diffs[:5]]}")
            print(f"  last 5:   {[round(d, 4) for d in r_diffs[-5:]]}")
            print()

        # One-sided mode logs R_mean / R_min / R_max / R_std instead.
        r_means = [s["R_mean"] for s in train_steps if "R_mean" in s]
        r_maxs = [s["R_max"] for s in train_steps if "R_max" in s]
        if r_means:
            print(f"One-sided population stats across {len(r_means)} steps:")
            print(f"  mean of step means: {mean(r_means):+.4f}")
            print(f"  mean of step maxes: {mean(r_maxs):+.4f}")
            print(f"  best step max:      {max(r_maxs):+.4f}")
            print()

        rt = [s["rollout_time_s"] for s in train_steps if "rollout_time_s" in s]
        if rt:
            print(f"Mean rollout time / step: {mean(rt):.2f} s "
                  f"(first {rt[0]:.1f}s, last {rt[-1]:.1f}s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    args = p.parse_args()
    summarize(Path(args.run_dir))


if __name__ == "__main__":
    main()
