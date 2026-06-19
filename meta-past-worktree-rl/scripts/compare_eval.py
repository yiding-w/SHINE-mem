"""Compare two eval JSONs written by scripts/eval.py.

Emits overall delta and per-context wins / losses so you can tell whether
a checkpoint actually beats the baseline beyond noise.

Usage:
    python scripts/compare_eval.py <baseline.json> <candidate.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> tuple[dict, dict]:
    with open(path) as f:
        doc = json.load(f)
    return doc["summary"], {c["context_id"]: c for c in doc["per_context"]}


def _fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f} ({sign}{d * 100:.2f} F1 pts)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    base_sum, base_ctx = _load(Path(args.baseline))
    cand_sum, cand_ctx = _load(Path(args.candidate))

    common = sorted(set(base_ctx) & set(cand_ctx))
    print(f"# {base_sum.get('label')!r}  vs  {cand_sum.get('label')!r}")
    print(f"common contexts: {len(common)} "
          f"(baseline had {len(base_ctx)}, candidate has {len(cand_ctx)})")

    if len(common) != len(base_ctx) or len(common) != len(cand_ctx):
        print("NOTE: context sets differ; comparison restricted to intersection.")

    deltas = []
    for cid in common:
        b = base_ctx[cid]["mean_f1"]
        c = cand_ctx[cid]["mean_f1"]
        deltas.append((cid, c - b, b, c))
    deltas.sort(key=lambda t: t[1])

    overall_base = sum(d[2] for d in deltas) / max(len(deltas), 1)
    overall_cand = sum(d[3] for d in deltas) / max(len(deltas), 1)
    print(f"\noverall baseline F1:  {overall_base:.4f}")
    print(f"overall candidate F1: {overall_cand:.4f}")
    print(f"overall delta:        {_fmt_delta(overall_cand - overall_base)}")

    wins = sum(1 for _, d, _, _ in deltas if d > 1e-9)
    losses = sum(1 for _, d, _, _ in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses
    print(f"\nper-context: {wins} wins / {losses} losses / {ties} ties")

    print(f"\ntop {args.top} regressions:")
    for cid, d, b, c in deltas[: args.top]:
        print(f"  {cid}  {_fmt_delta(d)}  (base {b:.3f} -> cand {c:.3f})")
    print(f"\ntop {args.top} improvements:")
    for cid, d, b, c in reversed(deltas[-args.top:]):
        print(f"  {cid}  {_fmt_delta(d)}  (base {b:.3f} -> cand {c:.3f})")


if __name__ == "__main__":
    main()
