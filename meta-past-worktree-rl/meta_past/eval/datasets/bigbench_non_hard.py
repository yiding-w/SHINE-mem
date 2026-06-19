"""BIG-Bench (non-Hard) — bucket B.

BBH covers 27 tasks; BIG-Bench has ~200 in total. For v1 we sample a
small held-out set of *non-BBH* tasks. Extend ``_TASKS`` to widen.

Each item: ``inputs`` is the prompt body, ``targets`` is a list of
acceptable gold strings; demos come from the same task's earlier
items.
"""

from __future__ import annotations

import random

from ..items import EvalItem
from . import DatasetSpec, register


# Representative non-Hard BIG-Bench tasks. Confirmed reachable via the
# tasksource mirror as of 2026-05.
_TASKS = (
    "known_unknowns",
    "anachronisms",
    "cause_and_effect",
)


def _fmt(inputs: str, target: str | None = None) -> str:
    s = inputs.rstrip()
    if target is not None:
        s += f"\nAnswer: {target}"
    return s


def load(*, limit: int | None = None, shots: int = 4, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    items: list[EvalItem] = []
    for task in _TASKS:
        try:
            ds = load_dataset("tasksource/bigbench", task, split="validation")
        except Exception:
            continue
        n = len(ds)
        for i, ex in enumerate(ds):
            rng = random.Random(f"{seed}-{task}-{i}")
            pool = [j for j in range(n) if j != i]
            demo_idx = rng.sample(pool, k=min(shots, len(pool)))
            demos: list[str] = []
            for j in demo_idx:
                d = ds[j]
                targs = d.get("targets") or []
                if not targs:
                    continue
                demos.append(_fmt(d.get("inputs", ""), targs[0]))
            items.append(EvalItem(
                context="\n\n".join(demos),
                question=_fmt(ex.get("inputs", "")),
                references=list(ex.get("targets", [])),
                metadata={"task": task, "shots": int(shots)},
                item_id=f"bigbench/{task}/{i}",
            ))
            if limit is not None and len(items) >= limit:
                return items
    return items


register(DatasetSpec(
    name="bigbench_non_hard",
    bucket="B",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="Held-out BIG-Bench tasks outside BBH-27.",
))
