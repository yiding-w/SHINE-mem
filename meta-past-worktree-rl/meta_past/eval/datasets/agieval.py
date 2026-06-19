"""AGIEval (English subtests) — bucket B.

For v1 we evaluate one subtest (``lsat-lr``); add subtests by extending
``_SUBTESTS``. Demos drawn from within the same subtest.
"""

from __future__ import annotations

import random

from ..items import EvalItem
from . import DatasetSpec, register


_SUBTESTS = ("lsat-lr",)


def _fmt(query: str, with_answer_letter: str | None = None) -> str:
    s = query.rstrip()
    if with_answer_letter is not None:
        s += f"\nAnswer: ({with_answer_letter})"
    return s


def _gold_letter(gold) -> str | None:
    if isinstance(gold, list) and gold:
        gold = gold[0]
    if isinstance(gold, int):
        return chr(65 + gold)
    if isinstance(gold, str):
        return gold.strip().upper().strip("()")
    return None


def load(*, limit: int | None = None, shots: int = 4, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    items: list[EvalItem] = []
    for sub in _SUBTESTS:
        ds = load_dataset(f"hails/agieval-{sub}", split="test")
        n = len(ds)
        for i, ex in enumerate(ds):
            rng = random.Random(f"{seed}-{sub}-{i}")
            pool = [j for j in range(n) if j != i]
            demo_idx = rng.sample(pool, k=min(shots, len(pool)))
            demos: list[str] = []
            for j in demo_idx:
                d = ds[j]
                let = _gold_letter(d.get("gold"))
                if let is None:
                    continue
                demos.append(_fmt(d["query"], with_answer_letter=let))
            items.append(EvalItem(
                context="\n\n".join(demos),
                question=_fmt(ex["query"]),
                references=[_gold_letter(ex.get("gold")) or ""],
                metadata={"subtest": sub, "n_options": 5, "shots": int(shots)},
                item_id=f"agieval/{sub}/{i}",
            ))
            if limit is not None and len(items) >= limit:
                return items
    return items


register(DatasetSpec(
    name="agieval",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 5},
    notes="LSAT / SAT / standardized-test style — far from BBH distribution.",
))
