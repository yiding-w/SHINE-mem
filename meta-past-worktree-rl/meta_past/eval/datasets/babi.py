"""bAbI — bucket B, 20 synthetic reasoning task families.

Demos and queries from the SAME ``task`` field; we run all 20 families
and aggregate. For per-family analysis use the ``task_family`` metadata.
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..items import EvalItem
from . import DatasetSpec, register


def _fmt(passage: str, question: str, answer: str | None = None) -> str:
    s = f"Story:\n{passage.rstrip()}\nQ: {question}\nA:"
    if answer is not None:
        s += f" {answer}"
    return s


def load(*, limit: int | None = None, shots: int = 4, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    train = load_dataset("Muennighoff/babi", split="train")
    test = load_dataset("Muennighoff/babi", split="test")
    # Bucket train by task for per-family demo pools.
    train_by_task: dict[int, list[int]] = defaultdict(list)
    for j, ex in enumerate(train):
        train_by_task[int(ex["task"])].append(j)
    items: list[EvalItem] = []
    for i, ex in enumerate(test):
        t = int(ex["task"])
        pool = train_by_task.get(t, [])
        if not pool:
            continue
        rng = random.Random(f"{seed}-{t}-{i}")
        demo_idx = rng.sample(pool, k=min(shots, len(pool)))
        demos = [
            _fmt(train[j]["passage"], train[j]["question"],
                 str(train[j]["answer"]))
            for j in demo_idx
        ]
        items.append(EvalItem(
            context="\n\n".join(demos),
            question=_fmt(ex["passage"], ex["question"]),
            references=[str(ex["answer"])],
            metadata={"task_family": t, "shots": int(shots)},
            item_id=f"babi/t{t}/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="babi",
    bucket="B",
    load=load,
    scorer="em",
    default_modes=("shine", "icl", "zero"),
    notes="20-family synthetic reasoning; cleanest bucket-B sanity probe.",
))
