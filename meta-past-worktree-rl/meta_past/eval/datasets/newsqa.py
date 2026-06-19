"""NewsQA — bucket A, extractive QA over CNN news articles."""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("lucadiliello/newsqa", split="validation")
    items: list[EvalItem] = []
    for i, ex in enumerate(ds):
        items.append(EvalItem(
            context=ex["context"],
            question=ex["question"],
            references=list(ex["answers"]),
            metadata={"key": ex.get("key", str(i))},
            item_id=f"newsqa/{ex.get('key', i)}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="newsqa",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="News-domain extractive QA.",
))
