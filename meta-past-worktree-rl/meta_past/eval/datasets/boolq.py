"""BoolQ — bucket A, passage + yes/no question."""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")
    items: list[EvalItem] = []
    for i, ex in enumerate(ds):
        items.append(EvalItem(
            context=ex["passage"],
            # Phrasing the question explicitly elicits a yes/no token.
            question=(f"{ex['question']}? Answer with just 'yes' or 'no'."),
            references=["yes" if ex["answer"] else "no"],
            metadata={"raw_answer": bool(ex["answer"])},
            item_id=f"boolq/{i}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="boolq",
    bucket="A",
    load=load,
    scorer="em",
    default_modes=("shine", "icl", "zero"),
    notes="Cheapest bucket-A sanity check.",
))
