"""DROP — bucket A, discrete reasoning over passages (counting, arithmetic, sorting)."""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("ucinlp/drop", split="validation")
    items: list[EvalItem] = []
    for ex in ds:
        ans = ex["answers_spans"]
        refs: list[str] = []
        if ans.get("spans"):
            refs.extend(ans["spans"])
        if not refs:
            refs.append(str(ans))
        items.append(EvalItem(
            context=ex["passage"],
            question=ex["question"],
            references=refs,
            metadata={"section_id": ex["section_id"], "query_id": ex["query_id"]},
            item_id=f"drop/{ex['query_id']}",
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="drop",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="Discrete reasoning. F1 on spans; numeric answers often score 0 — consider numeric_em variant later.",
))
