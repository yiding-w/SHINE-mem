"""MuSiQue — bucket A, multi-hop QA (training distribution).

Reuses the project's own loader so context format ('# Title\\nbody'
blocks) matches the trainer exactly. Heldout = last 256 contexts of
the validation split.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from ...data.musique_contexts import iter_train_val
    _, val = iter_train_val(train_size=0, val_size=256)
    items: list[EvalItem] = []
    for ctx in val:
        qa = ctx.qa[0]
        items.append(EvalItem(
            context=ctx.context,
            question=qa.question,
            references=list(qa.references),
            metadata={"context_id": ctx.context_id},
            item_id=ctx.context_id,
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="musique",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="One of the RL training datasets; eval if hypernet was trained elsewhere.",
))
