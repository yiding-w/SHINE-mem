"""SQuAD v1 — bucket A, context-grounded extractive QA.

Reuses the existing project loader so ``context`` / ``question`` /
``references`` exactly match what the trainer feeds the hypernet.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from ...data.squad_contexts import iter_train_val
    # SQuAD validation has ~2k passages; we re-use the same slice the
    # trainer uses for heldout (the last 256 after the training slice).
    # For eval we want the standard 256-passage heldout block.
    _, val = iter_train_val(train_size=1024, val_size=256)
    items: list[EvalItem] = []
    for ctx in val:
        for qa in ctx.qa:
            items.append(EvalItem(
                context=ctx.context,
                question=qa.question,
                references=list(qa.references),
                metadata={"context_id": ctx.context_id},
                item_id=f"squad/{ctx.context_id}/{qa.question[:32]}",
            ))
            if limit is not None and len(items) >= limit:
                return items
    return items


register(DatasetSpec(
    name="squad",
    bucket="A",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="SHINE pretraining target; ceiling-near baseline.",
))
