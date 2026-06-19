"""BIG-Bench Hard — bucket B, in-parameter few-shot.

Eval items have ``context = K demos from one BBH task family`` and
``question = a held-out item from the same family``. The K in eval
**can differ from the K used at training time**: that's the whole
point of the K-shot sweep — see whether the LoRA's capacity tracks
the number of demos.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, shots: int = 16, **_) -> list[EvalItem]:
    """``shots`` = K. We rebuild the context for each (task, query) with
    the requested K rather than reusing the trainer's pre-built
    contexts (which fixed K at construction time).
    """
    from ...data.bbh_contexts import iter_train_val
    # K_max governs the upper bound during context construction; the
    # bbh loader already does an 80/20 split per-task. Pass shots
    # through so we can request smaller K windows.
    _, val = iter_train_val(train_size=0, val_size=256, K_max=int(shots))
    items: list[EvalItem] = []
    for ctx in val:
        qa = ctx.qa[0]
        items.append(EvalItem(
            context=ctx.context,
            question=qa.question,
            references=list(qa.references),
            metadata={
                "context_id": ctx.context_id,
                "task_family": ctx.context_id.split("/")[1]
                                if "/" in ctx.context_id else "",
                "shots": int(shots),
            },
            item_id=ctx.context_id,
        ))
        if limit is not None and len(items) >= limit:
            return items
    return items


register(DatasetSpec(
    name="bbh",
    bucket="B",
    load=load,
    scorer="f1",
    default_modes=("shine", "icl", "zero"),
    notes="In-parameter few-shot, training distribution. K-shot sweep is the core experiment.",
))
