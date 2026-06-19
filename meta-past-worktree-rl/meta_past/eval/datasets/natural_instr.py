"""Super-NaturalInstructions (test split) — bucket B.

Each task ships with a natural-language ``definition`` plus
``positive_examples`` (the official demos). For SHINE-LoRA mode the
``context`` is ``definition + K positive examples``; the query is the
held-out item's ``inputs``. This matches what the dataset's authors
designed it for: in-context task transfer.

The repo is large (1000+ tasks × thousands of items); we stream and
cap by ``limit``.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def _format_demos(definition: str, positive_examples: list[dict],
                  shots: int) -> str:
    parts = []
    if definition:
        parts.append(f"Task definition:\n{definition}")
    if positive_examples:
        for i, pe in enumerate(positive_examples[:shots]):
            inp = pe.get("input", "")
            out = pe.get("output", "")
            parts.append(f"Example {i+1}:\nInput: {inp}\nOutput: {out}")
    return "\n\n".join(parts)


def load(*, limit: int | None = None, shots: int = 2, seed: int = 42,
         **_) -> list[EvalItem]:
    from datasets import load_dataset
    # Streaming = no big up-front download.
    ds = load_dataset("Muennighoff/natural-instructions", split="test",
                      streaming=True)
    items: list[EvalItem] = []
    for i, ex in enumerate(ds):
        if limit is not None and len(items) >= limit:
            break
        context = _format_demos(
            str(ex.get("definition", "")),
            list(ex.get("positive_examples", []) or []),
            shots,
        )
        targs = ex.get("targets") or [ex.get("target", "")]
        if isinstance(targs, str):
            targs = [targs]
        items.append(EvalItem(
            context=context,
            question=f"Input: {ex.get('inputs', ex.get('input', ''))}\nOutput:",
            references=[str(t) for t in targs if str(t).strip()],
            metadata={"task": ex.get("task_name", ""),
                      "shots": int(shots)},
            item_id=f"natinst/{ex.get('task_name', '?')}/{i}",
        ))
    return items


register(DatasetSpec(
    name="natural_instr",
    bucket="B",
    load=load,
    scorer="em",
    default_modes=("shine", "icl", "zero"),
    notes="Purpose-built held-out task generalization; def + K positive examples in context.",
))
