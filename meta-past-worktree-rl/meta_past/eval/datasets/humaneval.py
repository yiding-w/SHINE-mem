"""HumanEval — bucket C, Python function-completion side-effect probe.

Mode = ``zero`` only: the prompt is just the function signature +
docstring, and we measure pass@1 against the dataset's test script.
SHINE wasn't trained on code, so this answers "does the LoRA *hurt*
code abilities?" — if pass@1 plummets vs base Qwen3 with the same
prompt, the hypernet is encoding non-task-specific stuff.
"""

from __future__ import annotations

from ..items import EvalItem
from . import DatasetSpec, register


def load(*, limit: int | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    items: list[EvalItem] = []
    for ex in ds:
        items.append(EvalItem(
            context="",                       # no demos, no passage
            question=ex["prompt"],
            references=[ex["canonical_solution"]],
            metadata={
                "task_id": ex["task_id"],
                "entry_point": ex["entry_point"],
                "test": ex["test"],
                "prompt": ex["prompt"],       # scorer needs this to reassemble program
            },
            item_id=ex["task_id"],
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


register(DatasetSpec(
    name="humaneval",
    bucket="C",
    load=load,
    scorer="humaneval_pass1",
    default_modes=("zero",),                 # shine/icl don't make sense here
    notes="Bucket C side-effect probe — does SHINE-LoRA break code?",
))
