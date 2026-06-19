"""MMLU — bucket B, 57 subjects MCQ.

K-shot demos are drawn from the SAME subject's ``dev`` split as the
query (canonical few-shot MMLU protocol). Subjects can be filtered
via ``--shots-subject`` later; for v1 we evaluate the union.
"""

from __future__ import annotations

import random

from ._mcq import fmt_mcq_demo, fmt_mcq_query
from ..items import EvalItem
from . import DatasetSpec, register


# Canonical MMLU subject list (57 subjects). Hard-coded so we don't
# need a network call to enumerate.
_MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
)


def load(*, limit: int | None = None, shots: int = 5, seed: int = 42,
         subjects: tuple[str, ...] | None = None, **_) -> list[EvalItem]:
    from datasets import load_dataset
    use_subjects = subjects or _MMLU_SUBJECTS

    items: list[EvalItem] = []
    for subj in use_subjects:
        try:
            dev = load_dataset("cais/mmlu", subj, split="dev")
            test = load_dataset("cais/mmlu", subj, split="test")
        except Exception:
            continue
        n_dev = len(dev)
        for i, ex in enumerate(test):
            rng = random.Random(f"{seed}-{subj}-{i}")
            demo_idx = rng.sample(range(n_dev), k=min(shots, n_dev))
            demos = [
                fmt_mcq_demo(dev[j]["question"], dev[j]["choices"],
                             chr(65 + dev[j]["answer"]))
                for j in demo_idx
            ]
            items.append(EvalItem(
                context="\n\n".join(demos),
                question=fmt_mcq_query(ex["question"], ex["choices"]),
                references=[chr(65 + ex["answer"])],
                metadata={"subject": subj, "n_options": 4, "shots": int(shots)},
                item_id=f"mmlu/{subj}/{i}",
            ))
            if limit is not None and len(items) >= limit:
                return items
    return items


register(DatasetSpec(
    name="mmlu",
    bucket="B",
    load=load,
    scorer="mcq_letter",
    default_modes=("shine", "icl", "zero"),
    scorer_kwargs={"n_options": 4},
    notes="K-shot demos drawn from same-subject dev split.",
))
