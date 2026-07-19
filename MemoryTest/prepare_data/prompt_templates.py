from __future__ import annotations

import random
from typing import Iterable


DIRECT_QA_PREFIX = "Answer the question directly and output nothing else."


def build_natural_context(rows: Iterable[dict]) -> str:
    return "\n".join(str(row["text"]) for row in rows)


def build_structured_context(rows: Iterable[dict]) -> str:
    blocks = ["[MEMORY]"]
    for row in rows:
        blocks.extend(
            [
                f"person: {row['person']}",
                f"relation: {row.get('attribute', row.get('relation', 'unknown'))}",
                f"value: {row['answer']}",
                "",
            ]
        )
    blocks.append("[/MEMORY]")
    return "\n".join(blocks).strip()


def build_context(rows: Iterable[dict], context_format: str = "natural") -> str:
    rows = list(rows)
    if context_format == "natural":
        return build_natural_context(rows)
    if context_format == "structured":
        return build_structured_context(rows)
    if context_format == "mixed":
        return build_structured_context(rows) if len(rows) % 2 else build_natural_context(rows)
    raise ValueError(f"Unknown context format: {context_format}")


def direct_qa_prompt(question: str) -> str:
    return f"{DIRECT_QA_PREFIX}\n\nQuestion: {question}\nAnswer:"


def lora_sft_examples_for_fact(row: dict, rng: random.Random, max_variants: int = 3) -> list[dict[str, str]]:
    attribute = str(row.get("attribute", row.get("relation", "unknown")))
    person = str(row["person"])
    answer = str(row["answer"])
    text = str(row["text"])
    question = str(row["question"])
    candidates = [
        {
            "kind": "eval_style_qa",
            "prompt": question_prompt(question),
            "answer": answer,
        },
        {
            "kind": "direct_qa",
            "prompt": direct_qa_prompt(question),
            "answer": answer,
        },
        {
            "kind": "plain_qa",
            "prompt": f"Question: {question}\nAnswer:",
            "answer": answer,
        },
        {
            "kind": "cloze",
            "prompt": f"Complete the fact with the exact missing value.\n\n{text.rsplit(answer, 1)[0]}",
            "answer": answer,
        },
        {
            "kind": "relation_qa",
            "prompt": f"{DIRECT_QA_PREFIX}\n\nWhat is the {attribute} value for {person}?\nAnswer:",
            "answer": answer,
        },
    ]
    rng.shuffle(candidates)
    return candidates[:max_variants]


def question_prompt(question: str) -> str:
    return f"{DIRECT_QA_PREFIX}\n\nQuestion: {question}"


def reconstruction_prompt() -> str:
    # Match the original SHINE pretraining task exactly. The assistant target
    # determines whether the reconstructed content is facts, prose, or dialog.
    return "<RECON>"


def completion_prompt(session_prefix: str, source_session: int, observed_sessions: int) -> str:
    """Address one session and request its complete chronological history tail."""
    return (
        "<COMP>\n\n"
        f"The prefix identifies session {source_session} of {observed_sessions}. "
        "Reconstruct that complete session and every later observed session in "
        "chronological order. Output only the reconstructed text.\n\n"
        f"[SESSION {source_session} PREFIX]\n{session_prefix}"
    )


def format_answer(answer: str) -> str:
    return str(answer).strip()
