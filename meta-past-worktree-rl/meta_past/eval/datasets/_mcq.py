"""Shared MCQ formatting helpers for bucket-B adapters.

Kept tiny on purpose — each adapter still owns its load logic, but
the prompt-style helpers are common enough that duplicating them
across 10 files hurts readability.
"""

from __future__ import annotations

from typing import Sequence


def fmt_choices(choices: Sequence[str]) -> str:
    """``(A) first\\n(B) second\\n...`` over up to 10 options (A–J)."""
    return "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))


def fmt_mcq_demo(question: str, choices: Sequence[str],
                 answer_letter: str) -> str:
    return (f"Question: {question}\n"
            f"Options:\n{fmt_choices(choices)}\n"
            f"Answer: ({answer_letter})")


def fmt_mcq_query(question: str, choices: Sequence[str]) -> str:
    return (f"Question: {question}\n"
            f"Options:\n{fmt_choices(choices)}\n"
            f"Answer:")
