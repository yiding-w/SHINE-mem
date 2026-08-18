"""Pure-Python metrics shared by ControlSHINE evaluators."""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def answer_match(prediction: str, answer: str, aliases: list[str] | tuple[str, ...] = ()) -> bool:
    pred = f" {_normalize(prediction)} "
    candidates = [answer, *aliases]
    return any(candidate and f" {_normalize(candidate)} " in pred for candidate in candidates)


def recoverability_label(base_ok: bool, memory_ok: bool, context_ok: bool) -> str:
    if base_ok and memory_ok and context_ok:
        return "fully_recoverable"
    missing = [name for name, ok in (("base", base_ok), ("memory", memory_ok), ("context", context_ok)) if not ok]
    return "unrecoverable_" + "_".join(missing)
