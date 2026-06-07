from __future__ import annotations


def unavailable_forced_choice_result(reason: str = "not_implemented") -> dict:
    return {
        "available": False,
        "reason": reason,
        "top1": None,
        "top3": None,
        "gold_rank": None,
    }
