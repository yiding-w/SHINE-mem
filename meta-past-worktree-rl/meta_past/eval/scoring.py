"""Per-task scorers. All return a float in [0, 1].

Reuses the existing F1 normalizer from ``meta_past.reward.f1_reward``
for short-span tasks; adds MCQ-letter / numeric / pass@1 scorers for
the bucket-B and bucket-C tasks the F1 path doesn't handle well.
"""

from __future__ import annotations

import re
import string
from typing import Callable, Sequence

from ..reward.f1_reward import f1_reward


# ---------------------------------------------------------------------------
# Short-span (F1 / EM)
# ---------------------------------------------------------------------------


def f1(pred: str, refs: Sequence[str], **_) -> float:
    """SQuAD-style token F1, picking the max over multiple gold refs."""
    return float(f1_reward(pred, refs))


def _norm_em(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em(pred: str, refs: Sequence[str], **_) -> float:
    """Exact match after SQuAD-style normalization (lower / strip punct)."""
    p = _norm_em(pred)
    return float(any(_norm_em(r) == p for r in refs))


# ---------------------------------------------------------------------------
# MCQ letter
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)
_LETTER_PAREN_RE = re.compile(r"\(([A-J])\)", re.IGNORECASE)


def extract_letter(pred: str, n_options: int = 4) -> str | None:
    """Best-effort letter extraction from a model completion.

    Looks for ``(A)`` first, then a standalone ``A``. Returns the
    UPPERCASE letter, or ``None`` if nothing valid surfaced.
    """
    if not pred:
        return None
    m = _LETTER_PAREN_RE.search(pred)
    if m is None:
        m = _LETTER_RE.search(pred)
    if m is None:
        return None
    letter = m.group(1).upper()
    if ord(letter) - ord("A") >= n_options:
        return None
    return letter


def mcq_letter(pred: str, refs: Sequence[str], *, n_options: int = 4, **_) -> float:
    """1.0 iff the extracted letter matches any gold ref (also a letter)."""
    got = extract_letter(pred, n_options=n_options)
    if got is None:
        return 0.0
    gold = {r.strip().upper().strip("()") for r in refs if r}
    return float(got in gold)


# ---------------------------------------------------------------------------
# Numeric (GSM8K, DROP-number)
# ---------------------------------------------------------------------------


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(s: str) -> float | None:
    """Pull the last number out of a string. ``None`` if no number found."""
    if not s:
        return None
    matches = _NUM_RE.findall(s)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def numeric_em(pred: str, refs: Sequence[str], **_) -> float:
    """1.0 iff the predicted last-number equals (numerically) any gold."""
    p = extract_number(pred)
    if p is None:
        return 0.0
    for r in refs:
        g = extract_number(r)
        if g is None:
            continue
        # Allow tiny float drift for integer-equivalent answers.
        if abs(p - g) < 1e-6 or (abs(p) > 0 and abs(p - g) / max(abs(p), abs(g)) < 1e-4):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Pass@1 (HumanEval)
# ---------------------------------------------------------------------------


def humaneval_pass_at_1(pred: str, refs: Sequence[str], *,
                       test_code: str = "", entry_point: str = "",
                       prompt: str = "", timeout_s: float = 5.0, **_) -> float:
    """Execute the model's completion against the dataset's test script.

    Sandboxing here is **not** safe for arbitrary input — for HumanEval
    we run trusted dataset test code only. The completion is appended
    to the function header from ``prompt`` (so the function definition
    is complete) then ``test_code`` is exec'd in the same namespace.
    """
    if not entry_point or not test_code:
        return 0.0
    # The model may have echoed the prompt; strip if present.
    completion = pred
    if prompt and completion.startswith(prompt):
        completion = completion[len(prompt):]
    program = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"

    import multiprocessing as mp
    def _worker(prog, q):
        try:
            exec(compile(prog, "<humaneval>", "exec"), {"__name__": "__main__"})
            q.put(True)
        except Exception:
            q.put(False)

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(program, q))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(0.1)
        return 0.0
    try:
        return 1.0 if q.get_nowait() else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ScoreFn = Callable[..., float]

SCORERS: dict[str, ScoreFn] = {
    "f1": f1,
    "em": em,
    "mcq_letter": mcq_letter,
    "numeric_em": numeric_em,
    "humaneval_pass1": humaneval_pass_at_1,
}


def aggregate(scores: Sequence[float]) -> float:
    """Mean over a list of per-item scores. 0.0 on empty list."""
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
