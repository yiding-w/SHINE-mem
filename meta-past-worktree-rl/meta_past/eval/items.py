"""Canonical eval-item schema shared by all dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalItem:
    """One evaluation example, mode-agnostic.

    All three eval modes (``shine`` / ``icl`` / ``zero``) consume the
    same triple plus task-specific metadata:

    - ``context``: the text that ``shine`` mode compiles into a LoRA and
      that ``icl`` mode prepends to the prompt. Empty string for
      pure-zero-shot items (bucket C).
    - ``question``: the user-facing prompt for the LLM. For bucket B
      this is the held-out query (NOT the demos — demos live in
      ``context``).
    - ``references``: gold answers; the scorer decides what to do with
      them (F1 over the union, exact-match on the first, etc.).
    - ``metadata``: free-form bag for task-specific fields the scorer
      needs (e.g. MCQ ``choices``, BBH ``task_family``, HumanEval
      ``test`` / ``entry_point``).
    """

    context: str
    question: str
    references: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    # Identifier for logging / partial-result resume. Adapters should
    # set this to something stable across runs (e.g. f"squad/{qa_id}").
    item_id: str = ""
