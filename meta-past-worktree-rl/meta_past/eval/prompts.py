"""Prompt assembly for the three eval modes.

All modes share the same chat template + ``enable_thinking`` / ``force_think``
flags the trainer uses, so prompts the model sees at eval match training as
closely as possible.
"""

from __future__ import annotations

from typing import Any

from ..rollout.squad_rollout import _SYS_PROMPT


def _chat_kwargs(enable_thinking: bool | None) -> dict[str, Any]:
    if enable_thinking is None:
        return {}
    return {"enable_thinking": bool(enable_thinking)}


def _render(tokenizer, messages, *, enable_thinking: bool | None,
            force_think: bool) -> str:
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        **_chat_kwargs(enable_thinking),
    )
    if force_think:
        # Forces the model to start mid-thinking; it must emit </think>
        # before the answer. _extract_answer handles the asymmetric tags.
        text = text + "<think>\n"
    return text


def format_shine_prompt(tokenizer, question: str, *,
                        enable_thinking: bool | None = True,
                        force_think: bool = False) -> str:
    """Mode = ``shine``: the LoRA already encodes the context, so the
    prompt is just the question.
    """
    messages = [
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": question},
    ]
    return _render(tokenizer, messages,
                   enable_thinking=enable_thinking, force_think=force_think)


def format_icl_prompt(tokenizer, context: str, question: str, *,
                      enable_thinking: bool | None = True,
                      force_think: bool = False) -> str:
    """Mode = ``icl``: pack the context into the prompt body, no LoRA.

    For bucket-A passages the context is a passage; for bucket-B
    in-parameter few-shot data the context is the K demonstrations.
    Either way, this is the apples-to-apples baseline against ``shine``
    mode at matched K.
    """
    user_content = f"{context}\n\n{question}" if context else question
    messages = [
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return _render(tokenizer, messages,
                   enable_thinking=enable_thinking, force_think=force_think)


def format_zero_prompt(tokenizer, question: str, *,
                       enable_thinking: bool | None = True,
                       force_think: bool = False) -> str:
    """Mode = ``zero``: no LoRA, no context — just the question."""
    return format_shine_prompt(tokenizer, question,
                               enable_thinking=enable_thinking,
                               force_think=force_think)


def extract_answer(text: str) -> str:
    """Strip Qwen3's ``<think>...</think>`` block + common answer prefixes.

    Mirrors ``meta_past.rollout.squad_rollout._extract_answer`` (kept in
    sync). Handles three completion shapes:
      * ``...<think>X</think>Y``  — model emitted its own thinking block.
      * ``X</think>Y``            — force_think prefilled ``<think>\\n`` in
                                    the prompt; only the closing tag is in
                                    the completion.
      * ``Y``                     — no thinking, just the answer.
    """
    import re
    if "<think>" in text:
        rest = text.split("<think>", 1)[1]
        if "</think>" in rest:
            text = rest.split("</think>", 1)[1]
        else:
            text = ""
    elif "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text.strip(),
                  flags=re.IGNORECASE)
    return text.strip()
