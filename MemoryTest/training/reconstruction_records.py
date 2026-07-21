from __future__ import annotations

from MemoryTest.prepare_data.prompt_templates import cumulative_completion_prompt


def build_prefix_cumulative_record(tokenizer, observed_turns, args, rng) -> dict:
    """Build one prefix-addressed target covering session 1 through the current turn."""
    first_turn = observed_turns[0]
    first_ids = tokenizer(first_turn.text, add_special_tokens=False)["input_ids"]
    if not first_ids:
        raise ValueError(
            "Prefix-cumulative completion requires a non-empty first session, "
            f"but turn {first_turn.turn_id!r} is empty."
        )
    expose_empty = rng.random() < args.prefix_cumulative_empty_probability
    if expose_empty:
        prefix_tokens = 0
    else:
        prefix_ratio = rng.uniform(0.0, args.prefix_cumulative_max_prefix_ratio)
        # A non-empty branch must expose an actual address, while leaving at
        # least one token from session 1 for reconstruction supervision.
        prefix_tokens = max(1, round(len(first_ids) * prefix_ratio))
        prefix_tokens = min(prefix_tokens, max(0, len(first_ids) - 1))
    decode_kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    history_prefix = tokenizer.decode(first_ids[:prefix_tokens], **decode_kwargs)
    first_suffix = tokenizer.decode(first_ids[prefix_tokens:], **decode_kwargs)
    later_history = "\n".join(turn.text for turn in observed_turns[1:])
    target_suffix = first_suffix + (f"\n{later_history}" if later_history else "")
    complete_history = "\n".join(turn.text for turn in observed_turns)
    return {
        "category": "reconstruction",
        "prompt": cumulative_completion_prompt(history_prefix, len(observed_turns)),
        "answer": target_suffix,
        "reference": complete_history,
        "session_prefix": history_prefix,
        "prediction_prefix": history_prefix,
        "preserve_generation_whitespace": True,
        "source_turn": 1,
        "source_turn_id": first_turn.turn_id,
    }
