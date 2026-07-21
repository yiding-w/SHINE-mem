from __future__ import annotations

from MemoryTest.prepare_data.prompt_templates import (
    cumulative_completion_prompt,
    single_session_completion_prompt,
)


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


def build_single_session_retention_records(tokenizer, observed_turns, args, rng) -> list[dict]:
    """Build equally weighted, suffix-only readouts for every observed session."""
    records = []
    observed_sessions = len(observed_turns)
    decode_kwargs = {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
    for source_index, turn in enumerate(observed_turns):
        content_ids = tokenizer(turn.text, add_special_tokens=False)["input_ids"]
        if not content_ids:
            raise ValueError(
                "Single-session retention requires a non-empty session, "
                f"but turn {turn.turn_id!r} is empty."
            )
        prefix_ratio = rng.uniform(
            args.completion_prefix_min_ratio,
            args.completion_prefix_max_ratio,
        )
        # Always leave at least one source token for reconstruction supervision.
        prefix_tokens = max(1, round(len(content_ids) * prefix_ratio))
        prefix_tokens = min(prefix_tokens, max(0, len(content_ids) - 1))
        session_prefix = tokenizer.decode(content_ids[:prefix_tokens], **decode_kwargs)
        target_suffix = tokenizer.decode(content_ids[prefix_tokens:], **decode_kwargs)
        source_turn = source_index + 1
        records.append(
            {
                "category": "reconstruction",
                "prompt": single_session_completion_prompt(
                    session_prefix,
                    source_turn,
                    observed_sessions,
                ),
                "answer": target_suffix,
                "reference": turn.text,
                "session_prefix": session_prefix,
                "prediction_prefix": session_prefix,
                "preserve_generation_whitespace": True,
                "source_turn": source_turn,
                "source_turn_id": turn.turn_id,
                # Each session first receives its own token mean; those row means
                # are then averaged, so neither length nor tail overlap changes its weight.
                "loss_reduction": "record_mean",
            }
        )
    return records
