"""LoCoMo evaluation for a loaded SHINE-V2 / SHINE-v1-backend model.

This deliberately reuses delta-Mem's LoCoMo protocol parser and scorer, while
keeping model loading inside ``meta_train_tp.py``.  It therefore evaluates the
same checkpoint/configuration as MEMORY_QA_GEN without requiring the delta-Mem
virtualenv to import SHINE.

Set ``LOCOMO_SHINE_MODE=full`` to encode all conversation sessions together
into one SHINE context.  Set it to ``sequential`` to encode each session (or
each individual turn) in chronological order and accumulate detach-state W.
The latter requires a non-empty ``detach_state`` configuration.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from eval_memory_gen import (
    _conversation_loradict,
    _llm_model_forward,
    _model_path_from_cfg,
    _prepare_forward_loradict,
    _strip_think,
    _write_and_maybe_reset,
)
from utils.mytokenizer import NOTHINKING_CHAT_TEMPLATE, create_tokenizer


def _import_locomo_protocol():
    """Import the vendored delta-Mem evaluator protocol without its runtime."""
    configured = os.environ.get("DELTA_MEM_ROOT")
    defaults = [
        configured,
        str(Path(__file__).resolve().parents[1] / "third_party" / "delta-Mem"),
    ]
    for root in defaults:
        if root and (Path(root) / "deltamem" / "eval" / "locomo_protocol.py").is_file():
            if root not in sys.path:
                sys.path.insert(0, root)
            from deltamem.eval import locomo_protocol

            return locomo_protocol
    raise FileNotFoundError(
        "Could not find delta-Mem's LoCoMo protocol. Set DELTA_MEM_ROOT to the "
        "directory containing deltamem/eval/locomo_protocol.py."
    )


def _history_segments(protocol, sample: dict, granularity: str) -> list[list[dict[str, str]]]:
    """Return chronological context segments, excluding the shared system prompt."""
    conversation = sample["conversation"]
    session_numbers = sorted(
        int(key.split("_")[-1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("date_time")
    )
    segments: list[list[dict[str, str]]] = []
    for session_num in session_numbers:
        if granularity == "session":
            segments.append([protocol.build_session_message(conversation, session_num)])
        else:
            turns = conversation[f"session_{session_num}"]
            segments.extend(
                [[protocol.build_turn_message(conversation, session_num, turn)] for turn in turns]
            )
    return segments


def _context_tensor(tokenizer, messages, device, max_tokens: int, num_mem_tokens: int):
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    # Keep the newest evidence when the configured SHINE context budget is exceeded.
    ids = ids[-max_tokens:]
    pad_id = tokenizer.pad_token_id or 0
    context = torch.full(
        (1, max_tokens + num_mem_tokens), pad_id, dtype=torch.long, device=device
    )
    context[0, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return context, torch.tensor([len(ids)], dtype=torch.long, device=device), len(ids)


def _generate_answer(model, tokenizer, device, prompt_messages, loradict, ds_l, ds_w, max_new: int) -> str:
    """Greedy decoding through SHINE's dynamically adapted backbone."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )
    current = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    past_key_values = None
    if hasattr(model, "_active_w_transform"):
        model._active_w_transform = getattr(model, "w_transform_conversation", None)
    try:
        for _ in range(max_new):
            output = _llm_model_forward(
                model,
                input_ids=current,
                loradict=loradict,
                ds_l=ds_l,
                ds_w=ds_w,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = output.past_key_values
            token = int(model.llm.lm_head(output.last_hidden_state[:, -1, :]).argmax(-1).item())
            if token == im_end or token == eos_id:
                break
            generated.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=device)
    finally:
        if hasattr(model, "_active_w_transform"):
            model._active_w_transform = None
    return _strip_think(tokenizer.decode(generated, skip_special_tokens=True))


def _evaluate_sample(
    *,
    model,
    tokenizer,
    device,
    protocol,
    sample: dict,
    mode: str,
    granularity: str,
    max_context_tokens: int,
    decoder_context_modes: list[str],
    decoder_context_max_tokens: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build one sample's SHINE memory, then score all of its LoCoMo questions."""
    num_mem_tokens = int(getattr(model, "_num_mem_token", 0))
    num_contexts = 0
    if getattr(model, "detach_state", None) is not None:
        model.detach_state.reset()

    if mode == "full":
        history = protocol.build_locomo_history_messages(sample, message_granularity=granularity)
        ctx_ids, ctx_lens, context_len = _context_tensor(
            tokenizer, history, device, max_context_tokens, num_mem_tokens
        )
        ds_l, ds_w = model._read_detach_state()
        memory = model.compute_memory_states(
            ctx_ids, None, ctx_lens, nograd_loradict=ds_l, nograd_wdict=ds_w
        )
        generated_lora = model.generate_loradict(memory)
        answer_lora = _conversation_loradict(model, generated_lora)
        num_contexts = 1
    else:
        for segment in _history_segments(protocol, sample, granularity):
            ctx_ids, ctx_lens, _ = _context_tensor(
                tokenizer, segment, device, max_context_tokens, num_mem_tokens
            )
            ds_l, ds_w = model._read_detach_state()
            memory = model.compute_memory_states(
                ctx_ids, None, ctx_lens, nograd_loradict=ds_l, nograd_wdict=ds_w
            )
            generated_lora = model.generate_loradict(memory)
            _write_and_maybe_reset(model, generated_lora)
            num_contexts += 1
        ds_l, ds_w = model._read_detach_state()
        # Native V2 uses meta-LoRA; V1's adapter turns None into state-only LoRA.
        answer_lora = _conversation_loradict(model, None)
        context_len = max_context_tokens

    records: list[dict[str, Any]] = []
    for question_index, qa in enumerate(sample["qa"]):
        spec = protocol.prepare_locomo_question(
            qa,
            sample_id=str(sample["sample_id"]),
            question_index=question_index,
            seed=seed,
        )
        question_prompt = protocol.build_official_question_prompt(spec)
        conditions = {}
        for decoder_context_mode in decoder_context_modes:
            if decoder_context_mode == "memory_only":
                prompt_messages = [{"role": "user", "content": question_prompt}]
            else:
                # This is delta-Mem's official_prompt construction: the decoder
                # receives the raw (window-truncated) conversation plus question.
                prompt_messages = protocol.build_official_full_history_messages(
                    sample,
                    tokenizer,
                    spec,
                    max_context_tokens=decoder_context_max_tokens,
                )
            raw_prediction = _generate_answer(
                model,
                tokenizer,
                device,
                prompt_messages,
                answer_lora,
                ds_l,
                ds_w,
                max_new_tokens,
            )
            prediction = protocol.canonicalize_locomo_prediction(raw_prediction, spec)
            condition_name = f"shine_{mode}_{decoder_context_mode}"
            conditions[condition_name] = {
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "score": round(protocol.score_locomo_prediction(qa, prediction), 4),
                "turn_stats": {
                    "mode": mode,
                    "decoder_context_mode": decoder_context_mode,
                    "segment_granularity": granularity,
                    "num_contexts": num_contexts,
                    "context_max_tokens": max_context_tokens,
                    "decoder_context_max_tokens": (
                        decoder_context_max_tokens
                        if decoder_context_mode == "official_prompt"
                        else None
                    ),
                    "full_context_tokens": context_len if mode == "full" else None,
                },
            }
        records.append(
            {
                "question": qa["question"],
                "answer": qa.get("answer"),
                "adversarial_answer": qa.get("adversarial_answer"),
                "evidence": list(qa.get("evidence", [])),
                "category": int(qa["category"]),
                "conditions": conditions,
            }
        )
    return records, {"num_contexts": num_contexts}


@torch.no_grad()
def run_locomo_shine_gen(model, cfg, tp_cfg, my_device) -> None:
    """Evaluate a loaded SHINE model on delta-Mem's LoCoMo JSON/protocol."""
    from utils.myparallel import barrier, is_main_process

    if not is_main_process():
        barrier()
        return
    if int(tp_cfg.get("tensor_parallel_size", 1)) != 1:
        raise ValueError("LOCOMO_SHINE_GEN currently requires parallel.tensor_parallel_size=1.")

    protocol = _import_locomo_protocol()
    mode = os.environ.get("LOCOMO_SHINE_MODE", "sequential").strip().lower()
    if mode not in {"full", "sequential"}:
        raise ValueError("LOCOMO_SHINE_MODE must be 'full' or 'sequential'.")
    granularity = os.environ.get("LOCOMO_SEGMENT_GRANULARITY", "session").strip().lower()
    if granularity not in {"session", "turn"}:
        raise ValueError("LOCOMO_SEGMENT_GRANULARITY must be 'session' or 'turn'.")
    raw_decoder_context_mode = os.environ.get(
        "LOCOMO_DECODER_CONTEXT", "memory_only"
    ).strip().lower()
    if raw_decoder_context_mode == "both":
        decoder_context_modes = ["memory_only", "official_prompt"]
    elif raw_decoder_context_mode in {"memory_only", "official_prompt"}:
        decoder_context_modes = [raw_decoder_context_mode]
    else:
        raise ValueError(
            "LOCOMO_DECODER_CONTEXT must be 'memory_only', 'official_prompt', or 'both'."
        )
    detach_cfg = cfg.get("detach_state", None) or {}
    detach_type = str(detach_cfg.get("type", "empty"))
    if mode == "sequential" and (
        getattr(model, "detach_state", None) is None or detach_type == "empty"
    ):
        raise ValueError(
            "Sequential LoCoMo requires accumulated W. Relaunch with detach_state=full "
            "(detach_state=origin is type=empty and cannot retain prior segments)."
        )

    data_file = Path(
        os.environ.get(
            "LOCOMO_DATA_FILE",
            str(Path(os.environ.get("DELTA_MEM_ROOT", "")) / "data" / "locomo10.json"),
        )
    )
    if not data_file.is_file():
        raise FileNotFoundError(f"LoCoMo data file not found: {data_file}")
    max_context_tokens = int(
        os.environ.get("LOCOMO_CONTEXT_MAX_TOKENS", str(cfg.data.get("context_seq_length", 2048)))
    )
    decoder_context_max_tokens = int(
        os.environ.get("LOCOMO_DECODER_CONTEXT_MAX_TOKENS", "32768")
    )
    max_new_tokens = int(os.environ.get("LOCOMO_MAX_NEW_TOKENS", "50"))
    seed = int(os.environ.get("LOCOMO_SEED", "42"))
    limit = int(os.environ.get("LOCOMO_NUM_CONVERSATIONS", "-1"))
    categories = {int(x) for x in os.environ.get("LOCOMO_CATEGORIES", "1,2,3,4").split(",")}
    output_json = Path(os.environ.get("LOCOMO_OUTPUT_JSON", "outputs/locomo_shine.json"))

    raw_samples = json.loads(data_file.read_text(encoding="utf-8"))
    samples = [
        sample
        for sample in raw_samples
        if any(int(qa["category"]) in categories for qa in sample.get("qa", []))
    ]
    for sample in samples:
        sample["qa"] = [qa for qa in sample["qa"] if int(qa["category"]) in categories]
    if limit >= 0:
        samples = samples[:limit]

    tokenizer = create_tokenizer(
        _model_path_from_cfg(cfg),
        tokenizer_cfg=cfg.get("tokenizer", None),
        chat_template=NOTHINKING_CHAT_TEMPLATE,
    )
    condition_names = [f"shine_{mode}_{context_mode}" for context_mode in decoder_context_modes]
    records = []
    context_manager = (
        model.detach_state.eval_context()
        if getattr(model, "detach_state", None) is not None
        else nullcontext()
    )
    was_training = model.training
    model.eval()
    try:
        with context_manager:
            for sample_index, sample in enumerate(samples, start=1):
                qa_records, _ = _evaluate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=my_device,
                    protocol=protocol,
                    sample=sample,
                    mode=mode,
                    granularity=granularity,
                    max_context_tokens=max_context_tokens,
                    decoder_context_modes=decoder_context_modes,
                    decoder_context_max_tokens=decoder_context_max_tokens,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "speakers": {
                            "speaker_a": sample["conversation"]["speaker_a"],
                            "speaker_b": sample["conversation"]["speaker_b"],
                        },
                        "num_sessions": len(_history_segments(protocol, sample, "session")),
                        "qa": qa_records,
                    }
                )
                print(
                    f"[locomo_shine] {sample_index}/{len(samples)} "
                    f"sample_id={sample['sample_id']}",
                    flush=True,
                )
    finally:
        if was_training:
            model.train()

    payload = {
        "model_path": _model_path_from_cfg(cfg),
        "checkpoint": str(cfg.training.get("resume_from", "")),
        "data_file": str(data_file),
        "num_conversations": len(records),
        "num_questions": sum(len(record["qa"]) for record in records),
        "categories": sorted(categories),
        "shine": {
            "mode": mode,
            "segment_granularity": granularity,
            "context_max_tokens": max_context_tokens,
            "decoder_context_modes": decoder_context_modes,
            "decoder_context_max_tokens": decoder_context_max_tokens,
            "max_new_tokens": max_new_tokens,
            "records": records,
            "summary": protocol.summarize_locomo_records(
                records, condition_names=condition_names
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = payload["shine"]["summary"]
    print(
        f"\n[locomo_shine] mode={mode} decoder_context={','.join(decoder_context_modes)} "
        f"summary={json.dumps(summary)} -> {output_json}",
        flush=True,
    )
    barrier()
