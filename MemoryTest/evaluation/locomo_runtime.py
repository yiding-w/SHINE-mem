"""Model-facing LoCoMo probe for the recurrent SHINE post-training loop."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from MemoryTest.evaluation.locomo_probe import (
    CATEGORY_NAMES,
    build_question_messages,
    build_session_texts,
    canonicalize_prediction,
    distance_bucket,
    evidence_distance,
    prepare_question,
    score_prediction,
    summarize_records,
)
from MemoryTest.training.shine_train_utils import (
    clamp_lora_tensors,
    trainable_generate_context_lora,
    trainable_update_recurrent_memory,
)


LOGGER = logging.getLogger("posttrain_shine_memory")


def _generate_from_messages(
    metanetwork,
    tokenizer,
    lora_dict,
    messages: list[dict[str, str]],
    device,
    max_new_tokens: int,
    max_length: int,
) -> tuple[str, str]:
    from MemoryTest.case_test import extract_think_and_answer

    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        max_length=max_length,
        truncation=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = metanetwork.metamodel.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_cache=True,
        ignore_mem_token=True,
        loradict=lora_dict,
    )
    raw = tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
    _, answer = extract_think_and_answer(raw)
    return (answer.strip() if answer.strip() else raw), raw


@torch.no_grad()
def evaluate_locomo_probe(
    metanetwork,
    metalora,
    tokenizer,
    cfg,
    args: argparse.Namespace,
    device,
    sample: dict,
    questions: list[dict],
    session_numbers_to_use: list[int],
) -> dict:
    """Sequentially ingest one LoCoMo conversation and run fixed memory-only QA."""
    was_training = metanetwork.training
    metanetwork.eval()
    allowed_sessions = set(session_numbers_to_use)
    sessions = [
        session
        for session in build_session_texts(sample)
        if session["session_number"] in allowed_sessions
    ]
    if [session["session_number"] for session in sessions] != session_numbers_to_use:
        raise ValueError(
            f"Requested LoCoMo session window {session_numbers_to_use}, found "
            f"{[session['session_number'] for session in sessions]}"
        )
    memory_position_offset = args.memory_position_offset or cfg.test.context_max_length
    session_token_counts = [
        len(tokenizer(session["text"], add_special_tokens=True)["input_ids"])
        for session in sessions
    ]
    truncated_sessions = sum(
        token_count > cfg.test.context_max_length for token_count in session_token_counts
    )

    recurrent_memory = None
    recurrent_lora = None
    for session_index, session in enumerate(sessions):
        is_final = session_index + 1 == len(sessions)
        if is_final:
            recurrent_lora, recurrent_memory = trainable_generate_context_lora(
                session["text"],
                metanetwork,
                tokenizer,
                metalora,
                cfg,
                device,
                recurrent_memory=recurrent_memory,
                return_recurrent_state=True,
                memory_position_offset=memory_position_offset,
                recurrent_memory_policy=args.recurrent_memory_policy,
                recurrent_memory_max_banks=args.recurrent_memory_max_banks,
            )
            recurrent_lora = clamp_lora_tensors(recurrent_lora, args.generated_lora_clamp)
        else:
            recurrent_memory = trainable_update_recurrent_memory(
                session["text"],
                metanetwork,
                tokenizer,
                metalora,
                cfg,
                device,
                recurrent_memory=recurrent_memory,
                memory_position_offset=memory_position_offset,
                recurrent_memory_policy=args.recurrent_memory_policy,
                recurrent_memory_max_banks=args.recurrent_memory_max_banks,
            )

    condition_loras = {"recurrent": recurrent_lora}
    if args.locomo_eval_last_session_ablation:
        last_session_lora = trainable_generate_context_lora(
            sessions[-1]["text"],
            metanetwork,
            tokenizer,
            metalora,
            cfg,
            device,
            recurrent_memory=None,
            return_recurrent_state=False,
            memory_position_offset=memory_position_offset,
        )
        condition_loras["last_session_only"] = clamp_lora_tensors(
            last_session_lora,
            args.generated_lora_clamp,
        )

    records = []
    final_session_number = sessions[-1]["session_number"]
    for question in questions:
        spec = prepare_question(question, str(sample["sample_id"]), args.seed)
        messages = build_question_messages(spec)
        distance = evidence_distance(question, final_session_number)
        record = {
            "question_index": question["_question_index"],
            "question": question["question"],
            "answer": question.get("answer"),
            "category": int(question["category"]),
            "category_name": CATEGORY_NAMES.get(int(question["category"]), "unknown"),
            "evidence": list(question.get("evidence", [])),
            "evidence_distance": distance,
            "evidence_distance_bucket": distance_bucket(distance),
            "conditions": {},
        }
        for condition_name, lora_dict in condition_loras.items():
            answer, raw = _generate_from_messages(
                metanetwork,
                tokenizer,
                lora_dict,
                messages,
                device,
                args.locomo_eval_max_new_tokens,
                args.locomo_eval_question_max_length,
            )
            prediction = canonicalize_prediction(answer, spec)
            record["conditions"][condition_name] = {
                "prediction": prediction,
                "raw_prediction": raw,
                "score": score_prediction(question, prediction),
            }
        records.append(record)

    conditions = summarize_records(records, list(condition_loras))
    summary = {
        "sample_id": str(sample["sample_id"]),
        "num_sessions": len(sessions),
        "session_numbers": session_numbers_to_use,
        "max_session_tokens": max(session_token_counts),
        "truncated_sessions": truncated_sessions,
        "num_questions": len(questions),
        "categories": list(args.locomo_eval_categories),
        "questions_per_category": args.locomo_eval_questions_per_category,
        "conditions": conditions,
        "records": records,
    }
    if "last_session_only" in conditions:
        summary["recurrent_gain_over_last_session"] = (
            conditions["recurrent"]["overall_score"]
            - conditions["last_session_only"]["overall_score"]
        )
        summary["recurrent_macro_gain_over_last_session"] = (
            conditions["recurrent"]["macro_category_score"]
            - conditions["last_session_only"]["macro_category_score"]
        )
    if was_training:
        metanetwork.train()
    return summary


def report_locomo_probe(step: int, summary: dict, print_examples: bool) -> None:
    condition_text = ", ".join(
        f"{name}={values['overall_score']:.4f}/macro={values['macro_category_score']:.4f}"
        for name, values in summary["conditions"].items()
    )
    gain = summary.get("recurrent_gain_over_last_session")
    step0_gain = summary.get("recurrent_gain_from_step0")
    LOGGER.info(
        "LoCoMo probe step=%s sample=%s session_window=%s truncated_sessions=%s "
        "max_session_tokens=%s questions=%s %s recurrent_gain=%s step0_gain=%s",
        step,
        summary["sample_id"],
        summary["session_numbers"],
        summary["truncated_sessions"],
        summary["max_session_tokens"],
        summary["num_questions"],
        condition_text,
        f"{gain:.4f}" if gain is not None else "n/a",
        f"{step0_gain:.4f}" if step0_gain is not None else "n/a",
    )
    for condition_name, values in summary["conditions"].items():
        category_text = ", ".join(
            f"{category['name']}={category['score']:.4f}(n={category['count']})"
            for category in values["category_scores"].values()
        )
        distance_text = ", ".join(
            f"{bucket}={bucket_values['score']:.4f}(n={bucket_values['count']})"
            for bucket, bucket_values in values["distance_scores"].items()
        )
        LOGGER.info(
            "LoCoMo probe step=%s condition=%s categories=[%s] distance=[%s]",
            step,
            condition_name,
            category_text,
            distance_text,
        )
    if not print_examples:
        return
    seen_categories = set()
    for record in summary["records"]:
        category = record["category"]
        if category in seen_categories:
            continue
        seen_categories.add(category)
        recurrent = record["conditions"]["recurrent"]
        ablation = record["conditions"].get("last_session_only")
        LOGGER.info(
            "LoCoMo example step=%s category=%s distance=%s score=%.4f\n"
            "----- question -----\n%s\n"
            "----- reference -----\n%s\n"
            "----- recurrent prediction -----\n%s\n"
            "----- last-session-only prediction -----\n%s\n"
            "----- end LoCoMo example -----",
            step,
            record["category_name"],
            record["evidence_distance"],
            recurrent["score"],
            record["question"],
            record["answer"],
            recurrent["prediction"],
            ablation["prediction"] if ablation is not None else "n/a",
        )


def locomo_wandb_payload(summary: dict) -> dict:
    payload = {
        "locomo/num_sessions": summary["num_sessions"],
        "locomo/max_session_tokens": summary["max_session_tokens"],
        "locomo/truncated_sessions": summary["truncated_sessions"],
        "locomo/num_questions": summary["num_questions"],
    }
    if "recurrent_gain_over_last_session" in summary:
        payload["locomo/recurrent_gain_over_last_session"] = summary[
            "recurrent_gain_over_last_session"
        ]
        payload["locomo/recurrent_macro_gain_over_last_session"] = summary[
            "recurrent_macro_gain_over_last_session"
        ]
    if "recurrent_gain_from_step0" in summary:
        payload["locomo/recurrent_gain_from_step0"] = summary["recurrent_gain_from_step0"]
    for condition_name, values in summary["conditions"].items():
        prefix = f"locomo/{condition_name}"
        payload[f"{prefix}/overall_score"] = values["overall_score"]
        payload[f"{prefix}/macro_category_score"] = values["macro_category_score"]
        for category in values["category_scores"].values():
            payload[f"{prefix}/category_{category['name']}"] = category["score"]
        for bucket, bucket_values in values["distance_scores"].items():
            payload[f"{prefix}/distance_{bucket}"] = bucket_values["score"]
    return payload


def save_locomo_probe(output_dir: Path, step: int, summary: dict) -> Path:
    probe_dir = output_dir / "locomo_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    path = probe_dir / f"step_{step:08d}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
