from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

from MemoryTest.prepare_data.prompt_templates import (
    build_context,
    format_answer,
    question_prompt,
    reconstruction_prompt,
)


MEMORY_TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMORY_TEST_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MEMORY_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMORY_TEST_ROOT))


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def read_json(path: str | Path) -> list[dict]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sample_context(rows: list[dict], fact_counts: list[int], qa_per_context: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    count = rng.choice(fact_counts)
    if len(rows) < count:
        raise ValueError(f"Need at least {count} rows, found {len(rows)}")
    context_rows = rng.sample(rows, count)
    qa_count = min(qa_per_context, len(context_rows))
    qa_rows = rng.sample(context_rows, qa_count)
    return context_rows, qa_rows


def encode_context(tokenizer, context: str, max_length: int, device):
    enc = tokenizer(
        [context],
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def trainable_generate_context_lora(context: str, metanetwork, tokenizer, metalora, cfg, device):
    evidence_ids, evidence_mask = encode_context(tokenizer, context, cfg.test.context_max_length, device)
    return metanetwork.generate_lora_dict(evidence_ids, evidence_mask, metalora)


def encode_answer_batch(tokenizer, qa_rows: list[dict], max_length: int, device):
    prompt_ids_list = []
    answer_ids_list = []
    for row in qa_rows:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question_prompt(row["question"])}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        answer_text = " " + format_answer(row["answer"])
        if tokenizer.eos_token:
            answer_text += tokenizer.eos_token
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        prompt_ids_list.append(prompt_ids)
        answer_ids_list.append(answer_ids)

    input_rows = []
    label_rows = []
    for prompt_ids, answer_ids in zip(prompt_ids_list, answer_ids_list):
        input_ids = (prompt_ids + answer_ids)[-max_length:]
        labels = ([-100] * len(prompt_ids) + answer_ids)[-max_length:]
        input_rows.append(input_ids)
        label_rows.append(labels)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row) for row in input_rows)
    padded_inputs = []
    padded_labels = []
    attention_mask = []
    for input_ids, labels in zip(input_rows, label_rows):
        pad_len = max_len - len(input_ids)
        padded_inputs.append([pad_id] * pad_len + input_ids)
        padded_labels.append([-100] * pad_len + labels)
        attention_mask.append([0] * pad_len + [1] * len(input_ids))

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
    }


def encode_reconstruction(tokenizer, context_rows: list[dict], max_length: int, device):
    structured = build_context(context_rows, context_format="structured")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": reconstruction_prompt()}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    answer_text = "\n" + structured
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[-max_length:]
    labels = ([-100] * len(prompt_ids) + answer_ids)[-max_length:]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
        "pad_id": pad_id,
    }


def set_posttrain_requires_grad(metanetwork, metalora):
    trainable = []

    def add_lora_tensors(obj: Any):
        if torch.is_tensor(obj):
            obj.requires_grad_(True)
            trainable.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                add_lora_tensors(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                add_lora_tensors(value)

    for _, param in metanetwork.named_parameters():
        param.requires_grad_(False)
    for param in metanetwork.metanetwork.parameters():
        param.requires_grad_(True)
        trainable.append(param)
    for name, param in metanetwork.metamodel.named_parameters():
        if "mem_tokens" in name:
            param.requires_grad_(True)
            trainable.append(param)
    add_lora_tensors(metalora)
    return trainable


def compute_answer_loss(context_rows, qa_rows, context_format, metanetwork, metalora, tokenizer, cfg, device, max_length):
    context = build_context(context_rows, context_format=context_format)
    lora_dict = trainable_generate_context_lora(context, metanetwork, tokenizer, metalora, cfg, device)
    batch = encode_answer_batch(tokenizer, qa_rows, max_length=max_length, device=device)
    outputs = metanetwork.metamodel(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        loradict=lora_dict,
        ignore_mem_token=True,
    )
    return outputs.loss, lora_dict, context


def compute_reconstruction_loss(context_rows, lora_dict, metanetwork, tokenizer, device, max_length):
    batch = encode_reconstruction(tokenizer, context_rows, max_length=max_length, device=device)
    outputs = metanetwork.metamodel(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        loradict=lora_dict,
        ignore_mem_token=True,
    )
    return outputs.loss


def save_posttrain_checkpoint(output_dir: str | Path, metanetwork, metalora, extra_state: dict | None = None):
    from utils.mysaveload import save_checkpoint

    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(metanetwork, str(output_dir), metalora, extra_state=extra_state or {})
