from __future__ import annotations

import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from MemoryTest.prepare_data.prompt_templates import lora_sft_examples_for_fact

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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


def read_facts(path: str | Path) -> list[dict]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def select_fact_subset(rows: list[dict], num_facts: int, mode: str, seed: int, trial: int) -> list[dict]:
    if len(rows) < num_facts:
        raise ValueError(f"Need {num_facts} facts, found {len(rows)}")
    if mode == "head":
        return rows[:num_facts]
    if mode == "random":
        rng = random.Random(seed + trial * 10007 + num_facts)
        return rng.sample(rows, num_facts)
    raise ValueError(f"Unknown selection mode: {mode}")


def load_runtime_args(runtime_config_path: str | Path) -> Namespace:
    from MemoryTest.comparisons.experiment_utils import load_yaml_defaults, make_runtime_args

    return make_runtime_args(load_yaml_defaults(str(runtime_config_path)))


def load_frozen_lora_model(runtime_config_path: str | Path, rank: int, device_name: str | None = None, gpu_id: int | None = None):
    import torch
    from MemoryTest.case_test import build_cfg, load_tokenizer, resolve_device
    from utils.myinit import _import_class
    from utils.myseed import set_seed

    args = load_runtime_args(runtime_config_path)
    args.lora_r = int(rank)
    if device_name is not None:
        args.device = device_name
    if gpu_id is not None:
        args.gpu_id = int(gpu_id)

    set_seed(args.seed)
    device = resolve_device(args.device, args.gpu_id)
    cfg = build_cfg(args)
    cfg.model.lora_r = int(rank)

    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers
    cfg.num_mem_token = -1

    tokenizer = load_tokenizer(cfg)
    model = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
    model.resize_token_embeddings(len(tokenizer))
    freeze_backbone_for_lora_upper_bound(model)
    model.to(device)
    model.train()

    if hasattr(model, "set_generate_func"):
        model.set_generate_func(cfg.metanetwork.method)
    lora_dict = model.init_lora_dict(int(rank), scale=cfg.metanetwork.transformer_cfg.scale, device=device)
    return args, cfg, device, model, tokenizer, lora_dict


def freeze_backbone_for_lora_upper_bound(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def iter_lora_tensors(tree: Any):
    import torch

    if isinstance(tree, dict):
        for value in tree.values():
            yield from iter_lora_tensors(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from iter_lora_tensors(value)
    elif torch.is_tensor(tree) and tree.requires_grad:
        yield tree


def move_lora_to_cpu(tree: Any):
    import torch

    if torch.is_tensor(tree):
        return tree.detach().cpu()
    if isinstance(tree, dict):
        return {key: move_lora_to_cpu(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [move_lora_to_cpu(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(move_lora_to_cpu(value) for value in tree)
    return tree


def build_sft_records(facts: list[dict], seed: int, variants_per_fact: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    records = []
    for fact in facts:
        for example in lora_sft_examples_for_fact(fact, rng, max_variants=variants_per_fact):
            records.append(
                {
                    "fact_id": str(fact["id"]),
                    "attribute": str(fact.get("attribute", fact.get("relation", "unknown"))),
                    "prompt": example["prompt"],
                    "answer": example["answer"],
                    "kind": example["kind"],
                }
            )
    rng.shuffle(records)
    return records


def encode_answer_only(tokenizer, prompt: str, answer: str, max_length: int):
    user_messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        user_messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    answer_text = " " + str(answer).strip()
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[-max_length:]
    labels = ([-100] * len(prompt_ids) + answer_ids)[-max_length:]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def collate_batch(tokenizer, encoded_rows: list[dict], device):
    import torch

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row["input_ids"]) for row in encoded_rows)
    input_ids = []
    labels = []
    attention_mask = []
    for row in encoded_rows:
        pad_len = max_len - len(row["input_ids"])
        input_ids.append([pad_id] * pad_len + row["input_ids"])
        labels.append([-100] * pad_len + row["labels"])
        attention_mask.append([0] * pad_len + row["attention_mask"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def train_lora_dict(
    model,
    tokenizer,
    lora_dict,
    facts: list[dict],
    device,
    seed: int,
    variants_per_fact: int,
    max_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    progress_label: str | None = None,
) -> dict:
    import torch

    records = build_sft_records(facts, seed=seed, variants_per_fact=variants_per_fact)
    encoded = [encode_answer_only(tokenizer, row["prompt"], row["answer"], max_length=max_length) for row in records]
    optimizer = torch.optim.AdamW(list(iter_lora_tensors(lora_dict)), lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(seed)
    step = 0
    losses = []

    total_steps = math.ceil(len(encoded) / batch_size) * epochs if encoded else 0
    progress = tqdm(
        total=total_steps,
        desc=progress_label or "LoRA SFT",
        dynamic_ncols=True,
        leave=True,
    ) if tqdm is not None else None

    for epoch in range(epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_rows = [encoded[idx] for idx in order[start : start + batch_size]]
            batch = collate_batch(tokenizer, batch_rows, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                loradict=lora_dict,
                ignore_mem_token=True,
            )
            loss = outputs.loss
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(iter_lora_tensors(lora_dict)), grad_clip_norm)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            step += 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix({"epoch": epoch + 1, "loss": f"{loss_value:.4f}"})

    if progress is not None:
        progress.close()

    return {
        "num_records": len(records),
        "num_steps": step,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "losses": losses,
    }


def print_summary_table(summary_rows: list[dict]) -> None:
    header = "rank | num_facts | train_acc_mean | train_acc_std | test_acc_mean | test_acc_std"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['rank']} | {row['num_facts']} | "
            f"{row['train_acc_mean']:.4f} | {row['train_acc_std']:.4f} | "
            f"{row['test_acc_mean']:.4f} | {row['test_acc_std']:.4f}"
        )


def summarize_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)
