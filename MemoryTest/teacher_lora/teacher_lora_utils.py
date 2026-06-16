from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from MemoryTest.prepare_data.prompt_templates import build_context
from MemoryTest.training.lora_sft_utils import move_lora_to_cpu, resolve_path


def write_json(path: str | Path, payload: dict) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def make_context_id(index: int) -> str:
    return f"context_{index:06d}"


def sample_context_rows(
    rows: list[dict],
    fact_count: int,
    rng: random.Random,
    mode: str = "random",
    start_index: int = 0,
) -> list[dict]:
    if len(rows) < fact_count:
        raise ValueError(f"Need {fact_count} facts, found {len(rows)}")
    if mode == "random":
        return rng.sample(rows, fact_count)
    if mode == "head":
        return rows[:fact_count]
    if mode == "window":
        if not rows:
            return []
        start = start_index % len(rows)
        if start + fact_count <= len(rows):
            return rows[start : start + fact_count]
        return rows[start:] + rows[: fact_count - (len(rows) - start)]
    raise ValueError(f"Unknown context sampling mode: {mode}")


def resolve_context_format(context_format: str, index: int) -> str:
    if context_format == "mixed":
        return "structured" if index % 2 else "natural"
    return context_format


def make_context_payload(
    context_id: str,
    rows: list[dict],
    context_format: str,
    source_path: str | Path,
    extra: dict | None = None,
) -> dict:
    payload = {
        "context_id": context_id,
        "fact_ids": [str(row["id"]) for row in rows],
        "num_facts": len(rows),
        "context_format": context_format,
        "context": build_context(rows, context_format=context_format),
        "source_path": str(resolve_path(source_path)),
        "facts": rows,
    }
    if extra:
        payload.update(extra)
    return payload


def save_teacher_lora_entry(bank_dir: str | Path, context_id: str, lora_dict: Any, meta: dict) -> dict:
    entry_dir = resolve_path(bank_dir) / context_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    lora_path = entry_dir / "teacher_lora.pt"
    meta_path = entry_dir / "meta.json"
    torch.save(move_lora_to_cpu(lora_dict), lora_path)
    meta = dict(meta)
    meta["lora_path"] = str(lora_path)
    meta["meta_path"] = str(meta_path)
    write_json(meta_path, meta)
    return meta


def load_teacher_lora_entry(entry: dict | str | Path, device: torch.device, dtype: torch.dtype | None = None):
    if isinstance(entry, dict):
        lora_path = entry["lora_path"]
    else:
        path = resolve_path(entry)
        lora_path = path / "teacher_lora.pt" if path.is_dir() else path
    loaded = torch.load(lora_path, map_location="cpu", weights_only=False)
    return move_lora_to_device(loaded, device, dtype=dtype)


def move_lora_to_device(obj: Any, device: torch.device, dtype: torch.dtype | None = None):
    if torch.is_tensor(obj):
        tensor = obj.to(device)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor
    if isinstance(obj, dict):
        return {key: move_lora_to_device(value, device, dtype=dtype) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_lora_to_device(value, device, dtype=dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_lora_to_device(value, device, dtype=dtype) for value in obj)
    return obj


def load_bank_index(bank_dir: str | Path) -> list[dict]:
    index_path = resolve_path(bank_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Teacher LoRA bank index not found: {index_path}")
    payload = read_json(index_path)
    if isinstance(payload, dict) and "entries" in payload:
        return list(payload["entries"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Invalid teacher LoRA index format: {index_path}")


def write_bank_index(bank_dir: str | Path, payload: dict) -> None:
    write_json(resolve_path(bank_dir) / "index.json", payload)


def is_lora_leaf(obj: Any) -> bool:
    return isinstance(obj, dict) and "A" in obj and "B" in obj and torch.is_tensor(obj["A"]) and torch.is_tensor(obj["B"])


def iter_lora_leaf_pairs(student: Any, teacher: Any, prefix: str = ""):
    if is_lora_leaf(student):
        if not is_lora_leaf(teacher):
            raise ValueError(f"Teacher LoRA leaf missing at {prefix or '<root>'}")
        yield prefix or "<root>", student, teacher
        return
    if isinstance(student, dict):
        if not isinstance(teacher, dict):
            raise ValueError(f"Teacher LoRA branch missing at {prefix or '<root>'}")
        for key, student_value in student.items():
            if key not in teacher:
                raise ValueError(f"Teacher LoRA key missing at {prefix}.{key}" if prefix else f"Teacher LoRA key missing: {key}")
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_lora_leaf_pairs(student_value, teacher[key], next_prefix)
        return
    raise ValueError(f"Unexpected LoRA object at {prefix or '<root>'}: {type(student).__name__}")


def low_rank_frobenius_sq(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = A.float()
    B = B.float()
    a_gram = torch.matmul(A.transpose(-2, -1), A)
    b_gram = torch.matmul(B, B.transpose(-2, -1))
    return torch.einsum("...ij,...ji->...", a_gram, b_gram).clamp_min(0.0)


def low_rank_cross_frobenius(A_left: torch.Tensor, B_left: torch.Tensor, A_right: torch.Tensor, B_right: torch.Tensor) -> torch.Tensor:
    A_left = A_left.float()
    B_left = B_left.float()
    A_right = A_right.float()
    B_right = B_right.float()
    a_cross = torch.matmul(A_left.transpose(-2, -1), A_right)
    b_cross = torch.matmul(B_right, B_left.transpose(-2, -1))
    return torch.einsum("...ij,...ji->...", a_cross, b_cross)


def lora_leaf_delta_alignment_loss(
    student_leaf: dict,
    teacher_leaf: dict,
    eps: float = 1e-8,
    include_bias: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    A_s = student_leaf["A"]
    B_s = student_leaf["B"]
    A_t = teacher_leaf["A"].to(device=A_s.device, dtype=A_s.dtype)
    B_t = teacher_leaf["B"].to(device=B_s.device, dtype=B_s.dtype)
    student_norm_sq = low_rank_frobenius_sq(A_s, B_s)
    teacher_norm_sq = low_rank_frobenius_sq(A_t, B_t).detach()
    cross = low_rank_cross_frobenius(A_s, B_s, A_t, B_t)
    diff_sq = (student_norm_sq + teacher_norm_sq - 2.0 * cross).clamp_min(0.0)
    direction_loss = (diff_sq / (teacher_norm_sq + eps)).mean()

    bias_loss = direction_loss.new_tensor(0.0)
    if include_bias and student_leaf.get("C") is not None and teacher_leaf.get("C") is not None:
        C_s = student_leaf["C"].float()
        C_t = teacher_leaf["C"].to(device=student_leaf["C"].device, dtype=student_leaf["C"].dtype).float()
        bias_loss = ((C_s - C_t).pow(2).sum(dim=-1) / (C_t.pow(2).sum(dim=-1).detach() + eps)).mean()

    loss = direction_loss + bias_loss
    stats = {
        "direction_loss": float(direction_loss.detach().cpu()),
        "bias_loss": float(bias_loss.detach().cpu()),
        "student_norm": float(student_norm_sq.detach().mean().sqrt().cpu()),
        "teacher_norm": float(teacher_norm_sq.detach().mean().sqrt().cpu()),
    }
    return loss, stats


def lora_delta_alignment_loss(
    student_lora: Any,
    teacher_lora: Any,
    eps: float = 1e-8,
    include_bias: bool = True,
) -> tuple[torch.Tensor, dict]:
    losses = []
    module_stats = {}
    for path, student_leaf, teacher_leaf in iter_lora_leaf_pairs(student_lora, teacher_lora):
        loss, stats = lora_leaf_delta_alignment_loss(student_leaf, teacher_leaf, eps=eps, include_bias=include_bias)
        losses.append(loss)
        module_stats[path] = stats
    if not losses:
        raise ValueError("No LoRA leaves found for teacher alignment.")
    total = torch.stack(losses).mean()
    summary = {
        "num_modules": len(losses),
        "mean_module_loss": float(total.detach().cpu()),
        "modules": module_stats,
    }
    return total, summary
