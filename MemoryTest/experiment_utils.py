import json
from pathlib import Path

import compare_update_capacity as cap


def bootstrap_runtime(runtime_config_path: str):
    import torch
    from case_test import (
        answer_question,
        build_cfg,
        generate_context_lora,
        load_runtime,
        move_lora_to_cpu,
        resolve_device,
    )

    cap.torch = torch
    cap.answer_question = answer_question
    cap.generate_context_lora = generate_context_lora
    cap.move_lora_to_cpu = move_lora_to_cpu

    runtime_args = cap.make_runtime_args(cap.load_yaml_defaults(runtime_config_path))
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    metanetwork, metalora, tokenizer = load_runtime(cfg, runtime_args.checkpoint_dir, device)
    return runtime_args, device, cfg, metanetwork, metalora, tokenizer


def read_json_rows(path: str, min_count: int):
    resolved = cap.resolve_path(path)
    rows = json.loads(resolved.read_text(encoding="utf-8"))
    if len(rows) < min_count:
        raise ValueError(f"Need at least {min_count} rows, found {len(rows)} in {resolved}")
    return rows, resolved


def chunk_rows(rows, chunk_size: int, num_chunks: int):
    total = chunk_size * num_chunks
    return [rows[i:i + chunk_size] for i in range(0, total, chunk_size)]


def interleave_rows(left, right):
    mixed = []
    max_len = max(len(left), len(right))
    for idx in range(max_len):
        if idx < len(left):
            mixed.append(left[idx])
        if idx < len(right):
            mixed.append(right[idx])
    return mixed


def summarize_result(result):
    return {
        "correct": result["correct"],
        "total": result["total"],
        "accuracy": result["accuracy"],
    }


def write_payload(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
