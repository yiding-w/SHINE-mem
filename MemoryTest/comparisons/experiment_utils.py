import gc
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("experiment_utils")
MEMORY_TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMORY_TEST_ROOT.parent
DEFAULT_RUNTIME_CONFIG_PATH = MEMORY_TEST_ROOT / "config" / "case_test.yaml"

if str(MEMORY_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMORY_TEST_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = None
answer_question = None
generate_context_lora = None
move_lora_to_cpu = None
extract_think_and_answer = None


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (REPO_ROOT / resolved).resolve()


def load_yaml_defaults(path: str) -> dict:
    from omegaconf import OmegaConf

    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Runtime config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config must contain a YAML mapping: {cfg_path}")
    return data


def make_runtime_args(defaults: dict) -> Namespace:
    return Namespace(
        model_path=defaults["model_path"],
        checkpoint_dir=defaults["checkpoint_dir"],
        checkpoint_profile=str(defaults.get("checkpoint_profile", "auto")),
        device=defaults.get("device", "cuda"),
        gpu_id=int(defaults.get("gpu_id", 0)),
        seed=int(defaults.get("seed", 42)),
        context_max_length=int(defaults.get("context_max_length", 1550)),
        conversation_max_length=int(defaults.get("conversation_max_length", 5000)),
        max_new_tokens=int(defaults.get("max_new_tokens", 128)),
        lora_r=int(defaults.get("lora_r", 8)),
        metalora_r=int(defaults.get("metalora_r", 128)),
        metanetwork_layers=int(defaults.get("metanetwork_layers", 4)),
        use_system_prompt=bool(defaults.get("use_system_prompt", False)),
        enable_thinking=bool(defaults.get("enable_thinking", False)),
    )


def bootstrap_runtime(runtime_config_path: str):
    global torch, answer_question, generate_context_lora, move_lora_to_cpu, extract_think_and_answer

    import torch as torch_module
    from case_test import (
        answer_question as answer_question_func,
        build_cfg,
        extract_think_and_answer as extract_think_and_answer_func,
        generate_context_lora as generate_context_lora_func,
        load_runtime,
        move_lora_to_cpu as move_lora_to_cpu_func,
        resolve_device,
    )

    torch = torch_module
    answer_question = answer_question_func
    generate_context_lora = generate_context_lora_func
    move_lora_to_cpu = move_lora_to_cpu_func
    extract_think_and_answer = extract_think_and_answer_func

    runtime_args = make_runtime_args(load_yaml_defaults(runtime_config_path))
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    metanetwork, metalora, tokenizer = load_runtime(
        cfg,
        runtime_args.checkpoint_dir,
        device,
        checkpoint_profile=runtime_args.checkpoint_profile,
    )
    return runtime_args, device, cfg, metanetwork, metalora, tokenizer


def read_json_rows(path: str, min_count: int):
    resolved = resolve_path(path)
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


def weighted_average_lora(old_lora: Any, new_lora: Any, new_count: int):
    if old_lora is None or new_lora is None:
        if old_lora is not None or new_lora is not None:
            raise ValueError("LoRA structures do not match: one side is None and the other is not.")
        return None
    old_weight = float(new_count - 1) / float(new_count)
    new_weight = 1.0 / float(new_count)
    if torch.is_tensor(old_lora):
        return old_lora * old_weight + new_lora * new_weight
    if isinstance(old_lora, dict):
        return {key: weighted_average_lora(old_lora[key], new_lora[key], new_count) for key in old_lora}
    if isinstance(old_lora, list):
        return [weighted_average_lora(old_item, new_item, new_count) for old_item, new_item in zip(old_lora, new_lora)]
    if isinstance(old_lora, tuple):
        return tuple(weighted_average_lora(old_item, new_item, new_count) for old_item, new_item in zip(old_lora, new_lora))
    raise TypeError(f"Unsupported LoRA value type for averaging: {type(old_lora)!r}")


def sum_lora(old_lora: Any, new_lora: Any):
    if old_lora is None or new_lora is None:
        if old_lora is not None or new_lora is not None:
            raise ValueError("LoRA structures do not match: one side is None and the other is not.")
        return None
    if torch.is_tensor(old_lora):
        return old_lora + new_lora
    if isinstance(old_lora, dict):
        return {key: sum_lora(old_lora[key], new_lora[key]) for key in old_lora}
    if isinstance(old_lora, list):
        return [sum_lora(old_item, new_item) for old_item, new_item in zip(old_lora, new_lora)]
    if isinstance(old_lora, tuple):
        return tuple(sum_lora(old_item, new_item) for old_item, new_item in zip(old_lora, new_lora))
    raise TypeError(f"Unsupported LoRA value type for summation: {type(old_lora)!r}")


def concat_average_lora(old_lora: Any, new_lora: Any, new_count: int):
    if old_lora is None or new_lora is None:
        if old_lora is not None or new_lora is not None:
            raise ValueError("LoRA structures do not match: one side is None and the other is not.")
        return None
    if isinstance(old_lora, dict) and {"A", "B"}.issubset(old_lora.keys()):
        old_scale = ((new_count - 1.0) / new_count) ** 0.5
        new_scale = (1.0 / new_count) ** 0.5
        merged = {
            "A": torch.cat([old_lora["A"] * old_scale, new_lora["A"] * new_scale], dim=-1),
            "B": torch.cat([old_lora["B"] * old_scale, new_lora["B"] * new_scale], dim=-2),
        }
        old_c = old_lora.get("C")
        new_c = new_lora.get("C")
        if old_c is None or new_c is None:
            if old_c is not None or new_c is not None:
                raise ValueError("LoRA bias structures do not match: one C is None and the other is not.")
            merged["C"] = None
        else:
            merged["C"] = old_c * ((new_count - 1.0) / new_count) + new_c * (1.0 / new_count)
        return merged
    if isinstance(old_lora, dict):
        return {key: concat_average_lora(old_lora[key], new_lora[key], new_count) for key in old_lora}
    if isinstance(old_lora, list):
        return [concat_average_lora(old_item, new_item, new_count) for old_item, new_item in zip(old_lora, new_lora)]
    if isinstance(old_lora, tuple):
        return tuple(concat_average_lora(old_item, new_item, new_count) for old_item, new_item in zip(old_lora, new_lora))
    raise TypeError(f"Unsupported LoRA value type for concatenation: {type(old_lora)!r}")


def merge_lora(old_lora: Any, new_lora: Any, new_count: int, merge_method: str):
    if merge_method == "average":
        return weighted_average_lora(old_lora, new_lora, new_count)
    if merge_method == "sum":
        return sum_lora(old_lora, new_lora)
    if merge_method == "concat":
        return concat_average_lora(old_lora, new_lora, new_count)
    raise ValueError(f"Unknown merge method: {merge_method}")


def build_context(rows: list[dict]) -> str:
    return "\n".join(row["text"] for row in rows)


def answer_matches(expected_answer: str, model_answer: str) -> bool:
    expected = str(expected_answer).casefold()
    answer = str(model_answer).casefold()
    return expected in answer


def generate_merged_lora(
    fact_chunks,
    metanetwork,
    tokenizer,
    metalora,
    cfg,
    device,
    log_context: bool = False,
    condition_label: str = "A",
    merge_method: str = "average",
):
    merged_lora = None
    context_records = []
    LOGGER.info("%s: merge_method=%s", condition_label, merge_method)
    for update_idx, chunk in enumerate(fact_chunks, start=1):
        context = build_context(chunk)
        context_records.append(
            {
                "update_index": update_idx,
                "num_rows": len(chunk),
                "num_facts": len([row for row in chunk if "question" in row and "answer" in row]),
                "fact_ids": [row["id"] for row in chunk],
                "context": context,
            }
        )
        LOGGER.info("%s/update %s: generating LoRA from %s rows", condition_label, update_idx, len(chunk))
        if log_context:
            LOGGER.info("%s/update %s context:\n%s", condition_label, update_idx, context)
        new_lora = generate_context_lora(context, metanetwork, tokenizer, metalora, cfg, device)
        if merged_lora is None:
            merged_lora = new_lora
        else:
            LOGGER.info("%s/update %s: merging LoRA with method=%s", condition_label, update_idx, merge_method)
            merged_lora = merge_lora(merged_lora, new_lora, update_idx, merge_method)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return merged_lora, context_records



def evaluate_lora(label, facts, lora_dict, metanetwork, tokenizer, runtime_args, device):
    if lora_dict is None:
        LOGGER.info("Evaluating %s on %s questions without LoRA/context", label, len(facts))
    else:
        LOGGER.info("Evaluating %s on %s questions", label, len(facts))
    rows = []
    correct = 0
    for idx, fact in enumerate(facts, start=1):
        result = answer_question(
            question=fact["question"],
            metanetwork=metanetwork,
            tokenizer=tokenizer,
            lora_dict=lora_dict,
            device=device,
            max_new_tokens=runtime_args.max_new_tokens,
            max_conversation_length=runtime_args.conversation_max_length,
            use_system_prompt=runtime_args.use_system_prompt,
            enable_thinking=runtime_args.enable_thinking,
        )
        is_correct = answer_matches(fact["answer"], result["answer"])
        correct += int(is_correct)
        rows.append(
            {
                "index": idx,
                "id": fact["id"],
                "person": fact["person"],
                "question": fact["question"],
                "expected_answer": fact["answer"],
                "model_answer": result["answer"],
                "raw": result["raw"],
                "match_mode": "case_insensitive_answer_substring",
                "correct": is_correct,
            }
        )
    return {
        "label": label,
        "correct": correct,
        "total": len(facts),
        "accuracy": correct / len(facts) if facts else 0.0,
        "rows": rows,
    }


def answer_question_with_context(
    context: str,
    question: str,
    metanetwork,
    tokenizer,
    device,
    max_new_tokens: int,
    max_conversation_length: int,
    enable_thinking: bool,
):
    user_prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer the question directly."
    )
    input_enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=max_conversation_length,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=enable_thinking,
    )
    input_ids = input_enc["input_ids"].to(device)
    attention_mask = input_enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=None,
        )
    new_tokens = outputs[0, input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    think_text, answer_text = extract_think_and_answer(raw_text)
    return {
        "question": question,
        "think": think_text,
        "answer": answer_text,
        "raw": raw_text,
    }


def evaluate_in_context_baseline(label, facts, context: str, metanetwork, tokenizer, runtime_args, device):
    LOGGER.info("Evaluating %s on %s questions with prompt context and no LoRA", label, len(facts))
    rows = []
    correct = 0
    for idx, fact in enumerate(facts, start=1):
        result = answer_question_with_context(
            context=context,
            question=fact["question"],
            metanetwork=metanetwork,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=runtime_args.max_new_tokens,
            max_conversation_length=runtime_args.conversation_max_length,
            enable_thinking=runtime_args.enable_thinking,
        )
        is_correct = answer_matches(fact["answer"], result["answer"])
        correct += int(is_correct)
        rows.append(
            {
                "index": idx,
                "id": fact["id"],
                "person": fact["person"],
                "question": fact["question"],
                "expected_answer": fact["answer"],
                "model_answer": result["answer"],
                "raw": result["raw"],
                "match_mode": "case_insensitive_answer_substring",
                "correct": is_correct,
            }
        )
    return {
        "label": label,
        "correct": correct,
        "total": len(facts),
        "accuracy": correct / len(facts) if facts else 0.0,
        "rows": rows,
    }


def save_lora_snapshot(path: Path, lora_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(move_lora_to_cpu(lora_dict), path)
    LOGGER.info("Saved LoRA snapshot to %s", path)


def summarize_result(result):
    return {
        "correct": result["correct"],
        "total": result["total"],
        "accuracy": result["accuracy"],
    }


def write_payload(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
