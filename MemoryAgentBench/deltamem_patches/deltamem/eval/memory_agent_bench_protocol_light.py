"""MemoryAgentBench eval helpers without deltamem.core (for doc-to-lora / transformers 4.51 env)."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import sys
import time
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from datasets import DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from deltamem.eval.official_memory_agent_bench_templates import get_template

MEMORY_CONTEXT_QA_PROMPT_TEMPLATE = (
    "Use only the memory context below to answer the question.\n"
    "Reply with a short entity, phrase, number, or sentence only.\n"
    "If the answer is not supported by the context, reply exactly: I don't know.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

OFFICIAL_SOURCE_CONFIGS: dict[str, dict[str, int]] = {
    "eventqa_131072": {"context_max_length": 131072, "generation_max_length": 40, "max_test_samples": 5, "chunk_size": 4096},
    "eventqa_65536": {"context_max_length": 65536, "generation_max_length": 40, "max_test_samples": 5, "chunk_size": 4096},
    "eventqa_full": {"context_max_length": 800000, "generation_max_length": 40, "max_test_samples": 5, "chunk_size": 4096},
    "longmemeval_s_-1_500": {"context_max_length": 150000, "generation_max_length": 50, "max_test_samples": 500, "chunk_size": 4096},
    "longmemeval_s*": {"context_max_length": 400000, "generation_max_length": 50, "max_test_samples": 5, "chunk_size": 4096},
    "ruler_qa1_197K": {"context_max_length": 220000, "generation_max_length": 50, "max_test_samples": 100, "chunk_size": 4096},
    "ruler_qa2_421K": {"context_max_length": 524288, "generation_max_length": 50, "max_test_samples": 100, "chunk_size": 4096},
    "factconsolidation_mh_262k": {"context_max_length": 300000, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "factconsolidation_sh_262k": {"context_max_length": 300000, "generation_max_length": 10, "max_test_samples": 1, "chunk_size": 4096},
    "infbench_sum_eng_shots2": {"context_max_length": 220000, "generation_max_length": 1200, "max_test_samples": 1, "chunk_size": 4096},
    "detective_qa": {"context_max_length": 220000, "generation_max_length": 50, "max_test_samples": 1, "chunk_size": 4096},
    "icl_banking77_5900shot_balance": {"context_max_length": 220000, "generation_max_length": 20, "max_test_samples": 1, "chunk_size": 4096},
    "icl_clinic150_5900shot_balance": {"context_max_length": 220000, "generation_max_length": 20, "max_test_samples": 1, "chunk_size": 4096},
    "icl_nlu_5900shot_balance": {"context_max_length": 220000, "generation_max_length": 20, "max_test_samples": 1, "chunk_size": 4096},
    "icl_trec_coarse_5900shot_balance": {"context_max_length": 220000, "generation_max_length": 20, "max_test_samples": 1, "chunk_size": 4096},
    "icl_trec_fine_5900shot_balance": {"context_max_length": 220000, "generation_max_length": 20, "max_test_samples": 1, "chunk_size": 4096},
    "recsys_redial_full": {"context_max_length": 1480000, "generation_max_length": 300, "max_test_samples": 1, "chunk_size": 4096},
}

MEMORY_AGENT_BENCH_SPLIT_LABELS = {
    "Accurate_Retrieval": "Accurate Retrieval",
    "Test_Time_Learning": "Test-time Learning",
    "Long_Range_Understanding": "Long Range Understanding",
    "Conflict_Resolution": "Selective Forgetting",
    "Selective Forgetting": "Selective Forgetting",
}

MEMORY_AGENT_BENCH_CATEGORY_ORDER = [
    "Accurate Retrieval",
    "Test-time Learning",
    "Long Range Understanding",
    "Selective Forgetting",
]

FALLBACK_HF_HUB_CACHE_DIRS = (
    Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))),
    Path.home() / ".cache" / "huggingface" / "hub",
)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: str


@dataclass(frozen=True)
class MemoryAgentBenchRowTask:
    row_index: int
    item: dict
    question_count: int
    estimated_cost: int
    question_start: int
    question_end: int


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed(device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    resolved_device = device
    if enabled and device.startswith("cuda"):
        resolved_device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    if enabled and not dist.is_initialized():
        backend = "nccl" if resolved_device.startswith("cuda") else "gloo"
        timeout_minutes = int(os.environ.get("DIST_TIMEOUT_MINUTES", "180"))
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    return DistributedContext(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=resolved_device,
    )


def finalize_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def gather_indexed_records(
    indexed_records: list[tuple[int, dict[str, object]]],
    context: DistributedContext,
) -> list[dict[str, object]] | None:
    if not context.enabled:
        return [record for _, record in indexed_records]
    gathered: list[list[tuple[int, dict[str, object]]] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, indexed_records)
    if context.rank != 0:
        return None
    merged = [item for rank_records in gathered if rank_records is not None for item in rank_records]
    merged.sort(key=lambda item: item[0])
    return [record for _, record in merged]


def dataset_download_config(*, local_files_only: bool) -> DownloadConfig | None:
    if not local_files_only:
        return None
    return DownloadConfig(local_files_only=True)


def load_dataset_cached(*args, cache_dir: Path, local_files_only: bool, **kwargs):
    return load_dataset(
        *args,
        cache_dir=str(cache_dir),
        download_config=dataset_download_config(local_files_only=local_files_only),
        **kwargs,
    )


def candidate_hub_cache_dirs(primary_cache_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for cache_dir in (primary_cache_dir, *FALLBACK_HF_HUB_CACHE_DIRS):
        resolved = Path(cache_dir)
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)
    return candidates


def resolve_hub_file(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    hub_cache_dir: Path,
    local_files_only: bool,
) -> Path:
    repo_prefix = "datasets--" if repo_type == "dataset" else "models--"
    repo_dir_name = repo_prefix + repo_id.replace("/", "--")
    for cache_root in candidate_hub_cache_dirs(hub_cache_dir):
        snapshot_root = cache_root / repo_dir_name / "snapshots"
        if not snapshot_root.is_dir():
            continue
        for snapshot_dir in sorted(snapshot_root.iterdir(), key=lambda path: path.name, reverse=True):
            candidate = snapshot_dir / filename
            if candidate.is_file():
                return candidate
    if local_files_only:
        raise FileNotFoundError(f"Could not find cached {repo_type} file {repo_id}:{filename}")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            cache_dir=str(hub_cache_dir),
        )
    )


def load_external_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {file_path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_editdistance_shim() -> None:
    if "editdistance" in sys.modules:
        return

    def _levenshtein_distance(left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                current.append(
                    min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (0 if left_char == right_char else 1))
                )
            previous = current
        return previous[-1]

    module = types.ModuleType("editdistance")
    setattr(module, "eval", _levenshtein_distance)
    sys.modules["editdistance"] = module


def load_mab_eval_utils(repo_root: Path):
    _install_editdistance_shim()
    return load_external_module(
        "official_memory_agent_bench_eval_utils",
        str(repo_root / "utils" / "eval_other_utils.py"),
    )


def build_context_chunks(rows: list[dict], *, chunk_size: int, eval_utils_module) -> list[list[str]]:
    return [
        eval_utils_module.chunk_text_into_sentences(str(row.get("context", "")), chunk_size=chunk_size)
        for row in rows
    ]


def build_memorized_context(source: str, chunks: list[str]) -> str:
    memorize_template = get_template(source, "memorize", "Long_context_agent_deltamem")
    memorized = ""
    for chunk in chunks:
        format_kwargs = {"context": chunk}
        if "{time_stamp}" in memorize_template:
            format_kwargs["time_stamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        memorized += "\n" + memorize_template.format(**format_kwargs)
    return memorized.strip()


def truncate_text_by_tokens(
    text: str,
    *,
    tokenizer,
    max_tokens: int,
    keep: Literal["head", "tail"],
) -> str:
    if max_tokens <= 0 or not text:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    kept = token_ids[:max_tokens] if keep == "head" else token_ids[-max_tokens:]
    return tokenizer.decode(kept, skip_special_tokens=True).strip()


def truncate_memory_context(
    memory_context: str,
    *,
    tokenizer,
    context_max_length: int,
    raw_input_length_limit: int,
    buffer_length: int,
    generation_max_length: int,
) -> str:
    if raw_input_length_limit <= 0:
        return memory_context
    effective_input_length_limit = max(1, raw_input_length_limit - buffer_length - generation_max_length)
    if effective_input_length_limit > context_max_length + buffer_length:
        return memory_context
    truncated = memory_context
    if context_max_length > 0:
        truncated = truncate_text_by_tokens(
            truncated,
            tokenizer=tokenizer,
            max_tokens=context_max_length,
            keep="tail",
        )
    return truncate_text_by_tokens(
        truncated,
        tokenizer=tokenizer,
        max_tokens=effective_input_length_limit,
        keep="tail",
    )


def metadata_value(metadata: dict, key: str, question_index: int, default=None):
    values = metadata.get(key)
    if isinstance(values, list) and question_index < len(values):
        return values[question_index]
    return default


def _template_label_value(answer: object) -> str:
    if isinstance(answer, list):
        if not answer:
            return ""
        return str(answer[0]).strip()
    return str(answer).strip()


def build_query_answer_pairs(row: dict, *, source: str) -> list[tuple[str, object, object]]:
    query_template = get_template(source, "query", "Long_context_agent_deltamem")
    questions = row.get("questions") or []
    answers = row.get("answers") or []
    metadata = row.get("metadata") or {}
    pairs: list[tuple[str, object, object]] = []
    for question_index, question in enumerate(questions):
        answer = answers[question_index] if question_index < len(answers) else ""
        qa_metadata = {
            "question": question,
            "answer": answer,
            "label": _template_label_value(answer),
            "source": metadata.get("source", ""),
            "question_dates": metadata_value(metadata, "question_dates", question_index),
            "question_types": metadata_value(metadata, "question_types", question_index),
            "question_ids": metadata_value(metadata, "question_ids", question_index),
            "previous_events": metadata_value(metadata, "previous_events", question_index),
            "qa_pair_ids": metadata_value(metadata, "qa_pair_ids", question_index),
        }
        formatted_query = query_template.format(**qa_metadata)
        pairs.append((formatted_query, answer, qa_metadata.get("qa_pair_ids")))
    return pairs


def memory_agent_bench_source(item: dict) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("source", "")).strip()


def memory_agent_bench_source_config(item: dict) -> dict[str, int]:
    source = memory_agent_bench_source(item)
    config = OFFICIAL_SOURCE_CONFIGS.get(source)
    return dict(config) if config else {}


def memory_agent_bench_primary_metric_name(source: str) -> str:
    return _memory_agent_bench_source_spec(source)["metric_key"]


def _memory_agent_bench_source_spec(source: str) -> dict[str, str]:
    source = str(source or "")
    if source == "infbench_sum_eng_shots2":
        metric_key = "f1"
        dataset_name = "InfinityBench-Sum"
    elif source.startswith("recsys_"):
        metric_key = "recsys_recall@5"
        dataset_name = "Movie Recommendation"
    else:
        metric_key = "accuracy"
        dataset_name = source or "unknown"
    return {"dataset_name": dataset_name, "metric_key": metric_key}


def clip_context_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[... context truncated ...]\n\n"
    if max_chars <= len(marker) + 32:
        return text[-max_chars:]
    head_chars = max(1, (max_chars - len(marker)) // 3)
    tail_chars = max(1, max_chars - len(marker) - head_chars)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def extract_first_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text.strip()


def normalize_qa_span(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def hotpotqa_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_qa_span(prediction).split()
    gold_tokens = normalize_qa_span(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def memory_qa_is_correct(prediction: str, aliases: list[str]) -> bool:
    normalized_prediction = normalize_qa_span(prediction)
    normalized_first_line = normalize_qa_span(extract_first_line(prediction))
    if not normalized_prediction and not normalized_first_line:
        return False
    for alias in aliases:
        normalized_alias = normalize_qa_span(alias)
        if not normalized_alias:
            continue
        if normalized_alias == normalized_first_line or normalized_alias == normalized_prediction:
            return True
        if normalized_alias in normalized_prediction:
            return True
    return False


def qa_alias_max_f1(prediction: str, aliases: list[str]) -> float:
    candidates = [prediction, extract_first_line(prediction)]
    best = 0.0
    for candidate in candidates:
        if not str(candidate).strip():
            continue
        for alias in aliases:
            if not str(alias).strip():
                continue
            best = max(best, hotpotqa_f1(str(candidate), str(alias)))
    return best


def infer_model_context_window(model, tokenizer) -> int:
    direct_max_model_len = getattr(model, "max_model_len", None)
    if isinstance(direct_max_model_len, int) and 0 < direct_max_model_len < 10**7:
        return direct_max_model_len
    config = getattr(model, "config", None)
    for candidate in (model, config, getattr(model, "text_config", None), getattr(config, "text_config", None)):
        if candidate is None:
            continue
        for attr in ("max_position_embeddings", "sliding_window"):
            value = getattr(candidate, attr, None)
            if isinstance(value, int) and 0 < value < 10**7:
                return value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 10**7:
        return value
    return 32768


def prompt_token_count(tokenizer, prompt: str, *, use_chat_template: bool) -> int:
    if use_chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return int(input_ids.shape[-1])


def render_clipped_context_from_tokens(
    original_text: str,
    context_token_ids: list[int],
    *,
    tokenizer,
    keep_tokens: int,
    marker: str,
    marker_token_count: int,
) -> str:
    if keep_tokens <= 0:
        return ""
    if len(context_token_ids) <= keep_tokens:
        return original_text
    if keep_tokens <= marker_token_count + 2:
        return tokenizer.decode(context_token_ids[-keep_tokens:], skip_special_tokens=True).strip()
    head_tokens = max(1, (keep_tokens - marker_token_count) // 3)
    tail_tokens = max(1, keep_tokens - marker_token_count - head_tokens)
    head_text = tokenizer.decode(context_token_ids[:head_tokens], skip_special_tokens=True).rstrip()
    tail_text = tokenizer.decode(context_token_ids[-tail_tokens:], skip_special_tokens=True).lstrip()
    return head_text + marker + tail_text


def truncate_context_text_fast(
    context_text: str,
    *,
    tokenizer,
    max_prompt_tokens: int,
    prompt_overhead_tokens: int,
    max_context_chars: int = 0,
    keep: str = "head_tail",
) -> str:
    if max_context_chars > 0 and len(context_text) > max_context_chars:
        context_text = clip_context_text(context_text, max_context_chars) if keep != "tail" else context_text[-max_context_chars:]
    if not context_text:
        return context_text
    available_context_tokens = max(1, max_prompt_tokens - prompt_overhead_tokens - 32)
    context_token_ids = tokenizer.encode(context_text, add_special_tokens=False)
    if len(context_token_ids) <= available_context_tokens:
        return context_text
    if keep == "tail":
        return tokenizer.decode(context_token_ids[-available_context_tokens:], skip_special_tokens=True).strip()
    marker = "\n\n[... context truncated ...]\n\n"
    marker_token_count = len(tokenizer.encode(marker, add_special_tokens=False))
    return render_clipped_context_from_tokens(
        context_text,
        context_token_ids,
        tokenizer=tokenizer,
        keep_tokens=available_context_tokens,
        marker=marker,
        marker_token_count=marker_token_count,
    )


def clip_context_text_to_model_limit(
    context_text: str,
    *,
    tokenizer,
    prompt_builder,
    use_chat_template: bool,
    model_context_window: int,
    max_new_tokens: int,
    max_context_chars: int = 0,
    keep: str = "head_tail",
) -> str:
    if not context_text:
        return context_text
    max_prompt_tokens = max(1, model_context_window - max_new_tokens)
    prompt_overhead_tokens = prompt_token_count(tokenizer, prompt_builder(""), use_chat_template=use_chat_template)
    clipped_context = truncate_context_text_fast(
        context_text,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        prompt_overhead_tokens=prompt_overhead_tokens,
        max_context_chars=max_context_chars,
        keep=keep,
    )
    prompt_tokens = prompt_token_count(tokenizer, prompt_builder(clipped_context), use_chat_template=use_chat_template)
    if prompt_tokens <= max_prompt_tokens:
        return clipped_context
    overflow = prompt_tokens - max_prompt_tokens + 32
    context_token_ids = tokenizer.encode(clipped_context, add_special_tokens=False)
    keep_tokens = max(1, len(context_token_ids) - overflow)
    if keep == "tail":
        return tokenizer.decode(context_token_ids[-keep_tokens:], skip_special_tokens=True).strip()
    marker = "\n\n[... context truncated ...]\n\n"
    marker_token_count = len(tokenizer.encode(marker, add_special_tokens=False))
    return render_clipped_context_from_tokens(
        clipped_context,
        context_token_ids,
        tokenizer=tokenizer,
        keep_tokens=keep_tokens,
        marker=marker,
        marker_token_count=marker_token_count,
    )


def _render_message_prompt(tokenizer, messages: list[dict[str, str]], *, use_chat_template: bool) -> str:
    if use_chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def build_deltamem_unified_query_prompt(
    *,
    tokenizer,
    model_context_window: int,
    raw_context: str,
    question: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    question_text = str(question).strip()

    def prompt_builder(candidate_context: str, question_text=question_text) -> str:
        return MEMORY_CONTEXT_QA_PROMPT_TEMPLATE.format(context=candidate_context, question=question_text)

    final_context = clip_context_text_to_model_limit(
        raw_context,
        tokenizer=tokenizer,
        prompt_builder=prompt_builder,
        use_chat_template=False,
        model_context_window=model_context_window,
        max_new_tokens=max_new_tokens,
        max_context_chars=0,
        keep="head_tail",
    )
    return prompt_builder(final_context), final_context


def build_deltamem_official_query_prompt(
    *,
    tokenizer,
    model_context_window: int,
    raw_context: str,
    source: str,
    query: str,
    eval_utils,
    source_config: dict[str, Any],
    max_new_tokens: int,
    official_buffer_length: int = 4000,
) -> tuple[str, str]:
    chunk_size = int(source_config.get("chunk_size") or 4096)
    context_max_length = int(source_config.get("context_max_length") or 0)
    context_chunks = build_context_chunks([{"context": raw_context}], chunk_size=chunk_size, eval_utils_module=eval_utils)
    memorized_context = build_memorized_context(source, context_chunks[0])
    truncated_context = truncate_memory_context(
        memorized_context,
        tokenizer=tokenizer,
        context_max_length=context_max_length,
        raw_input_length_limit=model_context_window,
        buffer_length=official_buffer_length,
        generation_max_length=max_new_tokens,
    )
    system_message = get_template(source, "system", "Long_context_agent_deltamem")

    def prompt_builder(candidate_context: str, query=query) -> str:
        return _render_message_prompt(
            tokenizer,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{candidate_context}\n{query}".strip()},
            ],
            use_chat_template=False,
        )

    final_context = clip_context_text_to_model_limit(
        truncated_context,
        tokenizer=tokenizer,
        prompt_builder=prompt_builder,
        use_chat_template=False,
        model_context_window=model_context_window,
        max_new_tokens=max_new_tokens,
        max_context_chars=0,
        keep="tail",
    )
    return prompt_builder(final_context), final_context


def normalize_answer_aliases(values: list[object] | object) -> list[str]:
    raw_values = values if isinstance(values, list) else [values]
    aliases: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value).strip()
        if not text:
            continue
        normalized = normalize_qa_span(text)
        if not normalized or normalized in seen:
            continue
        aliases.append(text)
        seen.add(normalized)
    return aliases


def _metadata_list_value(metadata: dict, key: str, index: int, default=None):
    values = metadata.get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return default


def load_memory_agent_bench(
    *,
    cache_dir: Path,
    hub_cache_dir: Path,
    splits: list[str],
    sources: list[str] | None,
    max_samples: int | None,
    seed: int,
    local_files_only: bool,
) -> list[dict]:
    all_rows: list[dict] = []
    flat_refs: list[tuple[int, int]] = []
    selected_sources = None if not sources else set(sources)
    for split in splits:
        parquet_path = resolve_hub_file(
            repo_id="ai-hyz/MemoryAgentBench",
            repo_type="dataset",
            filename=f"data/{split}-00000-of-00001.parquet",
            hub_cache_dir=hub_cache_dir,
            local_files_only=local_files_only,
        )
        dataset = load_dataset_cached(
            "parquet",
            data_files=str(parquet_path),
            split="train",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        for row_idx, row in enumerate(dataset):
            materialized = dict(row)
            metadata = dict(materialized.get("metadata") or {})
            source = str(metadata.get("source", "")).strip()
            if selected_sources is not None and source not in selected_sources:
                continue
            questions = [str(value).strip() for value in materialized.get("questions", []) or []]
            answers = list(materialized.get("answers", []) or [])
            all_rows.append(
                {
                    "split": split,
                    "row_id": f"{split}:{row_idx}",
                    "context": str(materialized.get("context", "")),
                    "questions": questions,
                    "answers": answers,
                    "metadata": metadata,
                    "source": source,
                    "official_source_config": memory_agent_bench_source_config({"metadata": metadata}),
                }
            )
            current_row_index = len(all_rows) - 1
            for question_idx in range(len(questions)):
                flat_refs.append((current_row_index, question_idx))

    if max_samples is not None and len(flat_refs) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(flat_refs)
        flat_refs = flat_refs[:max_samples]

    row_to_selected: dict[int, list[int]] = defaultdict(list)
    for eval_index, (row_index, question_index) in enumerate(flat_refs):
        row_to_selected[row_index].append(question_index)

    items: list[dict] = []
    running_index = 0
    for row_index, row in enumerate(all_rows):
        selected_question_indices = row_to_selected.get(row_index)
        if not selected_question_indices:
            continue
        selected_question_indices = sorted(selected_question_indices)
        selected_questions: list[dict] = []
        for question_index in selected_question_indices:
            metadata = row["metadata"]
            selected_questions.append(
                {
                    "eval_index": running_index,
                    "question_index": question_index,
                    "question": row["questions"][question_index],
                    "answer_raw": row["answers"][question_index] if question_index < len(row["answers"]) else [],
                    "answer_aliases": normalize_answer_aliases(
                        row["answers"][question_index] if question_index < len(row["answers"]) else []
                    ),
                    "question_id": _metadata_list_value(metadata, "question_ids", question_index),
                    "question_type": _metadata_list_value(metadata, "question_types", question_index),
                    "question_date": _metadata_list_value(metadata, "question_dates", question_index),
                    "previous_event": _metadata_list_value(metadata, "previous_events", question_index),
                    "qa_pair_id": _metadata_list_value(metadata, "qa_pair_ids", question_index),
                }
            )
            running_index += 1
        items.append({**row, "selected_questions": selected_questions})
    return items


def _memory_agent_bench_row_estimated_cost(
    item: dict,
    *,
    default_max_new_tokens: int,
    use_official_generation_lengths: bool,
) -> int:
    question_count = len(item.get("selected_questions", []))
    context_chars = len(str(item.get("context", "")))
    context_token_estimate = max(1, context_chars // 4)
    if use_official_generation_lengths:
        gen_len = int(memory_agent_bench_source_config(item).get("generation_max_length") or default_max_new_tokens)
    else:
        gen_len = default_max_new_tokens
    return context_token_estimate + question_count * gen_len


def local_memory_agent_bench_row_tasks(
    items: list[dict],
    context: DistributedContext,
    *,
    default_max_new_tokens: int,
    use_official_generation_lengths: bool,
    max_questions_per_row_task: int = 0,
) -> list[MemoryAgentBenchRowTask]:
    candidates: list[MemoryAgentBenchRowTask] = []
    for row_index, item in enumerate(items):
        selected_questions = list(item.get("selected_questions", []))
        if not selected_questions:
            continue
        row_stride = max(1, int(max_questions_per_row_task)) if int(max_questions_per_row_task) > 0 else len(selected_questions)
        for question_start in range(0, len(selected_questions), row_stride):
            question_end = min(len(selected_questions), question_start + row_stride)
            task_item = dict(item)
            task_item["selected_questions"] = selected_questions[question_start:question_end]
            candidates.append(
                MemoryAgentBenchRowTask(
                    row_index=row_index,
                    item=task_item,
                    question_count=question_end - question_start,
                    estimated_cost=_memory_agent_bench_row_estimated_cost(
                        task_item,
                        default_max_new_tokens=default_max_new_tokens,
                        use_official_generation_lengths=use_official_generation_lengths,
                    ),
                    question_start=question_start,
                    question_end=question_end,
                )
            )
    if not context.enabled:
        return candidates
    rank_loads = [0] * context.world_size
    task_to_rank: dict[tuple[int, int, int], int] = {}
    for row_task in sorted(
        candidates,
        key=lambda task: (-task.estimated_cost, -task.question_count, str(task.item.get("row_id", "")), task.row_index, task.question_start),
    ):
        target_rank = min(range(context.world_size), key=lambda rank: (rank_loads[rank], rank))
        task_key = (row_task.row_index, row_task.question_start, row_task.question_end)
        task_to_rank[task_key] = target_rank
        rank_loads[target_rank] += row_task.estimated_cost
    return [
        row_task
        for row_task in candidates
        if task_to_rank[(row_task.row_index, row_task.question_start, row_task.question_end)] == context.rank
    ]


def _memory_agent_bench_record_score(record: dict[str, object]) -> tuple[dict[str, str], float]:
    source = str(record.get("source") or "")
    spec = _memory_agent_bench_source_spec(source)
    metric_key = spec["metric_key"]
    if metric_key == "accuracy":
        return spec, float(bool(record.get("correct")))
    if metric_key == "f1":
        f1_value = record.get("f1")
        if isinstance(f1_value, (int, float)):
            return spec, float(f1_value)
    stored_value = record.get(metric_key)
    if isinstance(stored_value, (int, float)):
        return spec, float(stored_value)
    return spec, 0.0


def summarize_memory_agent_bench(records: list[dict[str, object]]) -> dict[str, object]:
    categories: defaultdict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        category_name = MEMORY_AGENT_BENCH_SPLIT_LABELS.get(str(record.get("split") or ""), str(record.get("split") or "unknown"))
        dataset_name = _memory_agent_bench_source_spec(str(record.get("source") or ""))["dataset_name"]
        categories[category_name][dataset_name].append(record)

    category_payload: dict[str, dict[str, object]] = {}
    category_overall: dict[str, float] = {}
    total_weighted_score = 0.0
    total_num_samples = 0
    dataset_scores: dict[str, dict[str, object]] = {}

    for category_name in MEMORY_AGENT_BENCH_CATEGORY_ORDER:
        datasets = categories.get(category_name)
        if not datasets:
            continue
        dataset_payload: dict[str, dict[str, object]] = {}
        category_weighted_sum = 0.0
        category_num_samples = 0
        for dataset_name, dataset_records in sorted(datasets.items(), key=lambda item: item[0]):
            scores = [_memory_agent_bench_record_score(record)[1] for record in dataset_records]
            dataset_score = 0.0 if not scores else round(sum(scores) / len(scores), 4)
            dataset_summary = {
                "score": dataset_score,
                "num_samples": len(dataset_records),
                "metric_key": _memory_agent_bench_record_score(dataset_records[0])[0]["metric_key"],
            }
            dataset_payload[dataset_name] = dataset_summary
            dataset_scores[dataset_name] = {"category": category_name, **dataset_summary}
            category_weighted_sum += sum(scores)
            category_num_samples += len(dataset_records)
        overall = 0.0 if category_num_samples == 0 else round(category_weighted_sum / category_num_samples, 4)
        category_payload[category_name] = {
            "overall": overall,
            "num_samples": category_num_samples,
            "datasets": dataset_payload,
        }
        category_overall[category_name] = overall
        total_weighted_score += category_weighted_sum
        total_num_samples += category_num_samples

    overall = 0.0 if total_num_samples == 0 else round(total_weighted_score / total_num_samples, 4)
    return {
        "overall": overall,
        "primary_metric": "sample_weighted_category_overall",
        "primary_score": overall,
        "num_samples": len(records),
        "dataset_scores": dataset_scores,
        "category_overall": category_overall,
        "categories": category_payload,
    }


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
