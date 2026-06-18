"""SHINE agent evaluation on MemoryAgentBench (δ-mem protocol)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from deltamem.eval.official_memory_agent_bench import (
    build_context_chunks as build_official_mab_context_chunks,
    build_memorized_context as build_official_mab_memorized_context,
    build_query_answer_pairs as build_official_mab_query_answer_pairs,
    load_mab_eval_utils,
    truncate_memory_context as truncate_official_mab_memory_context,
)
from deltamem.eval.official_memory_agent_bench_templates import get_template as get_official_mab_template

if TYPE_CHECKING:
    from argparse import Namespace

    from deltamem.eval.benchmark_compare import MemoryAgentBenchRowTask
    from deltamem.eval.common import DistributedContext


def _shine_question_only_query(question: str) -> str:
    """Legacy SHINE query (question only; context encoded only in LoRA)."""
    return (
        "Use only the memory context below to answer the question.\n"
        "Reply with a short entity, phrase, number, or sentence only.\n"
        "If the answer is not supported by the context, reply exactly: I don't know.\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )


def _ensure_paths(*, shine_root: Path, mab_root: Path) -> None:
    for path in (shine_root, mab_root):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _load_agent_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_shine_root(agent_config: dict[str, Any], config_path: Path) -> Path:
    shine_root = agent_config.get("shine_root") or os.environ.get("SHINE_ROOT", "")
    if shine_root in ("..", ".", ""):
        return config_path.resolve().parents[2]
    return Path(shine_root).resolve()


def _configure_agent_for_suite(agent_config: dict[str, Any], args: Namespace) -> dict[str, Any]:
    configured = dict(agent_config)
    if args.memory_agent_bench_use_official_prompt:
        configured["temperature"] = 0.0
    else:
        configured["temperature"] = float(args.eval_temperature)
        configured["top_p"] = float(args.eval_top_p)
        configured["top_k"] = int(args.eval_top_k)
    configured["use_mab_generation_max_length"] = True
    # Default: same prompt as δ-mem base (context + question) + LoRA from context.
    configured.setdefault("query_include_context", True)
    env_cap = os.environ.get("SHINE_CONTEXT_MAX_LENGTH", "").strip()
    if env_cap.isdigit():
        configured["shine_context_max_length"] = int(env_cap)
    return configured


def _build_deltamem_unified_query_prompt(
    *,
    compare_mod,
    tokenizer,
    model_context_window: int,
    raw_context: str,
    question: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    question_text = str(question).strip()
    template = compare_mod.MEMORY_CONTEXT_QA_PROMPT_TEMPLATE

    def prompt_builder(candidate_context: str, question_text=question_text) -> str:
        return template.format(context=candidate_context, question=question_text)

    final_context = compare_mod.clip_context_text_to_model_limit(
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


def _build_deltamem_official_query_prompt(
    *,
    compare_mod,
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
    context_chunks = build_official_mab_context_chunks(
        [{"context": raw_context}],
        chunk_size=chunk_size,
        eval_utils_module=eval_utils,
    )
    memorized_context = build_official_mab_memorized_context(source, context_chunks[0])
    truncated_context = truncate_official_mab_memory_context(
        memorized_context,
        tokenizer=tokenizer,
        context_max_length=context_max_length,
        raw_input_length_limit=model_context_window,
        buffer_length=official_buffer_length,
        generation_max_length=max_new_tokens,
    )
    system_message = get_official_mab_template(source, "system", "Long_context_agent_deltamem")

    def prompt_builder(candidate_context: str, query=query) -> str:
        return compare_mod._render_message_prompt(
            tokenizer,
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{candidate_context}\n{query}".strip()},
            ],
            use_chat_template=False,
        )

    final_context = compare_mod.clip_context_text_to_model_limit(
        truncated_context,
        tokenizer=tokenizer,
        prompt_builder=prompt_builder,
        use_chat_template=False,
        model_context_window=model_context_window,
        max_new_tokens=max_new_tokens,
        max_context_chars=0,
        keep="tail",
    )
    rendered_prompt = prompt_builder(final_context)
    return rendered_prompt, final_context


def _evaluate_row(
    *,
    item: dict[str, Any],
    runner,
    eval_utils,
    compare_mod,
    agent_config: dict[str, Any],
    use_official_prompt: bool,
    max_context_chars: int,
) -> list[tuple[int, dict[str, object]]]:
    clip_context_text = compare_mod.clip_context_text
    memory_agent_bench_source = compare_mod.memory_agent_bench_source
    memory_agent_bench_primary_metric_name = compare_mod.memory_agent_bench_primary_metric_name
    memory_qa_is_correct = compare_mod.memory_qa_is_correct
    qa_alias_max_f1 = compare_mod.qa_alias_max_f1
    extract_first_line = compare_mod.extract_first_line
    query_include_context = bool(agent_config.get("query_include_context", True))

    raw_context = str(item.get("context", ""))
    if max_context_chars > 0 and len(raw_context) > max_context_chars:
        raw_context = clip_context_text(raw_context, max_context_chars)

    source = memory_agent_bench_source(item)
    source_config = dict(item.get("official_source_config") or {})
    chunk_size = int(source_config.get("chunk_size") or 4096)
    row_max_new_tokens = int(source_config.get("generation_max_length") or 128)
    selected_questions = list(item.get("selected_questions") or [])

    model_context_window = compare_mod.infer_model_context_window(
        runner.metanetwork.metamodel,
        runner.tokenizer,
    )
    runner.configure_for_row(
        model_context_window=model_context_window,
        max_new_tokens=row_max_new_tokens,
        max_context_chars=max_context_chars if max_context_chars > 0 else 0,
    )
    if query_include_context:
        runner.set_query_max_length(model_context_window)

    runner.reset_context()
    if use_official_prompt:
        context_chunks = build_official_mab_context_chunks(
            [{"context": raw_context}],
            chunk_size=chunk_size,
            eval_utils_module=eval_utils,
        )
        memorize_template = get_official_mab_template(source, "memorize", "Long_context_agent_deltamem")
        for chunk in context_chunks[0]:
            runner.memorize_chunk(chunk, memorize_template)
        query_pairs = build_official_mab_query_answer_pairs(
            {
                "questions": [q["question"] for q in selected_questions],
                "answers": [q.get("answer_raw", []) for q in selected_questions],
                "metadata": {
                    "source": source,
                    "question_dates": [q.get("question_date") for q in selected_questions],
                    "question_types": [q.get("question_type") for q in selected_questions],
                    "question_ids": [q.get("question_id") for q in selected_questions],
                    "previous_events": [q.get("previous_event") for q in selected_questions],
                    "qa_pair_ids": [q.get("qa_pair_id") for q in selected_questions],
                },
            },
            source=source,
        )
        query_texts = [q for q, _, _ in query_pairs]
    else:
        chunks = eval_utils.chunk_text_into_sentences(raw_context, chunk_size=chunk_size)
        dummy_template = "{context}"
        for chunk in chunks:
            runner.memorize_chunk(chunk, dummy_template)
        query_texts = [str(q["question"]).strip() for q in selected_questions]

    records: list[tuple[int, dict[str, object]]] = []
    for question_meta, query_text in zip(selected_questions, query_texts):
        prompt_context_chars = 0
        if query_include_context:
            if use_official_prompt:
                formatted_query, prompt_context = _build_deltamem_official_query_prompt(
                    compare_mod=compare_mod,
                    tokenizer=runner.tokenizer,
                    model_context_window=model_context_window,
                    raw_context=raw_context,
                    source=source,
                    query=query_text,
                    eval_utils=eval_utils,
                    source_config=source_config,
                    max_new_tokens=row_max_new_tokens,
                )
                prompt_context_chars = len(prompt_context)
                prompt_style = "official_memorize_query_templates_shine_lora"
            else:
                formatted_query, prompt_context = _build_deltamem_unified_query_prompt(
                    compare_mod=compare_mod,
                    tokenizer=runner.tokenizer,
                    model_context_window=model_context_window,
                    raw_context=raw_context,
                    question=str(question_meta["question"]),
                    max_new_tokens=row_max_new_tokens,
                )
                prompt_context_chars = len(prompt_context)
                prompt_style = "unified_memory_context_qa_shine_lora"
        else:
            formatted_query = (
                query_text
                if use_official_prompt
                else _shine_question_only_query(str(question_meta["question"]))
            )
            prompt_style = (
                "official_memorize_query_templates_shine"
                if use_official_prompt
                else "unified_memory_context_qa_shine_question_only"
            )

        output = runner.query(formatted_query)
        prediction = str(output.get("output") or "")
        answer_aliases = list(question_meta.get("answer_aliases") or [])
        correct = memory_qa_is_correct(prediction, answer_aliases)
        f1 = qa_alias_max_f1(prediction, answer_aliases)
        primary_metric = memory_agent_bench_primary_metric_name(source)
        primary_score = round(f1, 4) if primary_metric == "f1" else float(correct)
        eval_index = int(question_meta["eval_index"])
        records.append(
            (
                eval_index,
                {
                    "row_id": item.get("row_id"),
                    "split": item.get("split"),
                    "source": source,
                    "question_id": question_meta.get("question_id"),
                    "qa_pair_id": question_meta.get("qa_pair_id"),
                    "question_type": question_meta.get("question_type"),
                    "question_date": question_meta.get("question_date"),
                    "question": question_meta["question"],
                    "query": query_text if use_official_prompt else str(question_meta["question"]),
                    "answer_aliases": answer_aliases,
                    "prediction": prediction,
                    "extracted_answer": extract_first_line(prediction),
                    "context_chars": len(raw_context),
                    "prompt_context_chars": prompt_context_chars,
                    "max_new_tokens": row_max_new_tokens,
                    "prompt_style": prompt_style,
                    "query_include_context": query_include_context,
                    "correct": correct,
                    "f1": round(f1, 4),
                    "primary_metric": primary_metric,
                    "primary_score": primary_score,
                    "input_len": output.get("input_len"),
                    "output_len": output.get("output_len"),
                    "memory_construction_time": output.get("memory_construction_time"),
                    "query_time_len": output.get("query_time_len"),
                },
            )
        )
    return records


def evaluate_shine_memory_agent_bench(
    *,
    row_tasks: list[MemoryAgentBenchRowTask],
    args: Namespace,
    context: DistributedContext,
    progress_bar=None,
) -> list[tuple[int, dict[str, object]]]:
    from deltamem.eval import benchmark_compare as compare_mod
    from deltamem.eval.common import gather_indexed_records
    from methods.shine_runner import ShineMABRunner

    agent_config_path = Path(args.shine_agent_config).resolve()
    agent_config = _configure_agent_for_suite(_load_agent_config(agent_config_path), args)
    shine_root = _resolve_shine_root(agent_config, agent_config_path)
    mab_root = Path(args.external_memory_agent_bench_root).resolve()
    _ensure_paths(shine_root=shine_root, mab_root=mab_root)

    eval_utils = load_mab_eval_utils(mab_root)
    use_official_prompt = bool(args.memory_agent_bench_use_official_prompt)
    local_records: list[tuple[int, dict[str, object]]] = []

    for row_task in row_tasks:
        item = row_task.item
        source = compare_mod.memory_agent_bench_source(item)
        source_config = dict(item.get("official_source_config") or {})
        dataset_config = {
            "sub_dataset": source,
            "context_max_length": int(source_config.get("context_max_length") or 131072),
            "generation_max_length": int(source_config.get("generation_max_length") or 128),
        }
        runner = ShineMABRunner(agent_config, dataset_config)
        row_records = _evaluate_row(
            item=item,
            runner=runner,
            eval_utils=eval_utils,
            compare_mod=compare_mod,
            agent_config=agent_config,
            use_official_prompt=use_official_prompt,
            max_context_chars=int(args.memory_agent_bench_max_context_chars),
        )
        local_records.extend(row_records)
        if progress_bar is not None:
            progress_bar.update(row_task.question_count)
        del runner

    return gather_indexed_records(local_records, context)
