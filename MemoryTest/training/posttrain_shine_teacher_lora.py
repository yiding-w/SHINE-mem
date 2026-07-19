#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from collections import OrderedDict

import torch

from MemoryTest.prepare_data.prompt_templates import build_context, question_prompt, format_answer
from MemoryTest.training.posttrain_shine_memory import (
    build_optimizer,
    evaluate_current,
    load_shine_for_training,
    usable_fact_counts,
)
from MemoryTest.training.shine_train_utils import (
    append_jsonl,
    clamp_lora_tensors,
    compute_combined_lora_loss,
    lora_tensor_stats,
    read_json,
    resolve_path,
    save_posttrain_checkpoint,
    trainable_generate_context_lora,
)
from MemoryTest.teacher_lora.teacher_lora_utils import (
    load_bank_index,
    load_teacher_lora_entry,
    lora_delta_alignment_loss,
    read_json as read_teacher_json,
)


LOGGER = logging.getLogger("posttrain_shine_teacher_lora")

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train SHINE by aligning generated LoRA with offline teacher LoRA targets.")
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint-profile",
        choices=["auto", "pretrain", "ift"],
        default="auto",
    )
    parser.add_argument("--teacher-bank-dir", type=str, required=True)
    parser.add_argument("--val-file", type=str, default="MemoryTest/json_data/splits/semantic_val.json")
    parser.add_argument("--output-dir", type=str, default="MemoryTest/checkpoints/shine_teacher_lora_posttrain")
    parser.add_argument("--fact-counts", type=int, nargs="+", default=None, help="Optional filter over teacher-bank num_facts.")
    parser.add_argument("--max-train-contexts", type=int, default=0, help="Use only the first N matching teacher contexts after shuffle/filter; 0 means all.")
    parser.add_argument("--min-teacher-accuracy", type=float, default=0.0)
    parser.add_argument("--qa-per-context", type=int, default=4)
    parser.add_argument("--context-format", choices=["teacher", "natural", "structured", "mixed"], default="teacher")
    parser.add_argument("--eval-context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--metalora-learning-rate", type=float, default=None)
    parser.add_argument("--mem-token-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--teacher-align-weight", type=float, default=1.0)
    parser.add_argument("--answer-weight", type=float, default=0.1)
    parser.add_argument("--alignment-eps", type=float, default=1e-8)
    parser.add_argument("--include-bias-alignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-trials", type=int, default=20)
    parser.add_argument("--eval-fact-counts", type=int, nargs="+", default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-max-length", type=int, default=None)
    parser.add_argument("--conversation-max-length", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="bf16")
    parser.add_argument("--use-gradient-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-answer-gradient-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--empty-cache-every", type=int, default=0)
    parser.add_argument("--generated-lora-clamp", type=float, default=10.0)
    parser.add_argument("--freeze-metalora", action="store_true")
    parser.add_argument("--freeze-mem-tokens", action="store_true")
    parser.add_argument("--teacher-cache-size", type=int, default=0, help="Number of teacher LoRAs to keep in CPU RAM. 0 disables caching.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


class TeacherLoraCache:
    def __init__(self, max_size: int):
        self.max_size = max(0, max_size)
        self.cache: OrderedDict[str, object] = OrderedDict()

    def get(self, entry: dict, device: torch.device, dtype: torch.dtype):
        key = str(entry["lora_path"])
        if self.max_size > 0 and key in self.cache:
            cpu_lora = self.cache.pop(key)
            self.cache[key] = cpu_lora
            return move_cached_lora_to_device(cpu_lora, device, dtype)
        lora = load_teacher_lora_entry(entry, device=device, dtype=dtype)
        if self.max_size > 0:
            from MemoryTest.training.lora_sft_utils import move_lora_to_cpu

            self.cache[key] = move_lora_to_cpu(lora)
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        return lora


def move_cached_lora_to_device(obj, device: torch.device, dtype: torch.dtype):
    from MemoryTest.teacher_lora.teacher_lora_utils import move_lora_to_device

    return move_lora_to_device(obj, device=device, dtype=dtype)


def model_lora_dtype(metanetwork) -> torch.dtype:
    return metanetwork.metamodel.get_input_embeddings().weight.dtype


def load_teacher_entries(args: argparse.Namespace) -> list[dict]:
    entries = load_bank_index(args.teacher_bank_dir)
    filtered = []
    requested_counts = set(args.fact_counts or [])
    for entry in entries:
        if requested_counts and int(entry.get("num_facts", -1)) not in requested_counts:
            continue
        if float(entry.get("train_accuracy", 0.0)) < args.min_teacher_accuracy:
            continue
        meta = read_teacher_json(entry["meta_path"])
        merged = {**entry, "meta": meta, "facts": meta["facts"], "context": meta["context"]}
        filtered.append(merged)
    if not filtered:
        raise ValueError("No teacher LoRA entries remain after filtering.")
    rng = random.Random(args.seed)
    rng.shuffle(filtered)
    if args.max_train_contexts > 0:
        filtered = filtered[: args.max_train_contexts]
    LOGGER.info("Loaded %s teacher LoRA entries from %s", len(filtered), args.teacher_bank_dir)
    return filtered


def choose_context(entry: dict, context_format: str, step: int, rng: random.Random) -> tuple[str, str]:
    rows = entry["facts"]
    if context_format == "teacher":
        return str(entry["context"]), str(entry.get("context_format", entry["meta"].get("context_format", "teacher")))
    if context_format == "mixed":
        fmt = "structured" if (step + rng.randint(0, 1)) % 2 else "natural"
    else:
        fmt = context_format
    return build_context(rows, context_format=fmt), fmt


def build_answer_records(qa_rows: list[dict]) -> list[dict]:
    return [
        {
            "category": "answer",
            "prompt": question_prompt(row["question"]),
            "answer": " " + format_answer(row["answer"]),
        }
        for row in qa_rows
    ]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    teacher_entries = load_teacher_entries(args)
    val_rows = read_json(args.val_file)
    output_dir = resolve_path(args.output_dir)
    log_path = output_dir / "shine_teacher_lora_train_log.jsonl"
    rng = random.Random(args.seed)

    runtime_args, cfg, device, metanetwork, metalora, tokenizer, trainable = load_shine_for_training(args)
    optimizer = build_optimizer(trainable, args)
    teacher_cache = TeacherLoraCache(args.teacher_cache_size)
    teacher_dtype = model_lora_dtype(metanetwork)
    best_val_acc = -1.0

    eval_fact_counts = args.eval_fact_counts or args.fact_counts or [1, 2, 4, 8, 12, 20]
    args.eval_fact_counts = usable_fact_counts(eval_fact_counts, val_rows, "val")

    train_progress = tqdm(
        range(1, args.max_steps + 1),
        desc="SHINE teacher-LoRA posttrain",
        dynamic_ncols=True,
        leave=True,
    ) if tqdm is not None else range(1, args.max_steps + 1)

    for step in train_progress:
        entry = teacher_entries[(step - 1) % len(teacher_entries)]
        context_rows = entry["facts"]
        qa_rows = rng.sample(context_rows, min(args.qa_per_context, len(context_rows)))
        context, context_format = choose_context(entry, args.context_format, step, rng)
        teacher_lora = teacher_cache.get(entry, device=device, dtype=teacher_dtype)

        student_lora = trainable_generate_context_lora(
            context,
            metanetwork,
            tokenizer,
            metalora,
            cfg,
            device,
            use_gradient_checkpoint=args.use_gradient_checkpoint,
        )
        student_lora = clamp_lora_tensors(student_lora, args.generated_lora_clamp)
        lora_stats = lora_tensor_stats(student_lora)
        if lora_stats["finite"] < 1.0:
            raise FloatingPointError(f"Generated student LoRA is non-finite at step {step}.")

        align_loss, align_stats = lora_delta_alignment_loss(
            student_lora,
            teacher_lora,
            eps=args.alignment_eps,
            include_bias=args.include_bias_alignment,
        )
        answer_loss = align_loss.new_tensor(0.0)
        if args.answer_weight > 0:
            answer_records = build_answer_records(qa_rows)
            category_losses = compute_combined_lora_loss(
                answer_records,
                student_lora,
                metanetwork,
                tokenizer,
                device,
                args.answer_max_length,
                use_gradient_checkpoint=args.use_answer_gradient_checkpoint,
            )
            answer_loss = category_losses["answer"]
        loss = args.teacher_align_weight * align_loss + args.answer_weight * answer_loss

        if not torch.isfinite(loss):
            bad_record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "alignment_loss": float(align_loss.detach().cpu()),
                "answer_loss": float(answer_loss.detach().cpu()),
                "context_id": entry["context_id"],
                "context_fact_ids": [row["id"] for row in context_rows],
                "qa_fact_ids": [row["id"] for row in qa_rows],
                "context_format": context_format,
                "nonfinite": True,
                "lora_stats": lora_stats,
                "alignment_stats": align_stats,
            }
            append_jsonl(log_path, bad_record)
            raise FloatingPointError("Non-finite teacher-alignment loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            trainable_params = [param for group in trainable.values() for param in group]
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()

        log_record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "alignment_loss": float(align_loss.detach().cpu()),
            "answer_loss": float(answer_loss.detach().cpu()),
            "context_id": entry["context_id"],
            "context_fact_ids": [row["id"] for row in context_rows],
            "qa_fact_ids": [row["id"] for row in qa_rows],
            "context_format": context_format,
            "lora_stats": lora_stats,
            "alignment_stats": {
                "num_modules": align_stats["num_modules"],
                "mean_module_loss": align_stats["mean_module_loss"],
            },
        }
        if tqdm is not None:
            train_progress.set_postfix(
                {
                    "loss": f"{log_record['loss']:.4f}",
                    "align": f"{log_record['alignment_loss']:.4f}",
                    "answer": f"{log_record['answer_loss']:.4f}",
                    "lora_max": f"{lora_stats['max_abs']:.2f}",
                }
            )
        if step % 10 == 0 or step == 1:
            append_jsonl(log_path, log_record)
            LOGGER.info(
                "step=%s loss=%.6f align=%.6f answer=%.6f context=%s",
                step,
                log_record["loss"],
                log_record["alignment_loss"],
                log_record["answer_loss"],
                entry["context_id"],
            )

        if step % args.eval_every == 0 or step == args.max_steps:
            train_context_format = args.context_format
            try:
                args.context_format = args.eval_context_format
                val_summary = evaluate_current(metanetwork, metalora, tokenizer, cfg, val_rows, args, runtime_args, device, rng)
            finally:
                args.context_format = train_context_format
            log_record["val"] = val_summary
            append_jsonl(log_path, log_record)
            LOGGER.info("val step=%s accuracy=%.4f", step, val_summary["accuracy"])
            save_posttrain_checkpoint(output_dir / "latest", metanetwork, metalora, extra_state={"step": step, "val": val_summary, "config": vars(args)})
            if val_summary["accuracy"] > best_val_acc:
                best_val_acc = val_summary["accuracy"]
                save_posttrain_checkpoint(output_dir / "best", metanetwork, metalora, extra_state={"step": step, "val": val_summary, "config": vars(args)})
        elif step % args.save_every == 0:
            save_posttrain_checkpoint(output_dir / "latest", metanetwork, metalora, extra_state={"step": step, "config": vars(args)})
        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_summary = {
        "best_val_accuracy": best_val_acc,
        "max_steps": args.max_steps,
        "num_teacher_entries": len(teacher_entries),
        "config": vars(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del metanetwork, metalora, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
