#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import string
from pathlib import Path

import torch

from MemoryTest.prepare_data.prompt_templates import build_context, format_answer, question_prompt, reconstruction_prompt
from MemoryTest.evaluation.metrics import make_eval_row, summarize_examples
from MemoryTest.training.lora_sft_utils import load_runtime_args
from MemoryTest.training.shine_train_utils import (
    append_jsonl,
    cast_floating_tensors,
    compute_combined_lora_loss,
    read_json,
    resolve_path,
    sample_context,
    save_posttrain_checkpoint,
    set_posttrain_requires_grad,
    trainable_generate_context_lora,
)


LOGGER = logging.getLogger("posttrain_shine_memory")

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train SHINE on MemoryTest semantic facts.")
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--val-file", type=str, default="MemoryTest/json_data/splits/semantic_val.json")
    parser.add_argument("--test-file", type=str, default="MemoryTest/json_data/splits/semantic_test.json")
    parser.add_argument("--output-dir", type=str, default="MemoryTest/checkpoints/shine_memory_posttrain")
    parser.add_argument("--fact-counts", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20])
    parser.add_argument("--qa-per-context", type=int, default=4)
    parser.add_argument("--context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-trials", type=int, default=20)
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
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--use-reconstruction", action="store_true")
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def build_training_records(context_rows: list[dict], qa_rows: list[dict], use_contrastive: bool, use_reconstruction: bool) -> list[dict]:
    records = []
    for row in qa_rows:
        records.append(
            {
                "category": "answer",
                "prompt": question_prompt(row["question"]),
                "answer": " " + format_answer(row["answer"]),
            }
        )
    if use_contrastive:
        letters = list(string.ascii_uppercase)
        for qa in qa_rows:
            candidates = []
            seen = set()
            for row in [qa] + context_rows:
                answer = str(row["answer"])
                if answer.casefold() not in seen:
                    candidates.append(answer)
                    seen.add(answer.casefold())
            candidates = candidates[: min(len(candidates), len(letters))]
            gold_idx = candidates.index(str(qa["answer"]))
            option_lines = [f"{letters[idx]}. {answer}" for idx, answer in enumerate(candidates)]
            prompt = (
                "Choose the option that answers the question using the current memory.\n\n"
                f"Question: {qa['question']}\n"
                + "\n".join(option_lines)
                + "\nAnswer with the option letter only."
            )
            records.append(
                {
                    "category": "contrastive",
                    "prompt": prompt,
                    "answer": " " + letters[gold_idx],
                }
            )
    if use_reconstruction:
        records.append(
            {
                "category": "reconstruction",
                "prompt": reconstruction_prompt(),
                "answer": "\n" + build_context(context_rows, context_format="structured"),
            }
        )
    return records


def default_train_file() -> str:
    augmented = resolve_path("MemoryTest/json_data/splits/semantic_train_augmented.json")
    if augmented.exists():
        return str(augmented)
    return "MemoryTest/json_data/splits/semantic_train.json"


def load_shine_for_training(args: argparse.Namespace):
    from MemoryTest.case_test import build_cfg, load_runtime, resolve_device, resolve_torch_dtype

    runtime_args = load_runtime_args(args.runtime_config)
    checkpoint_dir = args.checkpoint_dir
    latest_dir = resolve_path(args.output_dir) / "latest"
    if args.resume and latest_dir.exists():
        checkpoint_dir = str(latest_dir)
    runtime_args.checkpoint_dir = checkpoint_dir
    if args.device is not None:
        runtime_args.device = args.device
    if args.gpu_id is not None:
        runtime_args.gpu_id = args.gpu_id
    if args.context_max_length is not None:
        runtime_args.context_max_length = args.context_max_length
    if args.conversation_max_length is not None:
        runtime_args.conversation_max_length = args.conversation_max_length
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    cfg.model.torch_dtype = args.torch_dtype
    metanetwork, metalora, tokenizer = load_runtime(cfg, runtime_args.checkpoint_dir, device)
    dtype = resolve_torch_dtype(args.torch_dtype)
    if isinstance(dtype, torch.dtype):
        metanetwork.to(device=device, dtype=dtype)
        metalora = cast_floating_tensors(metalora, dtype)
    if hasattr(metanetwork.metamodel, "config"):
        metanetwork.metamodel.config.use_cache = False
    if hasattr(metanetwork.metamodel, "gradient_checkpointing_enable") and args.use_answer_gradient_checkpoint:
        metanetwork.metamodel.gradient_checkpointing_enable()
    metanetwork.train()
    trainable = set_posttrain_requires_grad(metanetwork, metalora)
    return runtime_args, cfg, device, metanetwork, metalora, tokenizer, trainable


def encode_choice_batch(tokenizer, context_rows: list[dict], qa_rows: list[dict], max_length: int, device):
    import string

    letters = list(string.ascii_uppercase)
    input_rows = []
    label_rows = []
    for qa in qa_rows:
        candidates = []
        seen = set()
        for row in [qa] + context_rows:
            answer = str(row["answer"])
            if answer.casefold() not in seen:
                candidates.append(answer)
                seen.add(answer.casefold())
        candidates = candidates[: min(len(candidates), len(letters))]
        gold_idx = candidates.index(str(qa["answer"]))
        option_lines = [f"{letters[idx]}. {answer}" for idx, answer in enumerate(candidates)]
        prompt = (
            "Choose the option that answers the question using the current memory.\n\n"
            f"Question: {qa['question']}\n"
            + "\n".join(option_lines)
            + "\nAnswer with the option letter only."
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        answer_ids = tokenizer(" " + letters[gold_idx] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[-max_length:]
        labels = ([-100] * len(prompt_ids) + answer_ids)[-max_length:]
        input_rows.append(input_ids)
        label_rows.append(labels)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row) for row in input_rows)
    return {
        "input_ids": torch.tensor([[pad_id] * (max_len - len(row)) + row for row in input_rows], dtype=torch.long, device=device),
        "labels": torch.tensor([[-100] * (max_len - len(row)) + labels for row, labels in zip(input_rows, label_rows)], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[0] * (max_len - len(row)) + [1] * len(row) for row in input_rows], dtype=torch.long, device=device),
    }


def compute_contrastive_loss(context_rows, qa_rows, lora_dict, metanetwork, tokenizer, device, max_length, use_gradient_checkpoint: bool = False):
    batch = encode_choice_batch(tokenizer, context_rows, qa_rows, max_length=max_length, device=device)
    outputs = metanetwork.metamodel(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        loradict=lora_dict,
        ignore_mem_token=True,
        use_gradient_checkpoint=use_gradient_checkpoint,
    )
    return outputs.loss


def generate_answer(metanetwork, tokenizer, lora_dict, question: str, device, max_new_tokens: int, max_length: int):
    from MemoryTest.case_test import extract_think_and_answer
    from MemoryTest.prepare_data.prompt_templates import question_prompt

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_prompt(question)}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
    raw = tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True)
    _, answer = extract_think_and_answer(raw)
    return answer, raw


def evaluate_current(metanetwork, metalora, tokenizer, cfg, rows, args, runtime_args, device, rng) -> dict:
    metanetwork.eval()
    eval_rows = []
    progress = tqdm(
        range(args.eval_trials),
        desc="posttrain val",
        dynamic_ncols=True,
        leave=False,
    ) if tqdm is not None else range(args.eval_trials)
    with torch.no_grad():
        for trial in progress:
            context_rows, qa_rows = sample_context(rows, args.fact_counts, args.qa_per_context, rng)
            context = build_context(context_rows, context_format=args.context_format)
            from MemoryTest.case_test import generate_context_lora

            lora_dict = generate_context_lora(context, metanetwork, tokenizer, metalora, cfg, device)
            for fact in qa_rows:
                answer, raw = generate_answer(
                    metanetwork,
                    tokenizer,
                    lora_dict,
                    fact["question"],
                    device,
                    args.max_new_tokens or runtime_args.max_new_tokens,
                    runtime_args.conversation_max_length,
                )
                eval_rows.append(make_eval_row(len(eval_rows) + 1, fact, answer, raw=raw))
            if tqdm is not None:
                summary = summarize_examples(eval_rows)
                progress.set_postfix({"acc": f"{summary['accuracy']:.4f}", "n": summary["total"]})
    metanetwork.train()
    return summarize_examples(eval_rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not args.train_file:
        args.train_file = default_train_file()
    train_rows = read_json(args.train_file)
    val_rows = read_json(args.val_file)
    output_dir = resolve_path(args.output_dir)
    log_path = output_dir / "shine_posttrain_train_log.jsonl"
    rng = random.Random(args.seed)

    runtime_args, cfg, device, metanetwork, metalora, tokenizer, trainable = load_shine_for_training(args)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val_acc = -1.0

    train_progress = tqdm(
        range(1, args.max_steps + 1),
        desc="SHINE posttrain",
        dynamic_ncols=True,
        leave=True,
    ) if tqdm is not None else range(1, args.max_steps + 1)

    for step in train_progress:
        context_rows, qa_rows = sample_context(train_rows, args.fact_counts, args.qa_per_context, rng)
        context_format = rng.choice(["natural", "structured"]) if args.context_format == "mixed" else args.context_format
        context = build_context(context_rows, context_format=context_format)
        lora_dict = trainable_generate_context_lora(
            context,
            metanetwork,
            tokenizer,
            metalora,
            cfg,
            device,
            use_gradient_checkpoint=args.use_gradient_checkpoint,
        )
        training_records = build_training_records(context_rows, qa_rows, args.use_contrastive, args.use_reconstruction)
        category_losses = compute_combined_lora_loss(
            training_records,
            lora_dict,
            metanetwork,
            tokenizer,
            device,
            args.answer_max_length,
            use_gradient_checkpoint=args.use_answer_gradient_checkpoint,
        )
        answer_loss = category_losses["answer"]
        loss = answer_loss
        contrastive_loss = category_losses.get("contrastive")
        recon_loss = category_losses.get("reconstruction")
        if args.use_contrastive:
            loss = loss + args.contrastive_weight * contrastive_loss
        if args.use_reconstruction:
            loss = loss + args.recon_weight * recon_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
        optimizer.step()

        log_record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "answer_loss": float(answer_loss.detach().cpu()),
            "contrastive_loss": float(contrastive_loss.detach().cpu()) if contrastive_loss is not None else None,
            "reconstruction_loss": float(recon_loss.detach().cpu()) if recon_loss is not None else None,
            "context_fact_ids": [row["id"] for row in context_rows],
            "qa_fact_ids": [row["id"] for row in qa_rows],
            "context_format": context_format,
        }
        if tqdm is not None:
            postfix = {
                "loss": f"{log_record['loss']:.4f}",
                "answer": f"{log_record['answer_loss']:.4f}",
            }
            if contrastive_loss is not None:
                postfix["choice"] = f"{log_record['contrastive_loss']:.4f}"
            if recon_loss is not None:
                postfix["recon"] = f"{log_record['reconstruction_loss']:.4f}"
            train_progress.set_postfix(postfix)
        if step % 10 == 0 or step == 1:
            append_jsonl(log_path, log_record)
            LOGGER.info("step=%s loss=%.6f answer=%.6f", step, log_record["loss"], log_record["answer_loss"])

        if step % args.eval_every == 0 or step == args.max_steps:
            val_summary = evaluate_current(metanetwork, metalora, tokenizer, cfg, val_rows, args, runtime_args, device, rng)
            log_record["val"] = val_summary
            append_jsonl(log_path, log_record)
            LOGGER.info("val step=%s accuracy=%.4f", step, val_summary["accuracy"])
            if tqdm is not None:
                train_progress.set_postfix({"loss": f"{log_record['loss']:.4f}", "val_acc": f"{val_summary['accuracy']:.4f}", "best": f"{max(best_val_acc, val_summary['accuracy']):.4f}"})
            save_posttrain_checkpoint(output_dir / "latest", metanetwork, metalora, extra_state={"step": step, "val": val_summary, "config": vars(args)})
            if val_summary["accuracy"] > best_val_acc:
                best_val_acc = val_summary["accuracy"]
                save_posttrain_checkpoint(output_dir / "best", metanetwork, metalora, extra_state={"step": step, "val": val_summary, "config": vars(args)})
        elif step % args.save_every == 0:
            save_posttrain_checkpoint(output_dir / "latest", metanetwork, metalora, extra_state={"step": step, "config": vars(args)})
        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_summary = {"best_val_accuracy": best_val_acc, "max_steps": args.max_steps, "config": vars(args)}
    (output_dir / "summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del metanetwork, metalora, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
