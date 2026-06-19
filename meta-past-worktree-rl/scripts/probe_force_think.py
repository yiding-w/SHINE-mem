"""Probe: compare SHINE-orig completions with vs. without force_think.

For each hardcoded (context, question) example: build a LoRA from the
context via the SHINE hypernet, push to vLLM, sample greedily twice —
once with the prompt ending at ``<|im_start|>assistant\\n`` (force_think
off, the new default), once with ``<think>\\n`` appended (force_think
on). Prints the raw completion text for both so we can see whether the
SHINE-pretrained model gracefully fills the forced think block, or
emits garbage.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import torch

    from meta_past.eval.prompts import format_shine_prompt
    from meta_past.rl.lora_format import (
        QWEN3_TARGET_MODULES,
        peft_meta_for_qwen3,
        shine_loradict_to_peft_batch,
    )
    from meta_past.rl.vllm_engine import VLLMEngine, VLLMEngineConfig
    from meta_past.shine_adapter import ShineHypernet

    backbone = os.path.expanduser("~/huggingfacemodels/Qwen3-8B")
    ckpt = os.path.expanduser("~/huggingfacemodels/SHINE-ift_mqa_1qa")

    examples = [
        # (label, context, question)
        ("squad-style",
         "Albert Einstein developed the theory of general relativity in 1915. "
         "It built on his earlier special relativity (1905). He was awarded "
         "the Nobel Prize in Physics in 1921, although the prize was given for "
         "his explanation of the photoelectric effect rather than for "
         "relativity.",
         "In what year was general relativity developed?"),
        ("multi-hop",
         "The painting 'Starry Night' was created by Vincent van Gogh in 1889. "
         "Van Gogh was a Dutch post-impressionist painter. He was born in "
         "Zundert, in the southern Netherlands, in 1853.",
         "In what country was the creator of 'Starry Night' born?"),
        ("counting",
         "A bookshelf has 3 shelves. The top shelf has 7 books, the middle "
         "shelf has 5 books, and the bottom shelf has 12 books.",
         "How many books are on the bookshelf in total?"),
        ("entity-disambig",
         "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and "
         "Ronald Wayne in Cupertino, California. The company is best known "
         "for the iPhone, first released in 2007.",
         "In which city was Apple founded?"),
    ]

    print("=== booting vLLM ===")
    engine = VLLMEngine(VLLMEngineConfig(
        model_path=backbone,
        max_loras=4,
        max_lora_rank=8,
        max_model_len=2048,
        dtype="bfloat16",
        gpu_memory_utilization=0.40,
        enforce_eager=False,
        seed=0,
    ))
    engine.boot()

    print(f"=== loading SHINE hypernet from {ckpt} ===")
    net = ShineHypernet(
        ckpt_dir=ckpt,
        device="cuda:0",
        backbone=backbone,
        lora_r=8,
        metalora_r=128,
    )
    peft_meta = peft_meta_for_qwen3(lora_r=net.lora_r,
                                    target_modules=QWEN3_TARGET_MODULES)

    # Tokenize all contexts in one batched hypernet forward.
    tok = net.tokenizer
    ctx_enc = tok(
        [c for _, c, _ in examples],
        max_length=1024, truncation=True,
        return_tensors="pt", padding="max_length",
    )
    ev_ids = ctx_enc["input_ids"].to(net.device)
    ev_mask = ctx_enc["attention_mask"].to(net.device)
    with torch.no_grad():
        loradict = net.generate_lora(ev_ids, ev_mask)
    per_b = shine_loradict_to_peft_batch(loradict)
    per_b_cpu = [
        {k: v.detach().to("cpu", copy=True) for k, v in d.items()}
        for d in per_b
    ]
    lora_ids = list(range(1, 1 + len(examples)))

    engine.wake_up(tags=["weights", "kv_cache"])
    torch.cuda.empty_cache()
    engine.push_lora_batch(per_b_cpu, lora_ids, peft_meta)

    for force_think in (False, True):
        prompts = [
            format_shine_prompt(tok, q, enable_thinking=True,
                                force_think=force_think)
            for _, _, q in examples
        ]
        out = engine.complete(
            prompts=prompts,
            lora_ids=lora_ids,
            n=1,
            temperature=0.0,
            max_tokens=128,
        )
        print()
        print("#" * 70)
        print(f"#  force_think = {force_think}")
        print("#" * 70)
        for (label, _ctx, q), samples in zip(examples, out):
            text = samples[0].text
            print()
            print(f"--- [{label}] Q: {q}")
            print(f"COMPLETION (raw):\n{text!r}")

    engine.shutdown()


if __name__ == "__main__":
    main()
