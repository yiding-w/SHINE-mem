"""
Controlled diagnostic for the D2L runner — isolates "integration bug" vs
"D2L can't do long context". Run from the MAB root in the D2L env:

    cd ~/wyd/SHINE-mem/MemoryAgentBench
    python d2l_smoke_test.py

It exercises the EXACT DocToLoraRunner.internalize/query path on a known needle
at increasing context lengths and prints the model output each time.
"""

from __future__ import annotations

import os

import yaml

from methods.doc_to_lora_runner import DocToLoraRunner

CONFIG = "configs/agent_conf/DocToLora_Agents/doc_to_lora_agent_qwen3_4b.yaml"
NEEDLE = "The secret passcode for the vault is BANANA-42."
QUESTION = (
    "Use only the memory context to answer.\n"
    "Question: What is the secret passcode for the vault?\n"
    "Answer:"
)


def run_case(name: str, doc: str, agent_config: dict) -> None:
    dataset_config = {"sub_dataset": "needle_test", "generation_max_length": 32}
    runner = DocToLoraRunner(agent_config, dataset_config)

    # CONTROL: no context internalized (empty LoRA) — what the model says blind.
    runner.reset_context()
    ctrl = runner.query(QUESTION)["output"]

    # TREATMENT: internalize the doc containing the needle.
    runner.reset_context()
    runner.memorize_chunk(doc, "{context}")
    out = runner.query(QUESTION)
    approx_tokens = len(runner.ctx_tokenizer.encode(doc, add_special_tokens=False))

    print(f"\n===== {name} (~{approx_tokens} ctx tokens) =====")
    print("CONTROL (no ctx):", repr(ctrl))
    print("TREATMENT (ctx) :", repr(out["output"]))
    print("HIT:", "BANANA-42" in out["output"], "| LoRA changed output:", ctrl != out["output"])
    del runner


def main() -> None:
    with open(CONFIG, "r", encoding="utf-8") as f:
        agent_config = yaml.safe_load(f)
    # Deterministic for the test.
    agent_config["temperature"] = 0.0
    # Allow pointing at the real repo/checkpoint without editing the YAML:
    #   export D2L_ROOT=/ceph/home/muhan01/doc-to-lora
    #   export D2L_CHECKPOINT_PATH=/ceph/home/muhan01/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin
    if os.environ.get("D2L_ROOT"):
        agent_config["d2l_root"] = os.environ["D2L_ROOT"]
    if os.environ.get("D2L_CHECKPOINT_PATH"):
        agent_config["d2l_checkpoint_path"] = os.environ["D2L_CHECKPOINT_PATH"]
    print("Using checkpoint:", agent_config["d2l_checkpoint_path"], flush=True)

    filler_sentence = "The sky was clear and the market was busy that morning. "

    # 1) Needle alone (single short chunk) — tests basic internalize->generate.
    run_case("short", NEEDLE, agent_config)

    # 2) Needle in ~4k-token filler (still 1 chunk).
    run_case("mid_4k", filler_sentence * 350 + " " + NEEDLE + " " + filler_sentence * 350, agent_config)

    # 3) Needle in ~30k-token filler (multi-chunk -> exercises combine_lora merge).
    run_case("long_30k", filler_sentence * 2600 + " " + NEEDLE + " " + filler_sentence * 2600, agent_config)


if __name__ == "__main__":
    main()
