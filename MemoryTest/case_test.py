#!/usr/bin/env python
import argparse
import gc
import logging
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

SHINE_ROOT = Path(__file__).resolve().parents[1]
if str(SHINE_ROOT) not in sys.path:
    sys.path.insert(0, str(SHINE_ROOT))

from metanetwork_family import Metanetwork
from utils.myfreeze import freeze
from utils.myinit import _import_class
from utils.mysaveload import load_checkpoint
from utils.myseed import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

LOGGER = logging.getLogger("case_test")

MINIMAL_CHAT_TEMPLATE = """{%- for message in messages %}
{%- if message.role == \"system\" %}
{{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == \"user\" %}
{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == \"assistant\" %}
{{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and not enable_thinking %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
{%- endif %}
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question directly and output nothing else."
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "case_test.yaml"


def _load_arg_defaults(config_path: str):
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    defaults = OmegaConf.to_container(cfg, resolve=True)
    if defaults is None:
        return {}
    if not isinstance(defaults, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return defaults


def _add_configurable_argument(parser, defaults: dict, *flags, config_key: str, **kwargs):
    if config_key in defaults:
        kwargs["default"] = defaults[config_key]
    parser.add_argument(*flags, **kwargs)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file with MemoryTest defaults.",
    )
    config_args, _ = config_parser.parse_known_args()
    defaults = _load_arg_defaults(config_args.config)

    parser = argparse.ArgumentParser(
        description="Compress a context into SHINE LoRA and answer a question.",
        parents=[config_parser],
    )
    _add_configurable_argument(parser, defaults, "--context", type=str, config_key="context", help="Raw context string.")
    _add_configurable_argument(parser, defaults, "--context-file", type=str, config_key="context_file", help="Path to a UTF-8 text file containing the context.")
    _add_configurable_argument(parser, defaults, "--question", type=str, config_key="question", help="Question to answer.")
    _add_configurable_argument(
        parser,
        defaults,
        "--model-path",
        type=str,
        config_key="model_path",
        help="Path to the local Qwen3-8B model directory.",
    )
    _add_configurable_argument(
        parser,
        defaults,
        "--checkpoint-dir",
        type=str,
        config_key="checkpoint_dir",
        help="Path to the SHINE checkpoint directory containing metanetwork.pth and metalora.pth.",
    )
    _add_configurable_argument(
        parser,
        defaults,
        "--save-lora-path",
        type=str,
        config_key="save_lora_path",
        help="Where to save the generated LoRA dictionary.",
    )
    _add_configurable_argument(
        parser,
        defaults,
        "--load-lora-path",
        type=str,
        config_key="load_lora_path",
        help="If set, skip context compression and load an existing LoRA dictionary from this path.",
    )
    _add_configurable_argument(parser, defaults, "--device", type=str, choices=["cuda", "cpu"], config_key="device", help="Inference device type.")
    _add_configurable_argument(parser, defaults, "--gpu-id", type=int, config_key="gpu_id", help="CUDA device index to use when --device=cuda.")
    _add_configurable_argument(parser, defaults, "--seed", type=int, config_key="seed", help="Random seed.")
    _add_configurable_argument(parser, defaults, "--context-max-length", type=int, config_key="context_max_length", help="Max tokenized context length.")
    _add_configurable_argument(parser, defaults, "--conversation-max-length", type=int, config_key="conversation_max_length", help="Max tokenized conversation length.")
    _add_configurable_argument(parser, defaults, "--max-new-tokens", type=int, config_key="max_new_tokens", help="Maximum number of tokens to generate for the answer.")
    _add_configurable_argument(parser, defaults, "--lora-r", type=int, config_key="lora_r", help="LoRA rank used by the SHINE checkpoint.")
    _add_configurable_argument(parser, defaults, "--metalora-r", type=int, config_key="metalora_r", help="Meta-LoRA rank used by the SHINE checkpoint.")
    _add_configurable_argument(parser, defaults, "--metanetwork-layers", type=int, config_key="metanetwork_layers", help="Number of metanetwork transformer layers.")
    parser.set_defaults(use_system_prompt=bool(defaults.get("use_system_prompt", False)))
    parser.set_defaults(enable_thinking=bool(defaults.get("enable_thinking", False)))
    parser.add_argument("--use-system-prompt", action="store_true", help="Optionally prepend a short system prompt.")
    parser.add_argument("--enable-thinking", action="store_true", help="Allow Qwen thinking output during answer generation.")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    cfg = {
        "name": "case_test",
        "run": {
            "seed": args.seed,
            "use_amp": False,
            "gradient_accumulation_steps": 1,
            "device": args.device,
            "use_gradient_checkpoint": False,
        },
        "paths": {
            "model_path": args.model_path,
        },
        "data": {
            "context_max_length": args.context_max_length,
            "conversation_max_length": args.conversation_max_length,
        },
        "model": {
            "lora_r": args.lora_r,
            "metalora_r": args.metalora_r,
            "ift_additional_metalora_r": -1,
            "num_mem_token": 4,
            "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
            "config_class_path": "LoraQwen.Qwen3Config",
            "tokenizer_from": args.model_path,
            "model_from": args.model_path,
        },
        "metanetwork": {
            "type": "transformer",
            "method": "rl",
            "transformer_cfg": {
                "encoder_cfg": {
                    "d_model": 4096,
                    "nhead": 32,
                    "dim_feedforward": 8192,
                    "dropout": 0,
                    "activation": "gelu",
                    "layer_norm_eps": 0.00001,
                    "batch_first": True,
                    "norm_first": False,
                    "bias": True,
                },
                "couple_encoder_cfg": {
                    "d_model": 4096,
                    "nhead": 32,
                    "dim_feedforward": 8192,
                    "dropout": 0,
                    "activation": "gelu",
                    "layer_norm_eps": 0.00001,
                    "batch_first": True,
                    "norm_first": False,
                    "bias": True,
                },
                "layer_transformer_first": True,
                "mean_pool_size": 1,
                "num_layers": args.metanetwork_layers,
                "couple_num_layers": 0,
                "scale": 0.001,
            },
        },
        "test": {
            "context_max_length": args.context_max_length,
            "conversation_max_length": args.conversation_max_length,
            "max_new_tokens": args.max_new_tokens,
        },
        "hidden_size": -1,
        "num_layers": -1,
        "num_mem_token": 4,
    }
    return OmegaConf.create(cfg)


def extract_think_and_answer(text: str) -> Tuple[str, str]:
    think = ""
    answer = text
    if "<think>" in text:
        parts = text.split("<think>", 1)
        rest = parts[1]
        if "</think>" in rest:
            think, answer = rest.split("</think>", 1)
            think = think.strip()
            answer = answer.strip()
        else:
            think = rest.strip()
            answer = ""
    else:
        answer = text.strip()
    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    if "\n" in answer:
        for line in answer.splitlines():
            if line.strip():
                answer = line.strip()
                break
    return think, answer


def resolve_context(args: argparse.Namespace) -> str:
    if args.load_lora_path:
        return ""
    if bool(args.context) == bool(args.context_file):
        raise ValueError("Provide exactly one of --context or --context-file unless --load-lora-path is used.")
    if args.context_file:
        return Path(args.context_file).read_text(encoding="utf-8").strip()
    return args.context.strip()


def resolve_device(device_name: str, gpu_id: int) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"gpu-id {gpu_id} is out of range. Found {torch.cuda.device_count()} CUDA devices.")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    if device_name == "cuda":
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
    return torch.device("cpu")


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from, padding_side="left", use_fast=True)
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
    tokenizer.chat_template = MINIMAL_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    normalized = str(dtype_name).lower()
    if normalized in {"", "none"}:
        return None
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def load_runtime(cfg, checkpoint_dir: str, device: torch.device):
    set_seed(int(cfg.run.seed))
    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)

    LOGGER.info("Loading config from %s", cfg.model.model_from)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    LOGGER.info("Calculating num_mem_token")
    with torch.device("meta"):
        tmp_model = MetaModelCls(config)
    lora_params = tmp_model.lora_params_numel(cfg.model.lora_r)
    base_params = cfg.hidden_size * cfg.num_layers
    if lora_params % base_params != 0:
        raise ValueError(f"lora_params ({lora_params}) must be divisible by hidden*layers ({base_params})")
    config.num_mem_token = lora_params // base_params
    cfg.num_mem_token = config.num_mem_token
    del tmp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = load_tokenizer(cfg)

    dtype = resolve_torch_dtype(getattr(cfg.model, "torch_dtype", None))
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
        LOGGER.info("Loading main model from %s with torch_dtype=%s", cfg.model.model_from, dtype)
    else:
        LOGGER.info("Loading main model from %s", cfg.model.model_from)
    metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config, **model_kwargs)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.to(device)
    freeze(metamodel)

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    LOGGER.info("Loading checkpoint from %s", checkpoint_dir)
    metanetwork, metalora, _ = load_checkpoint(metanetwork, checkpoint_dir, device)
    metanetwork.eval()
    return metanetwork, metalora, tokenizer


def generate_context_lora(context: str, metanetwork, tokenizer, metalora, cfg, device: torch.device):
    evidence_enc = tokenizer(
        [context],
        max_length=cfg.test.context_max_length,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    )
    evidence_ids = evidence_enc["input_ids"].to(device)
    evidence_attention_mask = evidence_enc["attention_mask"].to(device)
    with torch.no_grad():
        return metanetwork.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)


def save_lora(lora_dict: Any, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_lora = move_lora_to_cpu(lora_dict)
    torch.save(cpu_lora, save_path)
    LOGGER.info("Saved generated LoRA to %s", save_path)


def load_generated_lora(load_path: str, device: torch.device):
    loaded = torch.load(load_path, map_location="cpu", weights_only=False)
    return move_lora_to_device(loaded, device)


def move_lora_to_cpu(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: move_lora_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_lora_to_cpu(v) for v in obj]
    return obj


def move_lora_to_device(obj: Any, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_lora_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_lora_to_device(v, device) for v in obj]
    return obj


def build_initial_messages(use_system_prompt: bool):
    if not use_system_prompt:
        return []
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


def answer_question(
    question: str,
    metanetwork,
    tokenizer,
    lora_dict,
    device: torch.device,
    max_new_tokens: int,
    max_conversation_length: int,
    use_system_prompt: bool,
    enable_thinking: bool,
):
    messages = build_initial_messages(use_system_prompt)
    messages.append({"role": "user", "content": question})
    input_enc = tokenizer.apply_chat_template(
        messages,
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
            loradict=lora_dict,
        )
    new_tokens = outputs[0, input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    think_text, answer_text = extract_think_and_answer(raw_text)
    messages.append({"role": "assistant", "content": answer_text})
    return {
        "question": question,
        "think": think_text,
        "answer": answer_text,
        "raw": raw_text,
        "messages": deepcopy(messages),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    LOGGER.info("Working directory: %s", project_root)

    device = resolve_device(args.device, args.gpu_id)
    cfg = build_cfg(args)
    context = resolve_context(args)

    metanetwork, metalora, tokenizer = load_runtime(cfg, args.checkpoint_dir, device)

    if args.load_lora_path:
        LOGGER.info("Loading generated LoRA from %s", args.load_lora_path)
        lora_dict = load_generated_lora(args.load_lora_path, device)
    else:
        LOGGER.info("Generating LoRA from input context")
        lora_dict = generate_context_lora(context, metanetwork, tokenizer, metalora, cfg, device)
        save_lora(lora_dict, args.save_lora_path)
        lora_dict = load_generated_lora(args.save_lora_path, device)

    result = answer_question(
        question=args.question,
        metanetwork=metanetwork,
        tokenizer=tokenizer,
        lora_dict=lora_dict,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_conversation_length=args.conversation_max_length,
        use_system_prompt=args.use_system_prompt,
        enable_thinking=args.enable_thinking,
    )

    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    if result["think"]:
        print(f"Think: {result['think']}")


if __name__ == "__main__":
    main()
