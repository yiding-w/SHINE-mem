"""
SHINE + local Qwen3 helpers for MemoryAgentBench.

Expects SHINE repo on PYTHONPATH (set SHINE_ROOT or pass shine_root in agent_config).
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import yaml
from transformers import AutoTokenizer


def _ensure_shine_on_path(shine_root: str) -> str:
    shine_root = os.path.abspath(shine_root)
    if shine_root not in sys.path:
        sys.path.insert(0, shine_root)
    return shine_root


def resolve_shine_checkpoint_dir(shine_model_root: str) -> str:
    """
    Accept either:
      - .../checkpoint-epoch-1  (contains metanetwork.pth)
      - .../train/checkpoint-epoch-1
      - .../SHINE-ift_mqa_1qa   (auto-pick latest checkpoint-* under train/ or root)
    """
    shine_model_root = os.path.abspath(shine_model_root)
    if os.path.isfile(os.path.join(shine_model_root, "metanetwork.pth")):
        return shine_model_root

    candidates: List[str] = []
    for pattern in (
        os.path.join(shine_model_root, "train", "checkpoint-*"),
        os.path.join(shine_model_root, "checkpoint-*"),
        os.path.join(shine_model_root, "**", "checkpoint-*"),
    ):
        candidates.extend(glob.glob(pattern, recursive=("**" in pattern)))

    candidates = [c for c in candidates if os.path.isfile(os.path.join(c, "metanetwork.pth"))]
    if not candidates:
        raise FileNotFoundError(
            f"No SHINE checkpoint (metanetwork.pth) under {shine_model_root}. "
            "Point shine_checkpoint_dir to a folder like train/checkpoint-epoch-1."
        )
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_shine_agent_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_SHINE_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_HF_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _shine_cache_key(agent_config: Dict[str, Any], resume_dir: str, cfg_path: str, base_model: str) -> str:
    return "|".join([resume_dir, cfg_path, base_model, str(agent_config.get("shine_root", ""))])


def _resolve_shine_lengths(
    agent_config: Dict[str, Any], dataset_config: Dict[str, Any]
) -> tuple[int, int, int]:
    """Return (evidence_max_len, query_max_len, max_new_tokens)."""
    if agent_config.get("use_mab_context_max_length", False):
        evidence_len = int(dataset_config.get("context_max_length", 131072))
    else:
        evidence_len = int(agent_config.get("shine_context_max_length", 4500))

    if agent_config.get("use_mab_conversation_max_length", False):
        query_len = int(agent_config.get("shine_conversation_max_length", 4096))
    else:
        query_len = int(agent_config.get("shine_conversation_max_length", 300))

    if agent_config.get("use_mab_generation_max_length", True):
        max_new = int(dataset_config.get("generation_max_length", 128))
    else:
        max_new = int(agent_config.get("max_new_tokens", 128))

    return evidence_len, query_len, max_new


class ShineMABRunner:
    """Context -> LoRA once per context; queries use question-only chat (SHINE mode)."""

    def __init__(self, agent_config: Dict[str, Any], dataset_config: Dict[str, Any]):
        from omegaconf import OmegaConf

        shine_root = _ensure_shine_on_path(
            agent_config.get("shine_root") or os.environ.get("SHINE_ROOT", "")
        )
        if not shine_root or not os.path.isdir(shine_root):
            raise ValueError("Set agent_config['shine_root'] or env SHINE_ROOT to SHINE repo path.")

        cfg_path = agent_config.get("shine_cfg_path")
        if not cfg_path:
            raise ValueError("agent_config must include shine_cfg_path (YAML for Qwen3-8B hypernet).")
        cfg_path = os.path.abspath(cfg_path)
        if not os.path.isfile(cfg_path):
            cfg_path = os.path.join(shine_root, cfg_path)
        shine_cfg = OmegaConf.load(cfg_path)

        # Paths
        base_model = agent_config.get("base_model_path") or shine_cfg.paths.model_path
        ckpt_root = agent_config.get("shine_checkpoint_dir") or agent_config.get("shine_model_root")
        if not ckpt_root:
            raise ValueError("Set shine_checkpoint_dir or shine_model_root in agent_config.")
        resume_dir = resolve_shine_checkpoint_dir(ckpt_root)

        device_name = agent_config.get("device", "cuda")
        self.device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if device_name == "cuda" and torch.cuda.is_available() else device_name
        )

        cache_key = _shine_cache_key(agent_config, resume_dir, cfg_path, base_model)
        if cache_key in _SHINE_MODEL_CACHE:
            cached = _SHINE_MODEL_CACHE[cache_key]
            self.tokenizer = cached["tokenizer"]
            self.metanetwork = cached["metanetwork"]
            self.metalora = cached["metalora"]
        else:
            from metanetwork_family import Metanetwork
            from utils.myfreeze import freeze
            from utils.mysaveload import load_checkpoint
            from utils.myinit import _import_class

            MetaModelCls = _import_class(shine_cfg.model.metamodel_class_path)
            ConfigCls = _import_class(shine_cfg.model.config_class_path)
            config = ConfigCls.from_pretrained(base_model)
            config.num_mem_token = -1
            shine_cfg.hidden_size = config.hidden_size
            shine_cfg.num_layers = config.num_hidden_layers

            if shine_cfg.metanetwork.type == "transformer":
                tmp_model = MetaModelCls.from_pretrained(base_model, config=config)
                config.num_mem_token = (
                    tmp_model.lora_params_numel(shine_cfg.model.lora_r)
                    // (shine_cfg.hidden_size * shine_cfg.num_layers)
                )
                shine_cfg.num_mem_token = config.num_mem_token
                del tmp_model
            else:
                config.num_mem_token = shine_cfg.num_mem_token

            self.tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", use_fast=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            metamodel = MetaModelCls.from_pretrained(base_model, config=config)
            metamodel.reset_mem_tokens()
            metanetwork = Metanetwork(metamodel, shine_cfg, metamodel.lora_params_numel(shine_cfg.model.lora_r))
            metanetwork.eval()
            metanetwork.to(self.device)
            freeze(metamodel)

            use_add = bool(shine_cfg.model.get("ift_additional_metalora_r", -1) >= 0)
            metanetwork, metalora, extra = load_checkpoint(
                metanetwork,
                resume_dir,
                str(self.device),
                load_ift_additional_metalora=use_add,
                zero_ift_additional_metalora=(shine_cfg.model.get("ift_additional_metalora_r", -1) == 0),
            )
            if use_add and extra is not None:
                from utils.myloradict import merge_loradicts

                metalora = merge_loradicts(metalora, extra)

            self.metanetwork = metanetwork
            self.metalora = metalora
            _SHINE_MODEL_CACHE[cache_key] = {
                "tokenizer": self.tokenizer,
                "metanetwork": self.metanetwork,
                "metalora": self.metalora,
            }
        self.context_max_length, self.conversation_max_length, self.max_new_tokens = (
            _resolve_shine_lengths(agent_config, dataset_config)
        )
        self.sub_dataset = dataset_config.get("sub_dataset", "")
        print(
            f"[ShineMABRunner] sub_dataset={self.sub_dataset} "
            f"evidence_max_len={self.context_max_length} "
            f"query_max_len={self.conversation_max_length} "
            f"max_new_tokens={self.max_new_tokens}",
            flush=True,
        )
        self.temperature = float(agent_config.get("temperature", 0.0))
        self.memory_time = 0.0

        self._chunks: List[str] = []
        self._loradict = None
        self._evidence_ids = None
        self._evidence_attention_mask = None

    def reset_context(self):
        self._chunks = []
        self._loradict = None
        self._evidence_ids = None
        self._evidence_attention_mask = None

    def memorize_chunk(self, chunk: str, memorize_template: str) -> None:
        formatted = memorize_template.format(
            context=chunk,
            **({"time_stamp": time.strftime("%Y-%m-%d %H:%M:%S")} if "{time_stamp}" in memorize_template else {}),
        )
        self._chunks.append(formatted.strip())
        self._loradict = None

    def _build_loradict(self):
        t0 = time.time()
        evidence = "\n".join(self._chunks).strip()
        enc = self.tokenizer(
            evidence,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        self._evidence_ids = enc["input_ids"].to(self.device)
        self._evidence_attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            self._loradict = self.metanetwork.generate_lora_dict(
                self._evidence_ids,
                self._evidence_attention_mask,
                self.metalora,
            )
        self.memory_time += time.time() - t0

    def query(self, formatted_query: str) -> Dict[str, Any]:
        if self._loradict is None:
            self._build_loradict()

        messages = [{"role": "user", "content": formatted_query}]
        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        input_ids = input_enc["input_ids"].to(self.device)
        attention_mask = input_enc["attention_mask"].to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.metanetwork.metamodel.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loradict=self._loradict,
                ignore_mem_token=True,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        query_time = time.time() - t0

        new_tokens = outputs[0, attention_mask.sum(dim=1).item() :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        input_len = int(attention_mask.sum().item())
        output_len = len(new_tokens)

        return {
            "output": text,
            "input_len": input_len,
            "output_len": output_len,
            "memory_construction_time": self.memory_time,
            "query_time_len": query_time,
        }


class HFLocalLongContextRunner:
    """Qwen3-8B baseline: accumulate memorized chunks in prompt (in-context)."""

    def __init__(self, agent_config: Dict[str, Any], dataset_config: Dict[str, Any]):
        from transformers import AutoModelForCausalLM

        model_path = agent_config["model_path"]
        device_name = agent_config.get("device", "cuda")
        self.device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if device_name == "cuda" and torch.cuda.is_available() else device_name
        )
        dtype = torch.bfloat16 if agent_config.get("bf16", True) else torch.float16

        if model_path in _HF_MODEL_CACHE:
            cached = _HF_MODEL_CACHE[model_path]
            self.tokenizer = cached["tokenizer"]
            self.model = cached["model"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=None,
            ).to(self.device)
            self.model.eval()
            _HF_MODEL_CACHE[model_path] = {"tokenizer": self.tokenizer, "model": self.model}

        self.context_max_length = int(dataset_config.get("context_max_length", 131072))
        self.input_length_limit = int(agent_config.get("input_length_limit", 131072))
        self.buffer_length = int(agent_config.get("buffer_length", 4096))
        if agent_config.get("use_mab_generation_max_length", True):
            self.max_new_tokens = int(dataset_config.get("generation_max_length", 128))
        else:
            self.max_new_tokens = int(agent_config.get("max_new_tokens", 128))
        self.temperature = float(agent_config.get("temperature", 0.0))
        self.memory_time = 0.0
        self.context = ""

    def reset_context(self):
        self.context = ""

    def memorize_chunk(self, chunk: str, memorize_template: str) -> None:
        formatted = memorize_template.format(
            context=chunk,
            **({"time_stamp": time.strftime("%Y-%m-%d %H:%M:%S")} if "{time_stamp}" in memorize_template else {}),
        )
        self.context = (self.context + "\n" + formatted).strip()
        self._truncate_context()

    def _truncate_context(self):
        ids = self.tokenizer.encode(self.context, add_special_tokens=False)
        limit = min(self.context_max_length, self.input_length_limit - self.buffer_length - self.max_new_tokens)
        if len(ids) > limit:
            ids = ids[-limit:]
            self.context = self.tokenizer.decode(ids, skip_special_tokens=True)

    def query(self, formatted_query: str, system_message: str) -> Dict[str, Any]:
        from utils.eval_data_utils import format_chat

        full_user = (self.context + "\n" + formatted_query).strip() if self.context else formatted_query
        messages = format_chat(message=full_user, system_message=system_message)

        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.input_length_limit,
            return_dict=True,
        )
        input_ids = input_enc["input_ids"].to(self.device)
        attention_mask = input_enc["attention_mask"].to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        query_time = time.time() - t0

        new_tokens = outputs[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "output": text,
            "input_len": int(attention_mask.sum().item()),
            "output_len": len(new_tokens),
            "memory_construction_time": 0.0,
            "query_time_len": query_time,
        }
