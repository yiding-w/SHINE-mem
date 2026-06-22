"""
doc-to-lora (D2L, SakanaAI) runner for MemoryAgentBench / δ-mem MAB eval.

Uses ModulatedPretrainedModel: context -> chunked internalize (8192 tok/chunk) ->
merged LoRA; query uses the same formatted prompt as δ-mem base when configured
via ``query_include_context`` in the eval harness (unified context + question).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from methods.d2l_attn_patch import apply_d2l_attn_patch, patch_d2l_state_dict_attn, resolve_d2l_attn

DEFAULT_D2L_CHUNK_LEN = 8192
# LoRA internalize budget — match SHINE shine_context_max_length (8196).
DEFAULT_D2L_EVIDENCE_MAX_TOKENS = 8196

_D2L_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _clip_context_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[... context truncated ...]\n\n"
    if max_chars <= len(marker) + 32:
        return text[-max_chars:]
    head_chars = max(1, (max_chars - len(marker)) // 3)
    tail_chars = max(1, max_chars - len(marker) - head_chars)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def _ensure_d2l_on_path(d2l_root: str) -> str:
    d2l_root = os.path.abspath(d2l_root)
    for p in (d2l_root, os.path.join(d2l_root, "src")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return d2l_root


def _resolve_runner_device(device_name: str) -> torch.device:
    explicit = os.environ.get("D2L_DEVICE", "").strip()
    if explicit:
        return torch.device(explicit)
    if isinstance(device_name, str) and device_name.startswith("cuda:"):
        return torch.device(device_name)
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device(device_name)


def _resolve_ctx_encoder_device(
    base_device: torch.device, agent_config: Dict[str, Any]
) -> torch.device | None:
    raw = agent_config.get("ctx_encoder_device") or os.environ.get("D2L_CTX_ENCODER_DEVICE", "")
    raw = str(raw).strip()
    if not raw:
        return None
    ctx_device = torch.device(raw)
    if ctx_device == base_device:
        return None
    return ctx_device


def _apply_device_layout(model, base_device: torch.device, ctx_device: torch.device | None):
    base_device = torch.device(base_device)
    if ctx_device is None:
        model.to(base_device)
        model.device = base_device
        return base_device, base_device
    ctx_device = torch.device(ctx_device)
    model.base_model.to(base_device)
    model.ctx_encoder.to(ctx_device)
    model.hypernet.to(ctx_device)
    model.device = base_device
    return base_device, ctx_device


def _model_cache_key(checkpoint_path: str, base_device: torch.device, ctx_device: torch.device) -> str:
    return f"{checkpoint_path}|base={base_device}|ctx={ctx_device}"


def _resolve_d2l_lengths(
    agent_config: Dict[str, Any], dataset_config: Dict[str, Any]
) -> tuple[int, int]:
    chunk_len = int(agent_config.get("d2l_chunk_len") or DEFAULT_D2L_CHUNK_LEN)
    if agent_config.get("use_mab_generation_max_length", True):
        max_new = int(dataset_config.get("generation_max_length", 128))
    else:
        max_new = int(agent_config.get("max_new_tokens", 128))
    return chunk_len, max_new


class DocToLoraRunner:
    """Context -> merged LoRA once per row; subsequent queries reuse internalized LoRA."""

    def __init__(self, agent_config: Dict[str, Any], dataset_config: Dict[str, Any]):
        d2l_root = agent_config.get("d2l_root") or os.environ.get("D2L_ROOT", "")
        if not d2l_root:
            mab_root = Path(__file__).resolve().parents[1]
            d2l_root = str((mab_root.parent.parent / "doc-to-lora").resolve())
        else:
            path = Path(d2l_root)
            if not path.is_absolute():
                mab_root = Path(__file__).resolve().parents[1]
                d2l_root = str((mab_root / path).resolve())
            else:
                d2l_root = str(path.resolve())
        d2l_root = _ensure_d2l_on_path(d2l_root)
        if not os.path.isdir(d2l_root):
            raise ValueError(f"doc-to-lora repo not found: {d2l_root}")

        checkpoint_path = agent_config.get("d2l_checkpoint_path")
        if not checkpoint_path:
            raise ValueError("agent_config must include d2l_checkpoint_path (D2L pytorch_model.bin).")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            mab_root = Path(__file__).resolve().parents[1]
            checkpoint_path = (mab_root / checkpoint_path).resolve()
        else:
            checkpoint_path = checkpoint_path.resolve()
        checkpoint_path = str(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"D2L checkpoint not found: {checkpoint_path}")

        self.device = _resolve_runner_device(agent_config.get("device", "cuda"))
        ctx_encoder_device = _resolve_ctx_encoder_device(self.device, agent_config)
        cache_key = _model_cache_key(checkpoint_path, self.device, ctx_encoder_device or self.device)

        if cache_key in _D2L_MODEL_CACHE:
            cached = _D2L_MODEL_CACHE[cache_key]
            self.model = cached["model"]
            self.tokenizer = cached["tokenizer"]
            self.ctx_tokenizer = cached["ctx_tokenizer"]
            self.ctx_device = cached["ctx_device"]
        else:
            from ctx_to_lora.model_loading import get_tokenizer
            from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

            use_flash_attn, attn_impl = resolve_d2l_attn(agent_config)
            if not use_flash_attn:
                # Idefics2 hardcodes flash_attention_2; without flash_attn use eager (not sdpa).
                fallback = "eager" if attn_impl == "sdpa" else attn_impl
                apply_d2l_attn_patch(fallback)
            else:
                os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "flash_attention_2"

            state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            if not use_flash_attn:
                patch_d2l_state_dict_attn(state_dict, "eager")
            self.model = ModulatedPretrainedModel.from_state_dict(
                state_dict,
                train=False,
                use_sequence_packing=False,
                use_flash_attn=use_flash_attn,
                base_model_kwargs={"attn_implementation": attn_impl},
            )
            self.device, self.ctx_device = _apply_device_layout(
                self.model, self.device, ctx_encoder_device
            )
            self.model.reset()
            self.model.enable_iterative_mode(True)

            self.tokenizer = get_tokenizer(self.model.base_model.name_or_path)
            self.ctx_tokenizer = get_tokenizer(self.model.ctx_encoder.base_model.name_or_path)

            _D2L_MODEL_CACHE[cache_key] = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "ctx_tokenizer": self.ctx_tokenizer,
                "ctx_device": self.ctx_device,
            }

        self.device = getattr(self.model, "device", self.device)
        self.ctx_device = getattr(self, "ctx_device", self.device)
        self.chunk_len, self.max_new_tokens = _resolve_d2l_lengths(agent_config, dataset_config)
        self.sub_dataset = dataset_config.get("sub_dataset", "")
        self.agent_config = agent_config
        self.temperature = float(agent_config.get("temperature", 0.0))
        self.memory_time = 0.0
        self._max_context_chars = 0
        self._evidence_max_tokens = DEFAULT_D2L_EVIDENCE_MAX_TOKENS
        self._query_max_length = 0

        _use_flash, _attn_impl = resolve_d2l_attn(agent_config)
        device_msg = f"base={self.device} ctx={self.ctx_device}"
        print(
            f"[DocToLoraRunner] sub_dataset={self.sub_dataset} "
            f"chunk_len={self.chunk_len} max_new_tokens={self.max_new_tokens} "
            f"evidence_max_tokens={self._evidence_max_tokens} "
            f"attn={_attn_impl} flash={_use_flash} {device_msg} "
            f"checkpoint={os.path.basename(checkpoint_path)}",
            flush=True,
        )

        self._chunks: List[str] = []
        self._internalized = False
        self._n_ctx_chunks = None

    def configure_for_row(
        self,
        *,
        model_context_window: int = 0,
        max_new_tokens: int = 0,
        max_context_chars: int = 0,
    ) -> None:
        """Cap internalize evidence tokens (like SHINE generate_lora_dict), not query prompt."""
        self._max_context_chars = int(max_context_chars) if max_context_chars > 0 else 0
        buffer = 512
        auto_cap = max(512, int(model_context_window or 32768) - int(max_new_tokens or 128) - buffer)
        if self._max_context_chars > 0:
            auto_cap = min(auto_cap, max(512, self._max_context_chars // 3))
        explicit = int(self.agent_config.get("d2l_context_max_length") or 0)
        budget = explicit if explicit > 0 else DEFAULT_D2L_EVIDENCE_MAX_TOKENS
        self._evidence_max_tokens = min(budget, auto_cap)
        print(
            f"[DocToLoraRunner] sub_dataset={self.sub_dataset} "
            f"evidence_max_tokens={self._evidence_max_tokens} "
            f"(budget={budget}, auto_cap={auto_cap})",
            flush=True,
        )

    def set_query_max_length(self, max_length: int) -> None:
        self._query_max_length = max(512, int(max_length))

    def reset_context(self):
        self._chunks = []
        self._internalized = False
        self._n_ctx_chunks = None
        self.model.reset()

    def memorize_chunk(self, chunk: str, memorize_template: str) -> None:
        formatted = memorize_template.format(
            context=chunk,
            **({"time_stamp": time.strftime("%Y-%m-%d %H:%M:%S")} if "{time_stamp}" in memorize_template else {}),
        )
        self._chunks.append(formatted.strip())
        self._internalized = False

    def _internalize_ctx_chunks(self, ctx_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Generate one LoRA per 8192-token chunk and STACK them (lower peak VRAM).

        IMPORTANT: store the RAW per-chunk LoRAs (leading dim = n_chunks) and let
        ``model.generate()`` run ``combine_lora`` exactly once via ``n_ctx_chunks``.
        Do NOT pre-combine here — ``generate()`` always combines, so pre-combining
        double-combines (rank explosion / split mismatch for multi-chunk contexts).
        This matches D2L's canonical ``_internalize_from_ids`` -> ``generate`` path.
        """
        ctx_device = getattr(self, "ctx_device", self.device)
        chunk_loras: List[Dict[str, Any]] = []

        with torch.no_grad():
            self.model.reset()
            self.model.patch_lora_forward()
            for chunk in ctx_tensors:
                chunk_ids = chunk.unsqueeze(0).to(ctx_device)
                chunk_mask = torch.ones_like(chunk_ids)
                loras, _ = self.model.generate_weights(chunk_ids, chunk_mask)
                chunk_loras.append(loras)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if len(chunk_loras) == 1:
                merged = chunk_loras[0]
            else:
                merged = {}
                for module in chunk_loras[0]:
                    merged[module] = {
                        "A": torch.cat([entry[module]["A"] for entry in chunk_loras], dim=0),
                        "B": torch.cat([entry[module]["B"] for entry in chunk_loras], dim=0),
                    }

            # Move to the base-model device so generate()'s combine_lora + apply land there.
            if ctx_device != self.device:
                merged = {
                    module: {k: v.to(self.device) for k, v in mats.items()}
                    for module, mats in merged.items()
                }

            self.model.generated_loras = merged  # uncombined; generate() combines once

        return torch.tensor([len(ctx_tensors)], dtype=torch.int32, device=self.device)

    def _build_lora(self):
        from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text

        t0 = time.time()
        evidence = "\n".join(self._chunks).strip()
        if self._max_context_chars > 0 and len(evidence) > self._max_context_chars:
            evidence = _clip_context_chars(evidence, self._max_context_chars)

        tokenized = tokenize_ctx_text({"context": [evidence]}, self.ctx_tokenizer)
        ctx_ids_flat = tokenized["ctx_ids"]
        if ctx_ids_flat and isinstance(ctx_ids_flat[0], list):
            ctx_ids_flat = ctx_ids_flat[0]

        if self._evidence_max_tokens > 0 and len(ctx_ids_flat) > self._evidence_max_tokens:
            ctx_ids_flat = ctx_ids_flat[-self._evidence_max_tokens :]

        model_name = self.model.base_model.name_or_path
        split_result = split_too_long_ctx(
            {"ctx_ids": ctx_ids_flat},
            model_name,
            num_chunk_probs=None,
            max_chunk_len=self.chunk_len,
            min_chunk_len=25,
            max_num_split=None,
            is_train=False,
        )
        ctx_ids_list = split_result["ctx_ids"]
        ctx_tensors = [torch.tensor(x, dtype=torch.long) for x in ctx_ids_list]
        self._n_ctx_chunks = self._internalize_ctx_chunks(ctx_tensors)
        self._internalized = True
        self.memory_time += time.time() - t0
        print(
            f"[DocToLoraRunner] internalized {len(ctx_tensors)} chunk(s) "
            f"from {len(ctx_ids_flat)} context tokens "
            f"(evidence_cap={self._evidence_max_tokens}, chunk_len={self.chunk_len})",
            flush=True,
        )

    def query(self, formatted_query: str) -> Dict[str, Any]:
        if not self._internalized:
            self._build_lora()

        messages = [{"role": "user", "content": formatted_query}]
        encode_kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
            "padding": False,
            "enable_thinking": False,
        }
        if self._query_max_length > 0:
            encode_kwargs["max_length"] = self._query_max_length
            encode_kwargs["truncation"] = True
        input_enc = self.tokenizer.apply_chat_template(messages, **encode_kwargs)
        chat_ids = input_enc["input_ids"].to(self.device)
        attention_mask = input_enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones_like(chat_ids)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            top_p = self.agent_config.get("top_p")
            top_k = self.agent_config.get("top_k")
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=chat_ids,
                n_ctx_chunks=self._n_ctx_chunks,
                **gen_kwargs,
            )
        query_time = time.time() - t0

        input_len = int(attention_mask.sum().item())
        new_tokens = outputs[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "output": text,
            "input_len": input_len,
            "output_len": int(len(new_tokens)),
            "memory_construction_time": self.memory_time,
            "query_time_len": query_time,
        }
