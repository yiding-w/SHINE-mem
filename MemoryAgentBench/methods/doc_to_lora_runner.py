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

DEFAULT_D2L_CHUNK_LEN = 8192

_D2L_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _ensure_d2l_on_path(d2l_root: str) -> str:
    d2l_root = os.path.abspath(d2l_root)
    for p in (d2l_root, os.path.join(d2l_root, "src")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return d2l_root


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

        device_name = agent_config.get("device", "cuda")
        self.device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if device_name == "cuda" and torch.cuda.is_available()
            else device_name
        )

        if checkpoint_path in _D2L_MODEL_CACHE:
            cached = _D2L_MODEL_CACHE[checkpoint_path]
            self.model = cached["model"]
            self.tokenizer = cached["tokenizer"]
            self.ctx_tokenizer = cached["ctx_tokenizer"]
        else:
            from ctx_to_lora.model_loading import get_tokenizer
            from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

            state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            attn_impl = (
                agent_config.get("attn_implementation")
                or os.environ.get("D2L_ATTN_IMPLEMENTATION")
                or os.environ.get("ATTN_IMPLEMENTATION")
                or "sdpa"
            )
            use_flash_attn = bool(agent_config.get("use_flash_attn", False))
            self.model = ModulatedPretrainedModel.from_state_dict(
                state_dict,
                train=False,
                use_sequence_packing=False,
                use_flash_attn=use_flash_attn,
                base_model_kwargs={"attn_implementation": attn_impl},
            )
            try:
                self.model.to(self.device)
            except Exception:
                pass
            self.model.reset()
            self.model.enable_iterative_mode(True)

            self.tokenizer = get_tokenizer(self.model.base_model.name_or_path)
            self.ctx_tokenizer = get_tokenizer(self.model.ctx_encoder.base_model.name_or_path)

            _D2L_MODEL_CACHE[checkpoint_path] = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "ctx_tokenizer": self.ctx_tokenizer,
            }

        self.device = getattr(self.model, "device", self.device)
        self.chunk_len, self.max_new_tokens = _resolve_d2l_lengths(agent_config, dataset_config)
        self.sub_dataset = dataset_config.get("sub_dataset", "")
        self.agent_config = agent_config
        self.temperature = float(agent_config.get("temperature", 0.0))
        self.memory_time = 0.0
        self._max_context_chars = 0
        self._query_max_length = 0

        print(
            f"[DocToLoraRunner] sub_dataset={self.sub_dataset} "
            f"chunk_len={self.chunk_len} max_new_tokens={self.max_new_tokens} "
            f"attn={agent_config.get('attn_implementation', os.environ.get('D2L_ATTN_IMPLEMENTATION', 'sdpa'))} "
            f"checkpoint={os.path.basename(checkpoint_path)}",
            flush=True,
        )

        self._chunks: List[str] = []
        self._internalized = False
        self._n_ctx_chunks = None

    def configure_for_row(self, *, max_context_chars: int = 0) -> None:
        self._max_context_chars = int(max_context_chars) if max_context_chars > 0 else 0

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

    def _build_lora(self):
        from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text

        t0 = time.time()
        evidence = "\n".join(self._chunks).strip()
        if self._max_context_chars > 0 and len(evidence) > self._max_context_chars:
            evidence = evidence[-self._max_context_chars :]

        tokenized = tokenize_ctx_text({"context": [evidence]}, self.ctx_tokenizer)
        ctx_ids_flat = tokenized["ctx_ids"]
        if ctx_ids_flat and isinstance(ctx_ids_flat[0], list):
            ctx_ids_flat = ctx_ids_flat[0]

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
        ctx_ids = torch.nn.utils.rnn.pad_sequence(
            ctx_tensors, batch_first=True, padding_value=self.ctx_tokenizer.pad_token_id or 0
        ).to(self.device)
        ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(x) for x in ctx_tensors], batch_first=True, padding_value=0
        ).to(self.device)

        with torch.no_grad():
            self.model.reset()
            self.model._internalize_from_ids(ctx_ids, ctx_attn_mask)

        self._n_ctx_chunks = torch.tensor([len(ctx_tensors)], dtype=torch.int32, device=self.device)
        self._internalized = True
        self.memory_time += time.time() - t0
        print(
            f"[DocToLoraRunner] internalized {len(ctx_tensors)} chunk(s) "
            f"from {len(ctx_ids_flat)} context tokens (chunk_len={self.chunk_len})",
            flush=True,
        )

    def query(self, formatted_query: str) -> Dict[str, Any]:
        if not self._internalized:
            self._build_lora()

        messages = [{"role": "user", "content": formatted_query}]
        chat_ids = self.tokenizer.apply_chat_template(
            messages,
            add_special_tokens=False,
            add_generation_prompt=True,
            return_attention_mask=False,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)
        if self._query_max_length > 0 and chat_ids.shape[1] > self._query_max_length:
            chat_ids = chat_ids[:, -self._query_max_length :]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
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

        input_len = int(chat_ids.shape[1])
        new_tokens = outputs[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "output": text,
            "input_len": input_len,
            "output_len": int(len(new_tokens)),
            "memory_construction_time": self.memory_time,
            "query_time_len": query_time,
        }
