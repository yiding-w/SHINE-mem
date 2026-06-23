# LoRA-adapted Qwen3_5 model
# Based on Qwen3_5.py with LoRA modifications following the pattern of LoraQwen3_5Moe.py

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from utils.myparallel import is_main_process_per_node

logger = logging.getLogger(__name__)

# transformers 5.x build skew: inject DynamicCache into qwen3_5 namespace
# if missing — see src_transformers/Qwen3_5.py for the same patch.
import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen3_5_mod
if not hasattr(_qwen3_5_mod, "DynamicCache"):
    from transformers.cache_utils import DynamicCache as _DynamicCache
    _qwen3_5_mod.DynamicCache = _DynamicCache

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    init, ACT2FN, Cache, DynamicCache, GenerationMixin,
    use_kernelized_func,
    create_causal_mask,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    BaseModelOutputWithPast, CausalLMOutputWithPast,
    ROPE_INIT_FUNCTIONS, dynamic_rope_update,
    ALL_ATTENTION_FUNCTIONS, PreTrainedModel,
    Unpack,
    TransformersKwargs, auto_docstring, can_return_tuple,
    maybe_autocast, merge_with_config_defaults,
    capture_outputs,
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
    Qwen3_5RMSNorm, Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5MLP,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    Qwen3_5ModelOutputWithPast,
    rotate_half, apply_rotary_pos_emb, repeat_kv, eager_attention_forward,
    apply_mask_to_padding_states,
)

from math import sqrt
from torch import Tensor


@dataclass
class MemoryQwen3_5ModelOutputWithPast(BaseModelOutputWithPast):
    """Extends BaseModelOutputWithPast with memory states."""
    memory_states: Optional[torch.Tensor] = None  # (batch_size, num_layer, num_mem_token, hidden_size)


@dataclass
class MemoryCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Extends CausalLMOutputWithPast with memory states."""
    memory_states: Optional[torch.Tensor] = None  # (batch_size, num_layer, num_mem_token, hidden_size)


class LoraLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, input: Tensor, loradict=None, nograd_loradict=None, nograd_wdict=None) -> Tensor:
        base = F.linear(input, self.weight, self.bias)
        if loradict is None and nograd_loradict is None and nograd_wdict is None:
            return base

        result = base
        if loradict is not None:
            result = result + self._compute_lora_delta(input, loradict)

        # If both nograd_loradict and nograd_wdict exist, merge nograd_loradict's A@B into W for a single matmul
        if nograd_loradict is not None and nograd_wdict is not None:
            result = result + self._compute_merged_w_delta(input, nograd_loradict, nograd_wdict)
        elif nograd_loradict is not None:
            result = result + self._compute_lora_delta(input, nograd_loradict)
        elif nograd_wdict is not None:
            result = result + self._compute_w_delta(input, nograd_wdict)
        return result

    def _compute_lora_delta(self, input: Tensor, loradict: dict) -> Tensor:
        """Compute the LoRA delta for a single loradict."""
        A = loradict["A"]              # [Lb, in, r]
        B = loradict["B"]              # [Lb, r, out]
        C = loradict.get("C", None)    # [Lb, out] or None

        Lb = A.shape[0]
        torch._assert(input.shape[-1] == self.in_features, "last dim must be in_features")
        torch._assert(input.shape[0] % Lb == 0, "input batch must be multiple of lora batch")
        num_beams = input.shape[0] // Lb

        # Flatten all middle dims (e.g., seq_len) into S for faster matmul
        # input: [B, ..., in] -> x: [Lb, beams, S, in]
        x = input.reshape(Lb, num_beams, -1, self.in_features)

        # [Lb, beams, S, in] @ [Lb, in, r] -> [Lb, beams, S, r]
        tmp = torch.matmul(x, A[:, None, :, :])
        # [Lb, beams, S, r] @ [Lb, r, out] -> [Lb, beams, S, out]
        lora_out = torch.matmul(tmp, B[:, None, :, :])

        if self.bias is None:
            torch._assert(C is None, "If bias is None, loradict['C'] must also be None")
        else:
            torch._assert(C is not None, "If bias is not None, loradict['C'] must also be not None")
            # C: [Lb, out] -> [Lb, 1, 1, out] broadcast across beams and S
            lora_out = lora_out + C[:, None, None, :]

        # Restore original middle dims: [Lb*beams, ..., out]
        lora_out = lora_out.reshape(*input.shape[:-1], self.out_features)
        return lora_out

    def _compute_w_delta(self, input: Tensor, wdict: dict) -> Tensor:
        """Compute the delta for a full-rank weight matrix W: {"W": [Lb, in, out], "C": [Lb, out] | None}."""
        W = wdict["W"]                 # [Lb, in, out]
        C = wdict.get("C", None)       # [Lb, out] or None

        Lb = W.shape[0]
        torch._assert(input.shape[-1] == self.in_features, "last dim must be in_features")
        torch._assert(input.shape[0] % Lb == 0, "input batch must be multiple of lora batch")
        num_beams = input.shape[0] // Lb

        x = input.reshape(Lb, num_beams, -1, self.in_features)

        # [Lb, beams, S, in] @ [Lb, 1, in, out] -> [Lb, beams, S, out]
        w_out = torch.matmul(x, W[:, None, :, :])

        if self.bias is None:
            torch._assert(C is None, "If bias is None, wdict['C'] must also be None")
        else:
            torch._assert(C is not None, "If bias is not None, wdict['C'] must also be not None")
            w_out = w_out + C[:, None, None, :]

        w_out = w_out.reshape(*input.shape[:-1], self.out_features)
        return w_out

    def _compute_merged_w_delta(self, input: Tensor, nograd_loradict: dict, nograd_wdict: dict) -> Tensor:
        """Merge nograd_loradict's A@B into nograd_wdict's W, then compute x @ merged_W."""
        A = nograd_loradict["A"]       # [Lb, in, r]
        B = nograd_loradict["B"]       # [Lb, r, out]
        C_lora = nograd_loradict.get("C", None)  # [Lb, out] or None

        W = nograd_wdict["W"]          # [Lb, in, out]
        C_w = nograd_wdict.get("C", None)  # [Lb, out] or None

        # Merge: W_merged = W + A @ B
        W_merged = W + torch.bmm(A, B)  # [Lb, in, out]

        # Merge bias: C_merged = C_w + C_lora (if applicable)
        if self.bias is None:
            C_merged = None
        else:
            C_merged = C_w
            if C_lora is not None:
                C_merged = C_merged + C_lora

        merged_wdict = {"W": W_merged, "C": C_merged}
        return self._compute_w_delta(input, merged_wdict)

    def lora_delta(self, input: Tensor, loradict: dict) -> Tensor:
        """Compute only the LoRA delta (without the base linear output)."""
        return self._compute_lora_delta(input, loradict)

    def lora_params_numel(self, r):
        if r == 0:
            return 0
        if not hasattr(self, "_lora_numel_cache"):
            self._lora_numel_cache = {}
        if r not in self._lora_numel_cache:
            self._lora_numel_cache[r] = (
                self.in_features * r + self.out_features * r
                + (self.out_features if self.bias is not None else 0)
            )
        return self._lora_numel_cache[r]

    def set_generate_func(self, method):
        if method == "rl":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, self.in_features, r) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, r, self.out_features) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        elif method == "rr":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, self.in_features, r) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, self.out_features, r).transpose(-1, -2) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        elif method == "lr":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, r, self.in_features).transpose(-1, -2) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, self.out_features, r).transpose(-1, -2) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        elif method == "ll":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, r, self.in_features).transpose(-1, -2) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, r, self.out_features) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        else:
            raise NotImplementedError(f"LoRA method {method} not implemented")

        self.generate_func = generate_func

    def generate_lora_dict(self, r, scale, plain_tensor):
        if r == 0:
            return None
        torch._assert(
            plain_tensor.shape[-1] == self.lora_params_numel(r),
            "plain_tensor last dim does not match lora_params_numel"
        )
        torch._assert(hasattr(self, "generate_func"), "generate_func not set")
        return self.generate_func(r, scale, plain_tensor)

    def init_lora_dict(self, r, scale, device, dtype):
        if r == 0:
            return None
        torch._assert(r > 0, "r must be positive")
        A = (torch.randn(size=(1, self.in_features, r), device=device, dtype=dtype) * sqrt(scale)).detach()
        A.requires_grad_()
        B = torch.zeros(size=(1, r, self.out_features), requires_grad=True, device=device, dtype=dtype)
        C = torch.zeros(size=(1, self.out_features), requires_grad=True, device=device, dtype=dtype) if self.bias is not None else None
        return {"A": A, "B": B, "C": C}


class LoraHelper:
    """Lightweight helper that mirrors LoraLinear's LoRA parameter management
    (lora_params_numel, set_generate_func, generate_lora_dict, init_lora_dict, lora_delta)
    but holds NO nn.Parameter / weight / bias — zero extra memory."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

    # ---- LoRA delta (same logic as LoraLinear.lora_delta) ----
    def lora_delta(self, input: Tensor, loradict: dict) -> Tensor:
        A = loradict["A"]              # [Lb, in, r]
        B = loradict["B"]              # [Lb, r, out]
        C = loradict.get("C", None)    # [Lb, out] or None

        Lb = A.shape[0]
        torch._assert(input.shape[-1] == self.in_features, "last dim must be in_features")
        torch._assert(input.shape[0] % Lb == 0, "input batch must be multiple of lora batch")
        num_beams = input.shape[0] // Lb

        x = input.reshape(Lb, num_beams, -1, self.in_features)
        tmp = torch.matmul(x, A[:, None, :, :])
        lora_out = torch.matmul(tmp, B[:, None, :, :])

        if C is not None:
            lora_out = lora_out + C[:, None, None, :]

        lora_out = lora_out.reshape(*input.shape[:-1], self.out_features)
        return lora_out

    # ---- parameter counting ----
    def lora_params_numel(self, r):
        if r == 0:
            return 0
        if not hasattr(self, "_lora_numel_cache"):
            self._lora_numel_cache = {}
        if r not in self._lora_numel_cache:
            self._lora_numel_cache[r] = (
                self.in_features * r + self.out_features * r
                + (self.out_features if self.has_bias else 0)
            )
        return self._lora_numel_cache[r]

    # ---- generate_func / generate_lora_dict / init_lora_dict ----
    def set_generate_func(self, method):
        in_f, out_f, has_bias = self.in_features, self.out_features, self.has_bias
        if method == "rl":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                plain_A = plain_tensor[:, idx: idx + in_f * r].view(-1, in_f, r)
                A = plain_A * sqrt(scale)
                idx += in_f * r
                plain_B = plain_tensor[:, idx: idx + out_f * r].view(-1, r, out_f)
                B = plain_B * sqrt(scale)
                idx += out_f * r
                C = plain_tensor[:, idx: idx + out_f].view(-1, out_f) * scale if has_bias else None
                return {"A": A, "B": B, "C": C}
        elif method == "rr":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + in_f * r].view(-1, in_f, r) * sqrt(scale)
                idx += in_f * r
                B = plain_tensor[:, idx: idx + out_f * r].view(-1, out_f, r).transpose(-1, -2) * sqrt(scale)
                idx += out_f * r
                C = plain_tensor[:, idx: idx + out_f].view(-1, out_f) * scale if has_bias else None
                return {"A": A, "B": B, "C": C}
        elif method == "lr":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + in_f * r].view(-1, r, in_f).transpose(-1, -2) * sqrt(scale)
                idx += in_f * r
                B = plain_tensor[:, idx: idx + out_f * r].view(-1, out_f, r).transpose(-1, -2) * sqrt(scale)
                idx += out_f * r
                C = plain_tensor[:, idx: idx + out_f].view(-1, out_f) * scale if has_bias else None
                return {"A": A, "B": B, "C": C}
        elif method == "ll":
            def generate_func(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + in_f * r].view(-1, r, in_f).transpose(-1, -2) * sqrt(scale)
                idx += in_f * r
                B = plain_tensor[:, idx: idx + out_f * r].view(-1, r, out_f) * sqrt(scale)
                idx += out_f * r
                C = plain_tensor[:, idx: idx + out_f].view(-1, out_f) * scale if has_bias else None
                return {"A": A, "B": B, "C": C}
        else:
            raise NotImplementedError(f"LoRA method {method} not implemented")
        self.generate_func = generate_func

    def generate_lora_dict(self, r, scale, plain_tensor):
        if r == 0:
            return None
        torch._assert(
            plain_tensor.shape[-1] == self.lora_params_numel(r),
            "plain_tensor last dim does not match lora_params_numel"
        )
        torch._assert(hasattr(self, "generate_func"), "generate_func not set")
        return self.generate_func(r, scale, plain_tensor)

    def init_lora_dict(self, r, scale, device, dtype):
        if r == 0:
            return None
        torch._assert(r > 0, "r must be positive")
        A = (torch.randn(size=(1, self.in_features, r), device=device, dtype=dtype) * sqrt(scale)).detach()
        A.requires_grad_()
        B = torch.zeros(size=(1, r, self.out_features), requires_grad=True, device=device, dtype=dtype)
        C = torch.zeros(size=(1, self.out_features), requires_grad=True, device=device, dtype=dtype) if self.has_bias else None
        return {"A": A, "B": B, "C": C}


class LoraQwen3_5MLP(Qwen3_5MLP):
    def __init__(self, config: Qwen3_5Config, intermediate_size: int):
        super().__init__(config, intermediate_size)
        self.gate_proj = LoraLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = LoraLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = LoraLinear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x, loradict=None, nograd_loradict=None, nograd_wdict=None):
        if loradict is None and nograd_loradict is None and nograd_wdict is None:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            gate_lora = loradict.get("gate") if loradict else None
            gate_nograd = nograd_loradict.get("gate") if nograd_loradict else None
            gate_w = nograd_wdict.get("gate") if nograd_wdict else None
            up_lora = loradict.get("up") if loradict else None
            up_nograd = nograd_loradict.get("up") if nograd_loradict else None
            up_w = nograd_wdict.get("up") if nograd_wdict else None
            down_lora = loradict.get("down") if loradict else None
            down_nograd = nograd_loradict.get("down") if nograd_loradict else None
            down_w = nograd_wdict.get("down") if nograd_wdict else None
            down_proj = self.down_proj(
                self.act_fn(self.gate_proj(x, gate_lora, gate_nograd, gate_w)) * self.up_proj(x, up_lora, up_nograd, up_w),
                down_lora, down_nograd, down_w
            )
        return down_proj

    def lora_params_numel(self, lora_ranks):
        """lora_ranks: dict with keys 'mlp_gate', 'mlp_up', 'mlp_down'."""
        cache_key = (lora_ranks["mlp_gate"], lora_ranks["mlp_up"], lora_ranks["mlp_down"])
        if not hasattr(self, "_lora_numel_cache"):
            self._lora_numel_cache = {}
        if cache_key not in self._lora_numel_cache:
            self._lora_numel_cache[cache_key] = (
                self.gate_proj.lora_params_numel(lora_ranks["mlp_gate"])
                + self.up_proj.lora_params_numel(lora_ranks["mlp_up"])
                + self.down_proj.lora_params_numel(lora_ranks["mlp_down"])
            )
        return self._lora_numel_cache[cache_key]

    def set_generate_func(self, method):
        self.gate_proj.set_generate_func(method)
        self.up_proj.set_generate_func(method)
        self.down_proj.set_generate_func(method)

    def generate_lora_dict(self, lora_ranks, scale, plain_tensor):
        torch._assert(plain_tensor.shape[-1] == self.lora_params_numel(lora_ranks), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match lora_params_numel {self.lora_params_numel(lora_ranks)}")
        idx = 0
        gate = self.gate_proj.generate_lora_dict(lora_ranks["mlp_gate"], scale, plain_tensor[:, idx: idx + self.gate_proj.lora_params_numel(lora_ranks["mlp_gate"])])
        idx += self.gate_proj.lora_params_numel(lora_ranks["mlp_gate"])
        up = self.up_proj.generate_lora_dict(lora_ranks["mlp_up"], scale, plain_tensor[:, idx: idx + self.up_proj.lora_params_numel(lora_ranks["mlp_up"])])
        idx += self.up_proj.lora_params_numel(lora_ranks["mlp_up"])
        down = self.down_proj.generate_lora_dict(lora_ranks["mlp_down"], scale, plain_tensor[:, idx: idx + self.down_proj.lora_params_numel(lora_ranks["mlp_down"])])
        return {"gate": gate, "up": up, "down": down}

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        gate = self.gate_proj.init_lora_dict(lora_ranks["mlp_gate"], scale, device, dtype)
        up = self.up_proj.init_lora_dict(lora_ranks["mlp_up"], scale, device, dtype)
        down = self.down_proj.init_lora_dict(lora_ranks["mlp_down"], scale, device, dtype)
        return {"gate": gate, "up": up, "down": down}


class LoraQwen3_5Attention(Qwen3_5Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper with LoRA support"""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # q_proj is a merged linear for query + gate (num_attention_heads * head_dim * 2).
        # We replace it with LoraLinear but do NOT pass loradict through its forward;
        # instead we apply two independent LoRA deltas for the query and gate halves.
        q_single_dim = config.num_attention_heads * self.head_dim
        self.q_proj = LoraLinear(
            config.hidden_size, q_single_dim * 2, bias=config.attention_bias
        )
        # Lightweight helpers for independent LoRA on query / gate halves.
        # They carry NO nn.Parameter — only LoRA management logic.
        self.q_query_lora = LoraHelper(
            config.hidden_size, q_single_dim, bias=config.attention_bias
        )
        self.q_gate_lora = LoraHelper(
            config.hidden_size, q_single_dim, bias=config.attention_bias
        )
        self.k_proj = LoraLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = LoraLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = LoraLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        loradict=None,
        nograd_loradict=None,
        nograd_wdict=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if loradict is None and nograd_loradict is None and nograd_wdict is None:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        else:
            # Base forward through the original merged q_proj
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            # query_states: [B, S, num_heads, head_dim]
            # gate:         [B, S, num_heads, head_dim]

            # Add independent LoRA deltas for query and gate (skip if rank=0 / None)
            # From loradict
            if loradict is not None and loradict.get("q_query") is not None:
                query_states = query_states + self.q_query_lora.lora_delta(hidden_states, loradict["q_query"]).view(*input_shape, -1, self.head_dim)
            if loradict is not None and loradict.get("q_gate") is not None:
                gate = gate + self.q_gate_lora.lora_delta(hidden_states, loradict["q_gate"]).view(*input_shape, -1, self.head_dim)
            # From nograd_loradict
            if nograd_loradict is not None and nograd_loradict.get("q_query") is not None:
                query_states = query_states + self.q_query_lora.lora_delta(hidden_states, nograd_loradict["q_query"]).view(*input_shape, -1, self.head_dim)
            if nograd_loradict is not None and nograd_loradict.get("q_gate") is not None:
                gate = gate + self.q_gate_lora.lora_delta(hidden_states, nograd_loradict["q_gate"]).view(*input_shape, -1, self.head_dim)
            # From nograd_wdict (full-rank W for q_query and q_gate)
            if nograd_wdict is not None and nograd_wdict.get("q_query") is not None:
                query_states = query_states + self._compute_helper_w_delta(hidden_states, nograd_wdict["q_query"], self.q_query_lora).view(*input_shape, -1, self.head_dim)
            if nograd_wdict is not None and nograd_wdict.get("q_gate") is not None:
                gate = gate + self._compute_helper_w_delta(hidden_states, nograd_wdict["q_gate"], self.q_gate_lora).view(*input_shape, -1, self.head_dim)
            gate = gate.reshape(*input_shape, -1)

            k_lora = loradict.get("k") if loradict else None
            k_nograd = nograd_loradict.get("k") if nograd_loradict else None
            k_w = nograd_wdict.get("k") if nograd_wdict else None
            v_lora = loradict.get("v") if loradict else None
            v_nograd = nograd_loradict.get("v") if nograd_loradict else None
            v_w = nograd_wdict.get("v") if nograd_wdict else None

            query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states, k_lora, k_nograd, k_w).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states, v_lora, v_nograd, v_w).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        if loradict is None and nograd_loradict is None and nograd_wdict is None:
            attn_output = self.o_proj(attn_output)
        else:
            o_lora = loradict.get("o") if loradict else None
            o_nograd = nograd_loradict.get("o") if nograd_loradict else None
            o_w = nograd_wdict.get("o") if nograd_wdict else None
            attn_output = self.o_proj(attn_output, o_lora, o_nograd, o_w)
        return attn_output, attn_weights

    def _compute_helper_w_delta(self, input: Tensor, wdict: dict, helper: 'LoraHelper') -> Tensor:
        """Compute x @ W delta using a LoraHelper's dimensions."""
        W = wdict["W"]                 # [Lb, in, out]
        C = wdict.get("C", None)       # [Lb, out] or None

        Lb = W.shape[0]
        num_beams = input.shape[0] // Lb
        x = input.reshape(Lb, num_beams, -1, helper.in_features)

        w_out = torch.matmul(x, W[:, None, :, :])

        if C is not None:
            w_out = w_out + C[:, None, None, :]

        w_out = w_out.reshape(*input.shape[:-1], helper.out_features)
        return w_out

    def lora_params_numel(self, lora_ranks):
        """lora_ranks: dict with keys 'q_query', 'q_gate', 'k_proj', 'v_proj', 'o_proj'."""
        cache_key = (lora_ranks["q_query"], lora_ranks["q_gate"],
                     lora_ranks["k_proj"], lora_ranks["v_proj"], lora_ranks["o_proj"])
        if not hasattr(self, "_lora_numel_cache"):
            self._lora_numel_cache = {}
        if cache_key not in self._lora_numel_cache:
            self._lora_numel_cache[cache_key] = (
                self.q_query_lora.lora_params_numel(lora_ranks["q_query"])
                + self.q_gate_lora.lora_params_numel(lora_ranks["q_gate"])
                + self.k_proj.lora_params_numel(lora_ranks["k_proj"])
                + self.v_proj.lora_params_numel(lora_ranks["v_proj"])
                + self.o_proj.lora_params_numel(lora_ranks["o_proj"])
            )
        return self._lora_numel_cache[cache_key]

    def set_generate_func(self, method):
        self.q_query_lora.set_generate_func(method)
        self.q_gate_lora.set_generate_func(method)
        self.k_proj.set_generate_func(method)
        self.v_proj.set_generate_func(method)
        self.o_proj.set_generate_func(method)

    def generate_lora_dict(self, lora_ranks, scale, plain_tensor):
        torch._assert(plain_tensor.shape[-1] == self.lora_params_numel(lora_ranks), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match lora_params_numel {self.lora_params_numel(lora_ranks)}")
        idx = 0
        q_query = self.q_query_lora.generate_lora_dict(lora_ranks["q_query"], scale, plain_tensor[:, idx: idx + self.q_query_lora.lora_params_numel(lora_ranks["q_query"])])
        idx += self.q_query_lora.lora_params_numel(lora_ranks["q_query"])
        q_gate = self.q_gate_lora.generate_lora_dict(lora_ranks["q_gate"], scale, plain_tensor[:, idx: idx + self.q_gate_lora.lora_params_numel(lora_ranks["q_gate"])])
        idx += self.q_gate_lora.lora_params_numel(lora_ranks["q_gate"])
        k = self.k_proj.generate_lora_dict(lora_ranks["k_proj"], scale, plain_tensor[:, idx: idx + self.k_proj.lora_params_numel(lora_ranks["k_proj"])])
        idx += self.k_proj.lora_params_numel(lora_ranks["k_proj"])
        v = self.v_proj.generate_lora_dict(lora_ranks["v_proj"], scale, plain_tensor[:, idx: idx + self.v_proj.lora_params_numel(lora_ranks["v_proj"])])
        idx += self.v_proj.lora_params_numel(lora_ranks["v_proj"])
        o = self.o_proj.generate_lora_dict(lora_ranks["o_proj"], scale, plain_tensor[:, idx: idx + self.o_proj.lora_params_numel(lora_ranks["o_proj"])])
        return {"q_query": q_query, "q_gate": q_gate, "k": k, "v": v, "o": o}

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        q_query = self.q_query_lora.init_lora_dict(lora_ranks["q_query"], scale, device, dtype)
        q_gate = self.q_gate_lora.init_lora_dict(lora_ranks["q_gate"], scale, device, dtype)
        k = self.k_proj.init_lora_dict(lora_ranks["k_proj"], scale, device, dtype)
        v = self.v_proj.init_lora_dict(lora_ranks["v_proj"], scale, device, dtype)
        o = self.o_proj.init_lora_dict(lora_ranks["o_proj"], scale, device, dtype)
        return {"q_query": q_query, "q_gate": q_gate, "k": k, "v": v, "o": o}


class LoraQwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            # GatedDeltaNet doesn't need LoRA
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
            # linear_attention layers: no LoRA at all (neither attention nor mlp)
            self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        elif self.layer_type == "full_attention":
            self.self_attn = LoraQwen3_5Attention(config, layer_idx)
            # full_attention layers: LoRA on both attention and mlp
            self.mlp = LoraQwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        loradict: Optional[dict] = None,
        nograd_loradict: Optional[dict] = None,
        nograd_wdict: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.layer_type == "linear_attention":
            # GatedDeltaNet doesn't use LoRA
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            # Self Attention with LoRA
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                loradict=loradict['attention'] if loradict is not None else None,
                nograd_loradict=nograd_loradict['attention'] if nograd_loradict is not None else None,
                nograd_wdict=nograd_wdict['attention'] if nograd_wdict is not None else None,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            # full_attention layers: pass MLP LoRA
            hidden_states = self.mlp(
                hidden_states,
                loradict=loradict['mlp'] if loradict is not None else None,
                nograd_loradict=nograd_loradict['mlp'] if nograd_loradict is not None else None,
                nograd_wdict=nograd_wdict['mlp'] if nograd_wdict is not None else None,
            )
        else:
            # linear_attention layers: no LoRA on MLP
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def lora_params_numel(self, lora_ranks):
        """lora_ranks: dict with all component rank keys."""
        cache_key = tuple(sorted(lora_ranks.items()))
        if not hasattr(self, "_lora_numel_cache"):
            self._lora_numel_cache = {}
        if cache_key not in self._lora_numel_cache:
            total = 0
            if self.layer_type == "full_attention":
                # Only full_attention layers have LoRA (attention + mlp)
                total += self.self_attn.lora_params_numel(lora_ranks)
                total += self.mlp.lora_params_numel(lora_ranks)
            # linear_attention layers: 0 LoRA params
            self._lora_numel_cache[cache_key] = total
        return self._lora_numel_cache[cache_key]

    def set_generate_func(self, method):
        if self.layer_type == "full_attention":
            self.self_attn.set_generate_func(method)
            self.mlp.set_generate_func(method)

    def generate_lora_dict(self, lora_ranks, scale, plain_tensor):
        torch._assert(plain_tensor.shape[-1] == self.lora_params_numel(lora_ranks), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match lora_params_numel {self.lora_params_numel(lora_ranks)}")
        if self.layer_type != "full_attention":
            # linear_attention layers: no LoRA
            return {"attention": None, "mlp": None}
        idx = 0
        attn_numel = self.self_attn.lora_params_numel(lora_ranks)
        attention = self.self_attn.generate_lora_dict(lora_ranks, scale, plain_tensor[:, idx: idx + attn_numel])
        idx += attn_numel
        mlp_numel = self.mlp.lora_params_numel(lora_ranks)
        mlp = self.mlp.generate_lora_dict(lora_ranks, scale, plain_tensor[:, idx: idx + mlp_numel])
        return {"attention": attention, "mlp": mlp}

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        if self.layer_type != "full_attention":
            # linear_attention layers: no LoRA
            return {"attention": None, "mlp": None}
        attention = self.self_attn.init_lora_dict(lora_ranks, scale, device, dtype)
        mlp = self.mlp.init_lora_dict(lora_ranks, scale, device, dtype)
        return {"attention": attention, "mlp": mlp}


@auto_docstring
class LoraQwen3_5PreTrainedModel(PreTrainedModel):
    config: Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LoraQwen3_5DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "hidden_states": LoraQwen3_5DecoderLayer,
        "attentions": LoraQwen3_5Attention,
    }
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5GatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, Qwen3_5RMSNorm):
            init.zeros_(module.weight)


class LoraQwen3_5TextModel(LoraQwen3_5PreTrainedModel):
    config: Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)

        _num_mem = getattr(config, 'num_mem_token', -1)
        if _num_mem is None or _num_mem <= 0:
            self.has_mem_token = False
        else:
            self.has_mem_token = True
            self.num_mem_token = _num_mem
            self.mem_tokens = nn.Parameter(
                torch.zeros((self.num_mem_token, config.hidden_size), requires_grad=True),
                requires_grad=True,
            )

        if is_main_process_per_node():
            if self.has_mem_token:
                logger.info(f"[LoraQwen3_5TextModel] Created {self.num_mem_token} memory tokens.")
            else:
                logger.info("[LoraQwen3_5TextModel] No memory tokens.")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LoraQwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def reset_mem_tokens(self):
        if self.has_mem_token:
            nn.init.zeros_(self.mem_tokens)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        loradict: Optional[dict] = None,
        nograd_loradict: Optional[dict] = None,
        nograd_wdict: Optional[dict] = None,
        use_mem_token: bool = False,
        context_lengths: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        loradict (`dict`, *optional*):
            A dictionary that maps each layer to its corresponding LoRA parameters.
        nograd_loradict (`dict`, *optional*):
            A dictionary with the same structure as loradict, but containing LoRA parameters
            that do not require gradient computation. Used to save memory.
        nograd_wdict (`dict`, *optional*):
            A dictionary that maps each layer to its corresponding full-rank weight matrices W.
            Each leaf is {"W": [Lb, in, out], "C": [Lb, out] | None}. No gradient.
        use_mem_token (`bool`, *optional*, defaults to `False`):
            Whether to use memory tokens during the forward pass. If set to `True` and the model
            has memory tokens configured, mem_token embeddings will be written into the
            placeholder positions indicated by ``context_lengths``.
        context_lengths (`torch.LongTensor`, *optional*):
            (B,) number of valid tokens per sample.  Required when ``use_mem_token=True``
            under Scheme B.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.has_mem_token and use_mem_token:
                # Scheme B: overwrite placeholder positions with mem_token embeddings.
                if context_lengths is None:
                    raise ValueError(
                        "context_lengths must be provided when use_mem_token=True (Scheme B)"
                    )
                mem = self.mem_tokens.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1)
                for i in range(inputs_embeds.shape[0]):
                    start = context_lengths[i].item()
                    inputs_embeds[i, start:start + self.num_mem_token] = mem[i]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # transformers 5.x added a required cache_position arg.
        _seq_len = inputs_embeds.shape[1]
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=torch.arange(_seq_len, device=inputs_embeds.device),
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Raise early if loradict is provided but is not a dict
        if loradict is not None and not isinstance(loradict, dict):
            raise TypeError(f"loradict must be a dict, got {type(loradict)}")
        if nograd_loradict is not None and not isinstance(nograd_loradict, dict):
            raise TypeError(f"nograd_loradict must be a dict, got {type(nograd_loradict)}")
        if nograd_wdict is not None and not isinstance(nograd_wdict, dict):
            raise TypeError(f"nograd_wdict must be a dict, got {type(nograd_wdict)}")

        if self.has_mem_token and use_mem_token:
            memory_states = torch.zeros(
                (hidden_states.shape[0], self.config.num_hidden_layers, self.num_mem_token, self.config.hidden_size)
            ).to(self.device)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                loradict=loradict[i] if loradict is not None else None,
                nograd_loradict=nograd_loradict[i] if nograd_loradict is not None else None,
                nograd_wdict=nograd_wdict[i] if nograd_wdict is not None else None,
                **kwargs,
            )

            if self.has_mem_token and use_mem_token:
                # Scheme B: extract mem_token hidden states from per-sample positions
                for b in range(hidden_states.shape[0]):
                    start = context_lengths[b].item()
                    memory_states[b, i, :, :] = hidden_states[b, start:start + self.num_mem_token].to(self.device)

        if self.has_mem_token and use_mem_token:
            # Scheme B: strip everything after valid tokens (mem_tokens + padding)
            max_valid = context_lengths.max().item()
            hidden_states = hidden_states[:, :max_valid, :]

        hidden_states = self.norm(hidden_states)

        return MemoryQwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            memory_states=memory_states if (self.has_mem_token and use_mem_token) else None,
        )

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask

    def lora_params_numel(self, lora_ranks):
        if not hasattr(self, "_lora_numel_cache"):
            self._lora_numel_cache = {}
        cache_key = tuple(sorted(lora_ranks.items()))
        if cache_key not in self._lora_numel_cache:
            self._lora_numel_cache[cache_key] = sum(layer.lora_params_numel(lora_ranks) for layer in self.layers)
        return self._lora_numel_cache[cache_key]

    def set_generate_func(self, method):
        for layer in self.layers:
            layer.set_generate_func(method)

    def generate_lora_dict(self, lora_ranks, scale, plain_tensor):
        torch._assert(plain_tensor.shape[-1] == self.lora_params_numel(lora_ranks), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match lora_params_numel {self.lora_params_numel(lora_ranks)}")
        idx = 0
        loradict = {}
        for i, layer in enumerate(self.layers):
            layer_lora_params_numel = layer.lora_params_numel(lora_ranks)
            loradict[i] = layer.generate_lora_dict(lora_ranks, scale, plain_tensor[:, idx: idx + layer_lora_params_numel])
            idx += layer_lora_params_numel
        return loradict

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        loradict = {}
        for i, layer in enumerate(self.layers):
            loradict[i] = layer.init_lora_dict(lora_ranks, scale, device, dtype)
        return loradict


@auto_docstring
class LoraQwen3_5ForCausalLM(LoraQwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Qwen3_5TextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]
    _keys_to_ignore_on_load_missing = [r"model\.mem_tokens"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LoraQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def reset_mem_tokens(self):
        self.model.reset_mem_tokens()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        loradict: Optional[dict] = None,
        nograd_loradict: Optional[dict] = None,
        nograd_wdict: Optional[dict] = None,
        use_mem_token: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        loradict (`dict` of `dict` of `torch.FloatTensor`, *optional*):
            A dictionary that maps each layer to its corresponding LoRA parameters. Each layer's LoRA parameters are
            stored in a nested dictionary.
        nograd_loradict (`dict` of `dict` of `torch.FloatTensor`, *optional*):
            A dictionary with the same structure as loradict, but containing LoRA parameters
            that do not require gradient computation. Used to save memory.
        nograd_wdict (`dict`, *optional*):
            A dictionary that maps each layer to its corresponding full-rank weight matrices W.
            Each leaf is {"W": [Lb, in, out], "C": [Lb, out] | None}. No gradient.
        use_mem_token (`bool`, *optional*, defaults to `False`):
            Whether to use memory tokens during the forward pass. If set to `True` and the model has memory tokens
            configured, they will be appended to the input embeddings and their hidden states will be collected as
            `memory_states` in the output.
        """

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MemoryQwen3_5ModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            loradict=loradict,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
            use_mem_token=use_mem_token,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return MemoryCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            memory_states=getattr(outputs, 'memory_states', None) if use_mem_token else None,
        )

    def lora_params_numel(self, lora_ranks):
        return self.model.lora_params_numel(lora_ranks)

    def set_generate_func(self, method):
        self.model.set_generate_func(method)

    def generate_lora_dict(self, lora_ranks, scale, plain_tensor):
        return self.model.generate_lora_dict(lora_ranks, scale, plain_tensor)

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        return self.model.init_lora_dict(lora_ranks, scale, device, dtype)


__all__ = [
    "LoraQwen3_5ForCausalLM",
    "LoraQwen3_5TextModel",
    "LoraQwen3_5PreTrainedModel",
    "compute_qwen3_5_layer_lora_numel",
]


def compute_qwen3_5_layer_lora_numel(config, lora_ranks: dict, layer_idx: int = 0, verbose: bool = False):
    """
    Compute per-layer LoRA parameter count for the Qwen3_5 family.

    Only ``full_attention`` layers carry LoRA:
      - attention LoRA (5 projections: q_query, q_gate, k, v, o)
      - MLP LoRA (3 projections: gate, up, down)
    ``linear_attention`` layers have 0 LoRA params.

    Args:
        config: Model config object (must have ``layer_types``).
        lora_ranks: Dict mapping component name to its LoRA rank, e.g.
            {"q_query": 16, "k_proj": 8, "mlp_gate": 4, ...}.
        layer_idx: Which layer to compute for.
        verbose: If True, return (total, detail_string) instead of just total.

    Returns:
        int if verbose=False, or (int, str) if verbose=True.
    """
    from src_transformers_lora.LoraHelper import _lora_linear_numel_detail

    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden // n_heads)
    attn_bias = getattr(config, "attention_bias", False)
    intermediate_size = config.intermediate_size

    layer_type = config.layer_types[layer_idx]
    total = 0
    details = []
    details.append(f"  Config: hidden={hidden}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
                   f"head_dim={head_dim}, attn_bias={attn_bias}, "
                   f"intermediate_size={intermediate_size}, lora_ranks={lora_ranks}")

    # Only full_attention layers carry LoRA (both attention and MLP)
    if layer_type != "full_attention":
        if verbose:
            return 0, f"  layer {layer_idx} is {layer_type} (no LoRA) => 0"
        return 0

    # Attention LoRA
    q_dim = n_heads * head_dim
    n, d = _lora_linear_numel_detail("q_query", hidden, q_dim, lora_ranks["q_query"], attn_bias)
    total += n; details.append(d)
    n, d = _lora_linear_numel_detail("q_gate", hidden, q_dim, lora_ranks["q_gate"], attn_bias)
    total += n; details.append(d)
    n, d = _lora_linear_numel_detail("k_proj", hidden, n_kv_heads * head_dim, lora_ranks["k_proj"], attn_bias)
    total += n; details.append(d)
    n, d = _lora_linear_numel_detail("v_proj", hidden, n_kv_heads * head_dim, lora_ranks["v_proj"], attn_bias)
    total += n; details.append(d)
    n, d = _lora_linear_numel_detail("o_proj", n_heads * head_dim, hidden, lora_ranks["o_proj"], attn_bias)
    total += n; details.append(d)

    # MLP LoRA
    details.append(f"  MLP LoRA:")
    n_gate, d = _lora_linear_numel_detail("mlp_gate", hidden, intermediate_size, lora_ranks["mlp_gate"], False)
    total += n_gate; details.append(d)
    n_up, d = _lora_linear_numel_detail("mlp_up", hidden, intermediate_size, lora_ranks["mlp_up"], False)
    total += n_up; details.append(d)
    n_down, d = _lora_linear_numel_detail("mlp_down", intermediate_size, hidden, lora_ranks["mlp_down"], False)
    total += n_down; details.append(d)

    details.append(f"  TOTAL = {total}")
    if verbose:
        return total, "\n".join(details)
    return total
