# LoRA-adapted Qwen3_5Moe model
# Based on Qwen3_5Moe.py with LoRA modifications following the pattern of LoraQwen3Moe.py

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from utils.myparallel import is_main_process_per_node

logger = logging.getLogger(__name__)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    init, ACT2FN, Cache, DynamicCache, GenerationMixin,
    use_experts_implementation,
    use_kernelized_func,
    create_causal_mask,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    MoeCausalLMOutputWithPast, MoeModelOutputWithPast,
    ROPE_INIT_FUNCTIONS, dynamic_rope_update,
    ALL_ATTENTION_FUNCTIONS, PreTrainedModel,
    Unpack,
    TransformersKwargs, auto_docstring, can_return_tuple,
    maybe_autocast, merge_with_config_defaults,
    OutputRecorder, capture_outputs,
    Qwen3_5MoeConfig, Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig,
    Qwen3_5MoeRMSNorm, Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeMLP, Qwen3_5MoeExperts, Qwen3_5MoeTopKRouter,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeModelOutputWithPast,
    load_balancing_loss_func,
    rotate_half, apply_rotary_pos_emb, repeat_kv, eager_attention_forward,
    apply_mask_to_padding_states,
)

from math import sqrt
from torch import Tensor


@dataclass
class MemoryQwen3_5MoeModelOutputWithPast(Qwen3_5MoeModelOutputWithPast):
    """Extends Qwen3_5MoeModelOutputWithPast with memory states."""
    memory_states: Optional[torch.Tensor] = None  # (batch_size, num_layer, num_mem_token, hidden_size)


@dataclass
class MemoryMoeCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    """Extends MoeCausalLMOutputWithPast with memory states."""
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

        # If both nograd_loradict and nograd_wdict exist, merge A@B into W for a single matmul
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

        x = input.reshape(Lb, num_beams, -1, self.in_features)
        tmp = torch.matmul(x, A[:, None, :, :])
        lora_out = torch.matmul(tmp, B[:, None, :, :])

        if self.bias is None:
            torch._assert(C is None, "If bias is None, loradict['C'] must also be None")
        else:
            torch._assert(C is not None, "If bias is not None, loradict['C'] must also be not None")
            lora_out = lora_out + C[:, None, None, :]

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
        A = nograd_loradict["A"]
        B = nograd_loradict["B"]
        C_lora = nograd_loradict.get("C", None)

        W = nograd_wdict["W"]
        C_w = nograd_wdict.get("C", None)

        W_merged = W + torch.bmm(A, B)

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

    def lora_delta_moe(self, input: Tensor, loradict: dict, batch_indices: Tensor) -> Tensor:
        """LoRA delta for MoE experts where tokens are routed independently.

        In MoE, hidden_states are flattened to (B*S, H) and then routed per-token.
        The number of tokens routed to each expert is arbitrary and NOT a multiple
        of batch_size.  ``batch_indices`` (shape [N]) maps each routed token back
        to its batch sample so we can apply the correct per-sample LoRA.

        Args:
            input: (N, in_features) — routed tokens for one expert.
            loradict: {"A": [Lb, in, r], "B": [Lb, r, out], "C": [Lb, out] | None}
            batch_indices: (N,) — batch index for each token, values in [0, Lb).
        Returns:
            (N, out_features)
        """
        A = loradict["A"]              # [Lb, in, r]
        B = loradict["B"]              # [Lb, r, out]
        C = loradict.get("C", None)    # [Lb, out] or None

        # Gather per-token LoRA params: A_tok [N, in, r], B_tok [N, r, out]
        A_tok = A[batch_indices]        # [N, in, r]
        B_tok = B[batch_indices]        # [N, r, out]

        # input: [N, in] -> [N, 1, in] @ [N, in, r] -> [N, 1, r]
        tmp = torch.bmm(input.unsqueeze(1), A_tok)   # [N, 1, r]
        lora_out = torch.bmm(tmp, B_tok).squeeze(1)  # [N, out]

        if C is not None:
            C_tok = C[batch_indices]    # [N, out]
            lora_out = lora_out + C_tok

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
                A = plain_tensor[:, idx: idx + in_f * r].view(-1, in_f, r) * sqrt(scale)
                idx += in_f * r
                B = plain_tensor[:, idx: idx + out_f * r].view(-1, r, out_f) * sqrt(scale)
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


class LoraQwen3_5MoeExperts(Qwen3_5MoeExperts):
    """Qwen3_5MoeExperts with per-expert LoRA support via LoraHelper.

    Each expert has 3 independent LoRA adapters:
      - expert_gate_lora: hidden -> intermediate  (gate half of gate_up_proj)
      - expert_up_lora:   hidden -> intermediate  (up half of gate_up_proj)
      - expert_down_lora: intermediate -> hidden  (down_proj)
    """

    def __init__(self, config):
        super().__init__(config)
        # gate and up are independent LoRA on the two halves of gate_up_proj
        self.expert_gate_lora = [
            LoraHelper(self.hidden_dim, self.intermediate_dim, bias=False)
            for _ in range(self.num_experts)
        ]
        self.expert_up_lora = [
            LoraHelper(self.hidden_dim, self.intermediate_dim, bias=False)
            for _ in range(self.num_experts)
        ]
        self.expert_down_lora = [
            LoraHelper(self.intermediate_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_experts)
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        loradict=None,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Pre-compute tokens_per_sample for batch index recovery.
        # hidden_states is (B_total * S, H) where B_total = Lb * num_beams.
        # Lb = A.shape[0] from loradict.  tokens_per_sample = B_total * S / Lb
        # so that batch_indices = token_idx // tokens_per_sample maps to [0, Lb).
        tokens_per_sample = None
        if loradict is not None:
            # Pick any expert's non-None sub-loradict to read Lb
            for _eidx in loradict:
                for _comp in ("gate", "up", "down"):
                    _ld = loradict[_eidx].get(_comp)
                    if _ld is not None:
                        Lb = _ld["A"].shape[0]
                        total_tokens = hidden_states.shape[0]  # B_total * S
                        tokens_per_sample = total_tokens // Lb
                        break
                if tokens_per_sample is not None:
                    break

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            # Base gate_up computation, then split
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            # Add independent LoRA deltas for gate and up
            if loradict is not None:
                eidx = expert_idx.item()
                expert_ld = loradict[eidx]
                # token_idx are indices into (B_total * S,)
                # Recover original batch index: each sample owns tokens_per_sample tokens
                if tokens_per_sample is not None:
                    batch_indices = token_idx // tokens_per_sample  # [N], values in [0, Lb)
                else:
                    batch_indices = None
                if expert_ld.get("gate") is not None and batch_indices is not None:
                    gate = gate + self.expert_gate_lora[expert_idx].lora_delta_moe(
                        current_state, expert_ld["gate"], batch_indices
                    )
                if expert_ld.get("up") is not None and batch_indices is not None:
                    up = up + self.expert_up_lora[expert_idx].lora_delta_moe(
                        current_state, expert_ld["up"], batch_indices
                    )
            current_hidden_states = self.act_fn(gate) * up
            down = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            # Add LoRA delta for down_proj
            if loradict is not None:
                if expert_ld.get("down") is not None and batch_indices is not None:
                    down = down + self.expert_down_lora[expert_idx].lora_delta_moe(
                        current_hidden_states, expert_ld["down"], batch_indices
                )
            current_hidden_states = down * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def lora_params_numel(self, lora_ranks):
        """lora_ranks: dict with keys 'expert_gate', 'expert_up', 'expert_down'."""
        cache_key = (lora_ranks["expert_gate"], lora_ranks["expert_up"], lora_ranks["expert_down"])
        if not hasattr(self, "_lora_numel_cache"):
            self._lora_numel_cache = {}
        if cache_key not in self._lora_numel_cache:
            # All experts have the same dimensions, so multiply by num_experts
            single_expert = (
                self.expert_gate_lora[0].lora_params_numel(lora_ranks["expert_gate"])
                + self.expert_up_lora[0].lora_params_numel(lora_ranks["expert_up"])
                + self.expert_down_lora[0].lora_params_numel(lora_ranks["expert_down"])
            )
            self._lora_numel_cache[cache_key] = single_expert * self.num_experts
        return self._lora_numel_cache[cache_key]

    def set_generate_func(self, method):
        for helper in self.expert_gate_lora:
            helper.set_generate_func(method)
        for helper in self.expert_up_lora:
            helper.set_generate_func(method)
        for helper in self.expert_down_lora:
            helper.set_generate_func(method)

    def generate_lora_dict(self, lora_ranks, scale, plain_tensor):
        torch._assert(
            plain_tensor.shape[-1] == self.lora_params_numel(lora_ranks),
            f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match lora_params_numel {self.lora_params_numel(lora_ranks)}"
        )
        r_gate = lora_ranks["expert_gate"]
        r_up = lora_ranks["expert_up"]
        r_down = lora_ranks["expert_down"]
        idx = 0
        loradict = {}
        for i in range(self.num_experts):
            gate_numel = self.expert_gate_lora[i].lora_params_numel(r_gate)
            gate = self.expert_gate_lora[i].generate_lora_dict(
                r_gate, scale, plain_tensor[:, idx: idx + gate_numel]
            )
            idx += gate_numel
            up_numel = self.expert_up_lora[i].lora_params_numel(r_up)
            up = self.expert_up_lora[i].generate_lora_dict(
                r_up, scale, plain_tensor[:, idx: idx + up_numel]
            )
            idx += up_numel
            down_numel = self.expert_down_lora[i].lora_params_numel(r_down)
            down = self.expert_down_lora[i].generate_lora_dict(
                r_down, scale, plain_tensor[:, idx: idx + down_numel]
            )
            idx += down_numel
            loradict[i] = {"gate": gate, "up": up, "down": down}
        return loradict

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        r_gate = lora_ranks["expert_gate"]
        r_up = lora_ranks["expert_up"]
        r_down = lora_ranks["expert_down"]
        loradict = {}
        for i in range(self.num_experts):
            gate = self.expert_gate_lora[i].init_lora_dict(r_gate, scale, device, dtype)
            up = self.expert_up_lora[i].init_lora_dict(r_up, scale, device, dtype)
            down = self.expert_down_lora[i].init_lora_dict(r_down, scale, device, dtype)
            loradict[i] = {"gate": gate, "up": up, "down": down}
        return loradict


class LoraQwen3_5MoeSparseMoeBlock(Qwen3_5MoeSparseMoeBlock):
    """Qwen3_5MoeSparseMoeBlock with LoRA support on experts."""

    def __init__(self, config):
        super().__init__(config)
        # Replace experts with LoRA-enabled version
        self.experts = LoraQwen3_5MoeExperts(config)

    def forward(self, hidden_states: torch.Tensor, loradict=None, nograd_loradict=None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights,
            loradict=loradict, nograd_loradict=nograd_loradict,
        )

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    def lora_params_numel(self, lora_ranks):
        return self.experts.lora_params_numel(lora_ranks)

    def set_generate_func(self, method):
        self.experts.set_generate_func(method)

    def generate_lora_dict(self, lora_ranks, scale, plain_tensor):
        return self.experts.generate_lora_dict(lora_ranks, scale, plain_tensor)

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        return self.experts.init_lora_dict(lora_ranks, scale, device, dtype)


class LoraQwen3_5MoeMLP(Qwen3_5MoeMLP):
    def __init__(self, config: Qwen3_5MoeConfig, intermediate_size: int):
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
        return {"gate": gate, "up": up, "down": down}

    def init_lora_dict(self, r, scale, device, dtype):
        gate = self.gate_proj.init_lora_dict(r, scale, device, dtype)
        up = self.up_proj.init_lora_dict(r, scale, device, dtype)
        down = self.down_proj.init_lora_dict(r, scale, device, dtype)
        return {"gate": gate, "up": up, "down": down}


class LoraQwen3_5MoeAttention(Qwen3_5MoeAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper with LoRA support"""

    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int):
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

            # Add independent LoRA deltas for query and gate from loradict
            if loradict is not None and loradict.get("q_query") is not None:
                query_states = query_states + self.q_query_lora.lora_delta(hidden_states, loradict["q_query"]).view(*input_shape, -1, self.head_dim)
            if loradict is not None and loradict.get("q_gate") is not None:
                gate = gate + self.q_gate_lora.lora_delta(hidden_states, loradict["q_gate"]).view(*input_shape, -1, self.head_dim)
            # Add independent LoRA deltas for query and gate from nograd_loradict
            if nograd_loradict is not None and nograd_loradict.get("q_query") is not None:
                query_states = query_states + self.q_query_lora.lora_delta(hidden_states, nograd_loradict["q_query"]).view(*input_shape, -1, self.head_dim)
            if nograd_loradict is not None and nograd_loradict.get("q_gate") is not None:
                gate = gate + self.q_gate_lora.lora_delta(hidden_states, nograd_loradict["q_gate"]).view(*input_shape, -1, self.head_dim)
            # Add full-rank W deltas for query and gate from nograd_wdict
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
        W = wdict["W"]
        C = wdict.get("C", None)

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


class LoraQwen3_5MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            # GatedDeltaNet doesn't need LoRA
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
            # linear_attention layers: no LoRA at all (neither attention nor experts)
            self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        elif self.layer_type == "full_attention":
            self.self_attn = LoraQwen3_5MoeAttention(config, layer_idx)
            # full_attention layers: LoRA on both attention and experts
            self.mlp = LoraQwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        **kwargs: Unpack[FlashAttentionKwargs],
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
            # full_attention layers: pass expert LoRA
            hidden_states = self.mlp(
                hidden_states,
                loradict=loradict['experts'] if loradict is not None else None,
                nograd_loradict=nograd_loradict['experts'] if nograd_loradict is not None else None,
            )
        else:
            # linear_attention layers: no LoRA on MoE
            hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
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
                # Only full_attention layers have LoRA (attention + experts)
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
            return {"attention": None, "experts": None}
        idx = 0
        attn_numel = self.self_attn.lora_params_numel(lora_ranks)
        attention = self.self_attn.generate_lora_dict(lora_ranks, scale, plain_tensor[:, idx: idx + attn_numel])
        idx += attn_numel
        mlp_numel = self.mlp.lora_params_numel(lora_ranks)
        experts = self.mlp.generate_lora_dict(lora_ranks, scale, plain_tensor[:, idx: idx + mlp_numel])
        return {"attention": attention, "experts": experts}

    def init_lora_dict(self, lora_ranks, scale, device, dtype):
        if self.layer_type != "full_attention":
            # linear_attention layers: no LoRA
            return {"attention": None, "experts": None}
        attention = self.self_attn.init_lora_dict(lora_ranks, scale, device, dtype)
        experts = self.mlp.init_lora_dict(lora_ranks, scale, device, dtype)
        return {"attention": attention, "experts": experts}


@auto_docstring
class LoraQwen3_5MoePreTrainedModel(PreTrainedModel):
    config: Qwen3_5MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LoraQwen3_5MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3_5MoeTopKRouter, index=0),
        "hidden_states": LoraQwen3_5MoeDecoderLayer,
        "attentions": LoraQwen3_5MoeAttention,
    }
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5MoeGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, Qwen3_5MoeRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3_5MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_5MoeSparseMoeBlock):
            init.normal_(module.gate.weight, mean=0.0, std=self.config.initializer_range)


class LoraQwen3_5MoeTextModel(LoraQwen3_5MoePreTrainedModel):
    config: Qwen3_5MoeTextConfig

    def __init__(self, config: Qwen3_5MoeTextConfig):
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
                logger.info(f"[LoraQwen3_5MoeTextModel] Created {self.num_mem_token} memory tokens.")
            else:
                logger.info("[LoraQwen3_5MoeTextModel] No memory tokens.")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LoraQwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
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
    ) -> Qwen3_5MoeModelOutputWithPast:
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

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
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
                loradict=loradict.get(i, None) if loradict is not None else None,
                nograd_loradict=nograd_loradict.get(i, None) if nograd_loradict is not None else None,
                nograd_wdict=nograd_wdict.get(i, None) if nograd_wdict is not None else None,
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

        return MemoryQwen3_5MoeModelOutputWithPast(
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
class LoraQwen3_5MoeForCausalLM(LoraQwen3_5MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Qwen3_5MoeTextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]
    _keys_to_ignore_on_load_missing = [r"model\.mem_tokens"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LoraQwen3_5MoeTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

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
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        loradict: Optional[dict] = None,
        nograd_loradict: Optional[dict] = None,
        nograd_wdict: Optional[dict] = None,
        use_mem_token: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        loradict (`dict` of `dict` of `torch.FloatTensor`, *optional*):
            A dictionary that maps each layer to its corresponding LoRA parameters. Each layer's LoRA parameters are
            stored in a nested dictionary.
        use_mem_token (`bool`, *optional*, defaults to `False`):
            Whether to use memory tokens during the forward pass. If set to `True` and the model has memory tokens
            configured, they will be appended to the input embeddings and their hidden states will be collected as
            `memory_states` in the output.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LoraQwen3_5MoeForCausalLM

        >>> model = LoraQwen3_5MoeForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate with LoRA
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MemoryQwen3_5MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
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

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MemoryMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
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
    "LoraQwen3_5MoeForCausalLM",
    "LoraQwen3_5MoeTextModel",
    "LoraQwen3_5MoePreTrainedModel",
    "compute_qwen3_5moe_layer_lora_numel",
]


def compute_qwen3_5moe_layer_lora_numel(config, lora_ranks: dict, layer_idx: int = 0, verbose: bool = False):
    """
    Compute per-layer LoRA parameter count for the Qwen3_5Moe family.

    Only ``full_attention`` layers carry LoRA:
      - attention LoRA (5 projections: q_query, q_gate, k, v, o)
      - expert LoRA (num_experts * 3 projections: gate, up, down per expert)
    ``linear_attention`` layers have 0 LoRA params.

    Args:
        config: Model config object (must have ``layer_types``).
        lora_ranks: Dict mapping component name to its LoRA rank, e.g.
            {"q_query": 16, "k_proj": 8, "expert_gate": 4, ...}.
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
    num_experts = config.num_experts
    moe_intermediate = config.moe_intermediate_size

    layer_type = config.layer_types[layer_idx]
    total = 0
    details = []
    details.append(f"  Config: hidden={hidden}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
                   f"head_dim={head_dim}, attn_bias={attn_bias}, num_experts={num_experts}, "
                   f"moe_intermediate={moe_intermediate}, lora_ranks={lora_ranks}")

    # Only full_attention layers carry LoRA (both attention and expert)
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

    # Expert LoRA
    details.append(f"  Expert LoRA ({num_experts} experts):")
    n_gate, d = _lora_linear_numel_detail("expert_gate", hidden, moe_intermediate, lora_ranks["expert_gate"], False)
    details.append(f"    per-expert: {d}")
    n_up, d = _lora_linear_numel_detail("expert_up", hidden, moe_intermediate, lora_ranks["expert_up"], False)
    details.append(f"    per-expert: {d}")
    n_down, d = _lora_linear_numel_detail("expert_down", moe_intermediate, hidden, lora_ranks["expert_down"], False)
    details.append(f"    per-expert: {d}")
    expert_total = (n_gate + n_up + n_down) * num_experts
    total += expert_total
    details.append(f"    expert_total = ({n_gate} + {n_up} + {n_down}) * {num_experts} = {expert_total}")

    details.append(f"  TOTAL = {total}")
    if verbose:
        return total, "\n".join(details)
    return total
