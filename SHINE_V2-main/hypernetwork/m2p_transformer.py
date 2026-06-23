"""
Unified Transformer (m2p_transformer.py)

Merges the architectures of Qwen3Moe and Qwen3_5Moe into a single configurable
transformer.  All architectural knobs are exposed as plain constructor parameters
(no config dataclass required).

Design principle for torch.compile friendliness:
  - Every architectural branch (gated attention, RMSNorm variant,
    MoE-vs-dense MLP, RoPE variant) is resolved **once** at __init__ time by
    binding the correct `forward` callable.  The hot path contains zero
    `if config.xxx` guards.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return hidden_states.reshape(b, h * n_rep, s, d)

def eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

# ---------------------------------------------------------------------------
# RMSNorm variants
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Standard RMSNorm (Qwen3Moe style): weight initialized to ones."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class RMSNormZeroInit(nn.Module):
    """RMSNorm with zero-init weight (Qwen3_5Moe style): output = norm(x) * (1 + w)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * x).to(dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gating (Qwen3_5Moe GatedDeltaNet style).

    output = RMSNorm(x) * weight * SiLU(gate)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = (self.weight * x).to(dtype)
        x = x * F.silu(gate.float()).to(dtype)
        return x

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

def _make_norm(dim: int, eps: float, zero_init: bool) -> nn.Module:
    return RMSNormZeroInit(dim, eps) if zero_init else RMSNorm(dim, eps)

# ---------------------------------------------------------------------------
# Rotary Embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Standard 1-D RoPE (Qwen3Moe style)."""

    def __init__(self, head_dim: int, max_position_embeddings: int = 131072,
                 rope_theta: float = 10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # position_ids: [B, S]
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class LLM_MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[hidden_act]
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# ---------------------------------------------------------------------------
# MoE components
# ---------------------------------------------------------------------------

class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int,
                 norm_topk_prob: bool = True):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states: Tensor):
        hidden_states = hidden_states.reshape(-1, self.weight.shape[1])
        logits = F.linear(hidden_states, self.weight)
        probs = F.softmax(logits, dtype=torch.float, dim=-1)
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_val = top_val / top_val.sum(dim=-1, keepdim=True)
        return logits, top_val.to(logits.dtype), top_idx

class Experts(nn.Module):
    def __init__(self, hidden_size: int, moe_intermediate_size: int, num_experts: int,
                 hidden_act: str = "silu"):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate_size))
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hits = (mask.sum(dim=(-1, -2)) > 0).nonzero()
        for eidx in hits:
            eidx = eidx[0]
            if eidx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[eidx])
            cur = hidden_states[token_idx]
            gate, up = F.linear(cur, self.gate_up_proj[eidx]).chunk(2, dim=-1)
            cur = self.act_fn(gate) * up
            cur = F.linear(cur, self.down_proj[eidx])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final

class SparseMoeBlock(nn.Module):
    def __init__(self, hidden_size: int, moe_intermediate_size: int, num_experts: int,
                 num_experts_per_tok: int, norm_topk_prob: bool = True,
                 hidden_act: str = "silu"):
        super().__init__()
        self.experts = Experts(hidden_size, moe_intermediate_size, num_experts, hidden_act)
        self.gate = TopKRouter(hidden_size, num_experts, num_experts_per_tok, norm_topk_prob)

    def forward(self, hidden_states: Tensor) -> Tensor:
        B, S, D = hidden_states.shape
        flat = hidden_states.reshape(-1, D)
        _, weights, indices = self.gate(flat)
        out = self.experts(flat, indices, weights)
        return out.reshape(B, S, D)

# ---------------------------------------------------------------------------
# Attention  (gated vs non-gated selected at init)
# ---------------------------------------------------------------------------

def _attn_forward_plain(
    self,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[Tensor, Tensor | None]:
    """Non-gated attention forward (Qwen3Moe style)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    attn_out, attn_w = eager_attention_forward(
        self, q, k, v, attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        **kwargs,
    )

    attn_out = attn_out.reshape(*input_shape, -1).contiguous()
    attn_out = self.o_proj(attn_out)
    return attn_out, attn_w

def _attn_forward_gated(
    self,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[Tensor, Tensor | None]:
    """Gated attention forward (Qwen3_5Moe style)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    q = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    attn_out, attn_w = eager_attention_forward(
        self, q, k, v, attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        **kwargs,
    )

    attn_out = attn_out.reshape(*input_shape, -1).contiguous()
    attn_out = attn_out * torch.sigmoid(gate)
    attn_out = self.o_proj(attn_out)
    return attn_out, attn_w

class Attention(nn.Module):
    """
    Unified attention module.

    Args:
        hidden_size: model hidden dimension.
        num_attention_heads: number of query heads.
        num_key_value_heads: number of KV heads (for GQA).
        head_dim: per-head dimension (default: hidden_size // num_attention_heads).
        attention_bias: whether projections have bias.
        attention_dropout: dropout rate.
        rms_norm_eps: epsilon for QK norm.
        use_gated_attention: if True, q_proj outputs query+gate and applies sigmoid gate.
        layer_idx: layer index (for KV cache).
        norm_zero_init: use zero-init RMSNorm for q/k norms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        layer_idx: int,
        head_dim: int | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_gated_attention: bool = True,
        norm_zero_init: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = attention_dropout
        self.is_causal = True
        self.use_gated_attention = use_gated_attention

        q_out_dim = num_attention_heads * self.head_dim * (2 if use_gated_attention else 1)
        self.q_proj = nn.Linear(hidden_size, q_out_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)

        self.q_norm = _make_norm(self.head_dim, rms_norm_eps, norm_zero_init)
        self.k_norm = _make_norm(self.head_dim, rms_norm_eps, norm_zero_init)

        # Bind the correct forward at init time — no branching in the hot path
        if use_gated_attention:
            self.forward = lambda *a, **kw: _attn_forward_gated(self, *a, **kw)
        else:
            self.forward = lambda *a, **kw: _attn_forward_plain(self, *a, **kw)

# ---------------------------------------------------------------------------
# Decoder Layer  (prenorm/postnorm × MoE/dense selected at init)
# ---------------------------------------------------------------------------

# -- Pre-Norm variants (norm → sublayer → residual add) --

def _decoder_forward_prenorm_dense(
    self,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None = None,
    past_key_values=None,
    **kwargs,
) -> Tensor:
    """Pre-norm decoder forward with dense MLP."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states, position_embeddings, attention_mask,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

def _decoder_forward_prenorm_moe(
    self,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None = None,
    past_key_values=None,
    **kwargs,
) -> Tensor:
    """Pre-norm decoder forward with MoE block."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states, position_embeddings, attention_mask,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

# -- Post-Norm variants (sublayer → residual add → norm) --

def _decoder_forward_postnorm_dense(
    self,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None = None,
    past_key_values=None,
    **kwargs,
) -> Tensor:
    """Post-norm decoder forward with dense MLP."""
    residual = hidden_states
    hidden_states, _ = self.self_attn(
        hidden_states, position_embeddings, attention_mask,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = self.input_layernorm(residual + hidden_states)

    residual = hidden_states
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_attention_layernorm(residual + hidden_states)
    return hidden_states

def _decoder_forward_postnorm_moe(
    self,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None = None,
    past_key_values=None,
    **kwargs,
) -> Tensor:
    """Post-norm decoder forward with MoE block."""
    residual = hidden_states
    hidden_states, _ = self.self_attn(
        hidden_states, position_embeddings, attention_mask,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = self.input_layernorm(residual + hidden_states)

    residual = hidden_states
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_attention_layernorm(residual + hidden_states)
    return hidden_states

class DecoderLayer(nn.Module):
    """
    Unified decoder layer.

    Args:
        hidden_size, num_attention_heads, num_key_value_heads, head_dim,
        attention_bias, attention_dropout, rms_norm_eps, use_gated_attention,
        norm_zero_init: forwarded to Attention.

        prenorm: if True, apply norm before sublayer (Pre-LN); if False, after (Post-LN).
        use_moe: if True, MLP is a SparseMoeBlock.
        intermediate_size: dense MLP intermediate size (used when use_moe=False).
        moe_intermediate_size: MoE expert intermediate size.
        num_experts, num_experts_per_tok, norm_topk_prob: MoE params.
        hidden_act: activation function name.
        layer_idx: layer index.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        layer_idx: int,
        head_dim: int | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_gated_attention: bool = True,
        norm_zero_init: bool = True,
        prenorm: bool = True,
        use_moe: bool = True,
        intermediate_size: int = 0,
        moe_intermediate_size: int = 512,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            layer_idx=layer_idx,
            head_dim=head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            use_gated_attention=use_gated_attention,
            norm_zero_init=norm_zero_init,
        )

        self.input_layernorm = _make_norm(hidden_size, rms_norm_eps, norm_zero_init)
        self.post_attention_layernorm = _make_norm(hidden_size, rms_norm_eps, norm_zero_init)

        if use_moe:
            self.mlp = SparseMoeBlock(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                norm_topk_prob=norm_topk_prob,
                hidden_act=hidden_act,
            )
            if prenorm:
                self.forward = lambda *a, **kw: _decoder_forward_prenorm_moe(self, *a, **kw)
            else:
                self.forward = lambda *a, **kw: _decoder_forward_postnorm_moe(self, *a, **kw)
        else:
            self.mlp = LLM_MLP(hidden_size, intermediate_size, hidden_act)
            if prenorm:
                self.forward = lambda *a, **kw: _decoder_forward_prenorm_dense(self, *a, **kw)
            else:
                self.forward = lambda *a, **kw: _decoder_forward_postnorm_dense(self, *a, **kw)

# ---------------------------------------------------------------------------
# Full Transformer Model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """
    Unified transformer model (decoder-only).

    This is a standalone nn.Module (not a HuggingFace PreTrainedModel) that can
    be used as a building block.  All config is passed as explicit parameters.

    Args:
        hidden_size: model hidden dimension.
        num_hidden_layers: number of decoder layers.
        num_attention_heads: number of query heads.
        num_key_value_heads: number of KV heads.
        head_dim: per-head dimension (default: hidden_size // num_attention_heads).
        attention_bias: whether attention projections have bias.
        attention_dropout: dropout rate for attention.
        rms_norm_eps: epsilon for RMSNorm.
        use_gated_attention: use gated attention (Qwen3_5Moe style).
        norm_zero_init: use zero-init RMSNorm (Qwen3_5Moe style).
        prenorm: if True, Pre-LN (norm before sublayer); if False, Post-LN.
        hidden_act: activation function name.
        rope_theta: RoPE base frequency.
        max_position_embeddings: max sequence length for RoPE.

        # Per-layer MLP config:
        #   layer_is_moe: specifies each layer's MLP type.
        #     - A single string "moe" or "full": all layers use that type.
        #     - A list of strings ["moe", "full", ...] of length num_hidden_layers.
        layer_is_moe: per-layer MLP type specification (required).
        intermediate_size: dense MLP intermediate size (for "full" layers).
        moe_intermediate_size: MoE expert intermediate size.
        num_experts: number of experts.
        num_experts_per_tok: top-k experts per token.
        norm_topk_prob: normalize top-k probabilities.
        initializer_range: std for weight init.
        last_norm: final norm type — "normal" (standard RMSNorm), "gated" (RMSNormGated
            with SiLU gate), or "none" (no final norm, identity placeholder).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_gated_attention: bool = True,
        norm_zero_init: bool = True,
        prenorm: bool = True,
        hidden_act: str = "silu",
        rope_theta: float = 10000000.0,
        max_position_embeddings: int = 262144,
        layer_is_moe: Union[str, List[str]] = "moe",
        intermediate_size: int = 11008,
        moe_intermediate_size: int = 512,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        initializer_range: float = 0.02,
        last_norm: str = "normal",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        if last_norm not in ("normal", "gated", "none"):
            raise ValueError(f"last_norm must be 'normal', 'gated', or 'none', got '{last_norm}'")
        self.last_norm_type = last_norm

        effective_head_dim = head_dim or (hidden_size // num_attention_heads)

        # Per-layer MoE flags: resolve string shorthand to list of bools
        if isinstance(layer_is_moe, str):
            if layer_is_moe == "moe":
                layer_is_moe_flags = [True] * num_hidden_layers
            elif layer_is_moe == "full":
                layer_is_moe_flags = [False] * num_hidden_layers
            else:
                raise ValueError(f"layer_is_moe must be 'moe', 'full', or a list, got '{layer_is_moe}'")
        elif isinstance(layer_is_moe, list):
            if len(layer_is_moe) != num_hidden_layers:
                raise ValueError(
                    f"layer_is_moe list length ({len(layer_is_moe)}) must equal "
                    f"num_hidden_layers ({num_hidden_layers})"
                )
            layer_is_moe_flags = []
            for i, v in enumerate(layer_is_moe):
                if v == "moe" or v is True:
                    layer_is_moe_flags.append(True)
                elif v == "full" or v is False:
                    layer_is_moe_flags.append(False)
                else:
                    raise ValueError(
                        f"layer_is_moe[{i}] must be 'moe' or 'full', got '{v}'"
                    )
        else:
            raise TypeError(f"layer_is_moe must be a str or list, got {type(layer_is_moe)}")

        # Decoder layers
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.layers.append(DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                layer_idx=i,
                head_dim=head_dim,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                rms_norm_eps=rms_norm_eps,
                use_gated_attention=use_gated_attention,
                norm_zero_init=norm_zero_init,
                prenorm=prenorm,
                use_moe=layer_is_moe_flags[i],
                intermediate_size=intermediate_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                norm_topk_prob=norm_topk_prob,
                hidden_act=hidden_act,
            ))

        # Final norm
        if last_norm == "gated":
            self.norm = RMSNormGated(hidden_size, rms_norm_eps)
            self.norm_gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        elif last_norm == "normal":
            self.norm = _make_norm(hidden_size, rms_norm_eps, norm_zero_init)
        else:  # "none"
            self.norm = nn.Identity()

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim=effective_head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Experts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, TopKRouter):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
    ) -> Tensor:
        """
        Args:
            inputs_embeds: [B, S, D] input embeddings.
            attention_mask: optional attention mask.
            position_ids: optional position ids.
            past_key_values: optional KV cache.

        Returns:
            last_hidden_state: [B, S, D]
        """
        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            past_len = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            position_ids = torch.arange(seq_len, device=inputs_embeds.device) + past_len
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        if self.last_norm_type == "gated":
            gate = self.norm_gate_proj(hidden_states)
            hidden_states = self.norm(hidden_states, gate)
        elif self.last_norm_type == "normal":
            hidden_states = self.norm(hidden_states)
        # else: "none" — identity, no-op
        return hidden_states

# ---------------------------------------------------------------------------
# Convenience: create configs matching known architectures
# ---------------------------------------------------------------------------

def make_qwen3_moe_config(
    num_hidden_layers: int = 24,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 4,
    intermediate_size: int = 8192,
    moe_intermediate_size: int = 1408,
    num_experts: int = 64,
    num_experts_per_tok: int = 8,
    rope_theta: float = 1000000.0,
    decoder_sparse_step: int = 1,
    mlp_only_layers: List[int] | None = None,
    **kwargs,
) -> dict:
    """
    Return a kwargs dict for TransformerModel matching Qwen3Moe architecture.

    Qwen3Moe uses:
      - Non-gated attention
      - Standard RMSNorm (ones init)
      - Mixed MoE/dense layers controlled by decoder_sparse_step and mlp_only_layers
    """
    if mlp_only_layers is None:
        mlp_only_layers = []

    layer_is_moe = []
    for i in range(num_hidden_layers):
        is_moe = (i not in mlp_only_layers) and (num_experts > 0 and (i + 1) % decoder_sparse_step == 0)
        layer_is_moe.append("moe" if is_moe else "full")

    return dict(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_gated_attention=False,
        norm_zero_init=False,
        prenorm=True,
        layer_is_moe=layer_is_moe,
        rope_theta=rope_theta,
        **kwargs,
    )

def make_qwen3_5_moe_config(
    num_hidden_layers: int = 40,
    hidden_size: int = 2048,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 2,
    head_dim: int = 256,
    moe_intermediate_size: int = 512,
    num_experts: int = 256,
    num_experts_per_tok: int = 8,
    rope_theta: float = 10000000.0,
    **kwargs,
) -> dict:
    """
    Return a kwargs dict for TransformerModel matching Qwen3_5Moe architecture.

    Qwen3_5Moe uses:
      - Gated attention (sigmoid gate on attn output)
      - Zero-init RMSNorm
      - All MoE layers (no dense MLP)
    """
    return dict(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        use_gated_attention=True,
        norm_zero_init=True,
        prenorm=True,
        layer_is_moe="moe",
        rope_theta=rope_theta,
        **kwargs,
    )

def make_qwen3_5_config(
    num_hidden_layers: int = 64,
    hidden_size: int = 5120,
    num_attention_heads: int = 24,
    num_key_value_heads: int = 4,
    head_dim: int = 256,
    intermediate_size: int = 17408,
    rope_theta: float = 10000000.0,
    **kwargs,
) -> dict:
    """
    Return a kwargs dict for TransformerModel matching Qwen3_5 (non-MoE) architecture.

    Qwen3_5 uses:
      - Gated attention (sigmoid gate on attn output)
      - Zero-init RMSNorm
      - All dense MLP layers (no MoE)
    """
    return dict(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        use_gated_attention=True,
        norm_zero_init=True,
        prenorm=True,
        layer_is_moe="full",
        rope_theta=rope_theta,
        **kwargs,
    )

__all__ = [
    "LLM_MLP",
    "Attention",
    "DecoderLayer",
    "TransformerModel",
    "SparseMoeBlock",
    "TopKRouter",
    "Experts",
    "RMSNorm",
    "RMSNormGated",
    "RMSNormZeroInit",
    "RotaryEmbedding",
    "make_qwen3_moe_config",
    "make_qwen3_5_moe_config",
    "make_qwen3_5_config",
]
