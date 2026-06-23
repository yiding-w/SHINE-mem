"""
Tensor-parallel hypernetwork.

The PP version (``hypernetwork.py``) distributes the m2p_transformer
layers across pipeline stages and threads gradient-aware send/recv
through every layer transition. Under TP every rank holds the **full**
m2p_transformer with identical parameters and identical inputs
(memory_states is replicated across the TP group after the LLM's
o_proj all-reduce), so the m2p_transformer forward is just a plain
local forward — no collectives.

This module exposes ``TPHypernetwork``, which provides:

  * ``self.m2p_transformer`` — a fresh ``TransformerModel`` built from
    ``m2p_transformer_cfg.init``.
  * ``self.layer_pos_emb`` and ``self.token_pos_emb`` — learnable 2D
    positional embeddings added once at the input.
  * ``forward(memory_states)`` — applies the memory_method reshape,
    adds positional embeddings, runs every m2p layer (alternating
    horizontal / vertical attention per ``layer_types``), and returns
    the processed memory_states.
  * ``state_dict`` / ``load_state_dict`` work directly on this single
    nn.Module — no per-stage stitching.

Initialisation may differ across ranks (no global seed is set before
constructing). The training loop broadcasts from TP rank 0 after
construction to ensure all TP ranks hold identical params. DP grad
sync (AVG across DP replicas) is done at the training-loop level
after backward.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from hypernetwork.m2p_transformer import TransformerModel
from utils.myparallel import is_main_process_per_node

logger = logging.getLogger(__name__)


__all__ = ["TPHypernetwork"]


def _resolve_m2p_init_cfg(m2p_transformer_cfg):
    """Pull the ``init`` sub-dict out of the Hydra-style m2p config."""
    from omegaconf import OmegaConf
    if hasattr(m2p_transformer_cfg, "_metadata"):
        full_cfg = OmegaConf.to_container(m2p_transformer_cfg, resolve=True)
    elif isinstance(m2p_transformer_cfg, dict):
        full_cfg = dict(m2p_transformer_cfg)
    else:
        full_cfg = dict(m2p_transformer_cfg)
    init_cfg = full_cfg.get("init", full_cfg)
    return full_cfg, init_cfg


class TPHypernetwork(nn.Module):
    """Replicated m2p_transformer + 2D positional embeddings.

    Replaces the PP version in ``hypernetwork.py``. Every TP rank holds
    the full m2p_transformer (same params, same forward, same output);
    DP replicas may differ until grad sync brings them back in line.
    """

    def __init__(
        self,
        m2p_transformer_cfg,
        num_llm_layers: int,
        num_full_attn_layers: int,
        num_mem_token: int,
        memory_method: str = "only_full_1for1",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if memory_method not in ("only_full_4for1", "only_full_1for1"):
            raise ValueError(
                f"Unsupported memory_method '{memory_method}'. "
                f"Supported: 'only_full_4for1', 'only_full_1for1'."
            )
        self._memory_method = memory_method
        self._num_llm_layers = num_llm_layers
        self._num_full_attn_layers = num_full_attn_layers
        self._num_mem_token = num_mem_token
        self._dtype = dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        full_cfg, init_cfg = _resolve_m2p_init_cfg(m2p_transformer_cfg)
        self._init_cfg = init_cfg
        self._layer_types = full_cfg.get("layer_types", None) or init_cfg.get("layer_types", None)
        if self._layer_types is None:
            self._layer_types = ["h"] * init_cfg.get("num_hidden_layers", 8)

        # ---- Build m2p_transformer (replicated, no PP) ----
        # TransformerModel.__init__ accepts keyword args that overlap with
        # init_cfg. Forward only the keys it recognises.
        from inspect import signature
        tm_keys = set(signature(TransformerModel.__init__).parameters.keys())
        tm_kwargs = {k: v for k, v in init_cfg.items() if k in tm_keys}
        self.m2p_transformer = TransformerModel(**tm_kwargs).to(device=device, dtype=dtype)

        hidden_size = self.m2p_transformer.hidden_size
        self._hidden_size = hidden_size

        # ---- Effective L / M after memory_method reshape ----
        if memory_method == "only_full_4for1":
            if num_llm_layers % 4 != 0:
                raise ValueError(
                    f"num_llm_layers ({num_llm_layers}) must be divisible by 4 for only_full_4for1"
                )
            self._effective_L = num_llm_layers // 4
            self._effective_M = num_mem_token * 4
        else:  # only_full_1for1
            self._effective_L = num_full_attn_layers
            self._effective_M = num_mem_token

        # ---- 2D positional embeddings (zero-init) ----
        self.layer_pos_emb = nn.Parameter(
            torch.zeros(self._effective_L, 1, hidden_size, device=device, dtype=dtype)
        )
        self.token_pos_emb = nn.Parameter(
            torch.zeros(1, self._effective_M, hidden_size, device=device, dtype=dtype)
        )

        if is_main_process_per_node():
            logger.info(
                f"[TPHypernetwork] effective_L={self._effective_L} effective_M={self._effective_M} "
                f"hidden={hidden_size} num_m2p_layers={len(self.m2p_transformer.layers)} "
                f"layer_types={self._layer_types}"
            )

    # ------------------------------------------------------------------
    # Reshape: input memory_states → (B, effective_L, effective_M, H)
    # ------------------------------------------------------------------

    def _initial_reshape(self, memory_states: Tensor) -> Tensor:
        if self._memory_method == "only_full_4for1":
            B, L, M, H = memory_states.shape
            if L % 4 != 0:
                raise ValueError(f"L dimension ({L}) must be divisible by 4 for only_full_4for1")
            L4 = L // 4
            hs = memory_states.reshape(B, L4, 4, M, H)
            hs = hs.transpose(2, 3)
            hs = hs.reshape(B, L4, M * 4, H)
            return hs
        else:  # only_full_1for1
            return memory_states

    # ------------------------------------------------------------------
    # Per-layer h/v forward (no PP comm)
    # ------------------------------------------------------------------

    def _h_layer_forward(self, m2p_layer, memory_states: Tensor) -> Tensor:
        """Horizontal attention: attend over ``effective_M`` tokens within
        a layer group. Reshape ``(B, L, M, H) → (B*L, M, H)`` so the
        attention sees ``M`` as its seq dim.
        """
        B, L, M, H = memory_states.shape
        hs = memory_states.reshape(B * L, M, H)
        h_pos_ids = torch.arange(M, device=hs.device).unsqueeze(0)
        h_pos_emb = self.m2p_transformer.rotary_emb(hs, h_pos_ids)
        hs = m2p_layer(hs, position_embeddings=h_pos_emb, attention_mask=None)
        return hs.reshape(B, L, M, H)

    def _v_layer_forward(self, m2p_layer, memory_states: Tensor) -> Tensor:
        """Vertical attention: attend over ``effective_L`` layers within a
        token group. Transpose then reshape ``(B, L, M, H) →
        (B*M, L, H)`` so the attention sees ``L`` as its seq dim.
        """
        B, L, M, H = memory_states.shape
        hs = memory_states.transpose(1, 2).reshape(B * M, L, H)
        v_pos_ids = torch.arange(L, device=hs.device).unsqueeze(0)
        v_pos_emb = self.m2p_transformer.rotary_emb(hs, v_pos_ids)
        hs = m2p_layer(hs, position_embeddings=v_pos_emb, attention_mask=None)
        return hs.reshape(B, M, L, H).transpose(1, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, memory_states: Tensor) -> Tensor:
        """Run the m2p_transformer over ``memory_states``.

        Args:
            memory_states: ``(B, L_input, M, H)`` where ``L_input`` is
                ``num_llm_layers`` for ``only_full_4for1`` or
                ``num_full_attn_layers`` for ``only_full_1for1``.

        Returns:
            Processed memory_states with shape ``(B, effective_L, effective_M, H)``.
        """
        hs = self._initial_reshape(memory_states)
        hs = hs + self.token_pos_emb + self.layer_pos_emb

        for layer_idx, m2p_layer in enumerate(self.m2p_transformer.layers):
            kind = self._layer_types[layer_idx]
            if kind == "h":
                hs = self._h_layer_forward(m2p_layer, hs)
            elif kind == "v":
                hs = self._v_layer_forward(m2p_layer, hs)
            else:
                raise ValueError(f"Unknown m2p layer_type '{kind}' at index {layer_idx}")

        # Final norm (gated or normal) — m2p_transformer.forward handles
        # this for the standalone case; we reimplement here because we
        # already short-circuited through the layer loop.
        last_norm_type = self.m2p_transformer.last_norm_type
        if last_norm_type == "gated":
            # RMSNormGated expects (x, gate). The gate is produced by
            # norm_gate_proj(x) — but only when norm_zero_init makes the
            # norm output zero this would zero the model. Our use case
            # follows the production code, which applies norm + gate.
            B, L, M, H = hs.shape
            hs_flat = hs.reshape(B * L, M, H)
            gate = self.m2p_transformer.norm_gate_proj(hs_flat)
            hs_flat = self.m2p_transformer.norm(hs_flat, gate)
            hs = hs_flat.reshape(B, L, M, H)
        elif last_norm_type == "normal":
            hs = self.m2p_transformer.norm(hs)
        # else "none" — Identity, no-op

        return hs
