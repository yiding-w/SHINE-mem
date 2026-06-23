"""
Compressed MLP W-Transform.

Implements the Compress-MLP-Decompress architecture:
    z = L^T @ W @ R           -> [B, k, k]
    z_tilde = z + MLP(flatten(z))   -> [B, k, k]  (residual)
    W_tilde = W + L @ z_tilde @ R^T

Optional enhancements:
  - Asymmetric bases (scheme B): Use separate L_enc/R_enc for compression
    and L_dec/R_dec for decompression, allowing the compress and decompress
    directions to be independently optimized.
  - FiLM Conditioning (scheme E): Compute global statistics of W (mean, std,
    norm) and use them to modulate z via learned scale/shift (FiLM) before
    feeding into the MLP. Stats are log1p-normalized for numerical stability.
  - Cross-Projection Attention (scheme H): After each projection's z_tilde
    is computed, stack all projections' z_tilde within the same layer and
    apply a lightweight multi-head attention to allow information exchange
    between projections (e.g. q/k coordination). Two-pass design: first
    pass computes all z_tilde, then cross-attn, then second pass decompresses.

In TP mode
In TP mode, W is pre-sliced (Colwise: output dim sharded, Rowwise: input
dim sharded). The compression step computes a partial z on each rank and
uses all-reduce(SUM) to obtain the full z. The MLP runs on the full z
(identical on all ranks). The decompression step produces only the local
shard of delta_W (no communication needed).

Each layer has its own independent set of CompressMLP instances (one per
projection type), so different layers can learn different transforms.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)

# TP sharding plan (must match full_tp.py)
_COLWISE_PROJS = frozenset({"q_query", "q_gate", "k", "v", "gate", "up"})
_ROWWISE_PROJS = frozenset({"o", "down"})

# Supported statistics for FiLM conditioning
_SUPPORTED_STATS = frozenset({"mean", "std", "norm", "max", "min"})


class _AllReduceSumForward(torch.autograd.Function):
    """All-reduce SUM in forward; identity in backward.

    Used for the compress step in TP mode: each rank computes a partial z,
    then all-reduce SUM produces the full z on every rank. In backward,
    since z_full = sum(z_partial_i), dL/dz_partial_i = dL/dz_full (identity).
    """

    @staticmethod
    def forward(ctx, z_partial, tp_group):
        ctx.tp_group = tp_group
        z_full = z_partial.clone()
        dist.all_reduce(z_full, op=dist.ReduceOp.SUM, group=tp_group)
        return z_full

    @staticmethod
    def backward(ctx, grad_output):
        # Identity backward: each rank's partial contribution to the sum
        # receives the full upstream gradient.
        return grad_output, None


@torch.compiler.disable
def _all_reduce_sum_forward(z: torch.Tensor, tp_group) -> torch.Tensor:
    """Public wrapper around _AllReduceSumForward.apply (compiler-opaque)."""
    return _AllReduceSumForward.apply(z, tp_group)


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {name}")


class CrossProjectionAttn(nn.Module):
    """Lightweight multi-head self-attention across projections within a layer.

    Follows the same attention style as m2p_transformer.py:
      - Separate Q/K/V projections (no bias)
      - Scaled dot-product attention with float32 softmax for numerical stability
      - F.dropout for training-time dropout
      - Output projection with zero-init for clean residual at start

    Input: Z [B, num_projs, k²] — stacked z_tilde from all projections.
    Output: Z' [B, num_projs, k²] — z_tilde after cross-projection interaction.

    Since num_projs is small (typically 7), this is extremely cheap.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} not divisible by num_heads={num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = dropout

        # Q, K, V projections (no bias, matching m2p_transformer style)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Zero-init output projection for clean residual at start
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Apply cross-projection attention with residual.

        Args:
            Z: [B, num_projs, d_model]

        Returns:
            Z + Attn(Z): [B, num_projs, d_model]
        """
        B, N, D = Z.shape  # N = num_projs, D = k²
        hidden_shape = (B, N, self.num_heads, self.head_dim)

        # Q, K, V: [B, N, D] -> [B, num_heads, N, head_dim]
        query = self.q_proj(Z).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(Z).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(Z).view(hidden_shape).transpose(1, 2)

        # Scaled dot-product attention (float32 softmax for stability)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention: [B, num_heads, N, head_dim]
        attn_output = torch.matmul(attn_weights, value)

        # Reshape back: [B, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, N, -1)
        attn_output = self.o_proj(attn_output)

        # Residual connection
        return Z + attn_output


class CompressMLP(nn.Module):
    """Single projection Compress-MLP-Decompress transform.

    Supports both PP mode (full W) and TP mode (sharded W).
    In TP mode, L and R are stored at FULL dimensions. During forward,
    the appropriate local slice of L or R is used for compression, and
    all-reduce(SUM) is performed to obtain the full compressed z.

    Optional enhancements (controlled by constructor args):
      - Asymmetric bases: use separate L_enc/R_enc and L_dec/R_dec.
      - FiLM conditioning: modulate z with scale/shift derived from W's stats
        (log1p-normalized for numerical stability).
    """

    def __init__(
        self,
        d_in_full: int,
        d_out_full: int,
        k: int = 16,
        mlp_ratio: int = 4,
        activation: str = "gelu",
        tp_mode: bool = False,
        tp_rank: int = 0,
        tp_world: int = 1,
        tp_group=None,
        # --- Enhancement B: Asymmetric Bases ---
        asymmetric: bool = False,
        # --- Enhancement E: FiLM Conditioning ---
        conditioning: str = "none",  # "none" | "film" | "concat"
        cond_stats: Optional[List[str]] = None,  # e.g. ["mean", "std", "norm"]
    ):
        super().__init__()
        self.d_in_full = d_in_full
        self.d_out_full = d_out_full
        self.k = k
        self.tp_mode = tp_mode
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.tp_group = tp_group

        # --- Enhancement config ---
        self.asymmetric = asymmetric
        self.conditioning = conditioning.lower() if conditioning else "none"
        self.cond_stats = cond_stats or []

        # Validate stats
        for s in self.cond_stats:
            if s not in _SUPPORTED_STATS:
                raise ValueError(f"Unknown stat '{s}'. Supported: {_SUPPORTED_STATS}")

        # L: [d_in_full, k], R: [d_out_full, k] -- always full dimensions
        # In asymmetric mode, these serve as L_enc / R_enc (compression only)
        self.L = nn.Parameter(torch.randn(d_in_full, k) * (1.0 / math.sqrt(d_in_full)))
        self.R = nn.Parameter(torch.randn(d_out_full, k) * (1.0 / math.sqrt(d_out_full)))

        # --- Enhancement B: Asymmetric Bases ---
        if self.asymmetric:
            # Separate L_dec, R_dec for decompression
            self.L_dec = nn.Parameter(torch.randn(d_in_full, k) * (1.0 / math.sqrt(d_in_full)))
            self.R_dec = nn.Parameter(torch.randn(d_out_full, k) * (1.0 / math.sqrt(d_out_full)))

        # --- Enhancement E: FiLM Conditioning layers ---
        d_stats = len(self.cond_stats)  # number of scalar stats per batch element
        if self.conditioning == "film" and d_stats > 0:
            # FiLM: stats -> (gamma, beta) for modulating z_flat
            self.film_net = nn.Sequential(
                nn.Linear(d_stats, k * k),
                _get_activation(activation),
                nn.Linear(k * k, 2 * k * k),  # output: [gamma; beta]
            )
            # Init to identity modulation: gamma=1, beta=0
            nn.init.zeros_(self.film_net[-1].weight)
            # bias: first k*k = 1 (gamma), last k*k = 0 (beta)
            with torch.no_grad():
                self.film_net[-1].bias[:k * k].fill_(1.0)
                self.film_net[-1].bias[k * k:].fill_(0.0)
            mlp_input_dim = k * k
        elif self.conditioning == "concat" and d_stats > 0:
            # Concat: append stats to z_flat as MLP input
            mlp_input_dim = k * k + d_stats
        else:
            # No conditioning
            mlp_input_dim = k * k

        # MLP: mlp_input_dim -> hidden -> k*k
        hidden_dim = k * k * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim, bias=True),
            _get_activation(activation),
            nn.Linear(hidden_dim, k * k, bias=True),
        )
        # Zero-init last layer for clean residual at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)



    def _compute_stats(self, W: torch.Tensor) -> torch.Tensor:
        """Compute global statistics of W for conditioning.

        Args:
            W: [B, d_in, d_out] (possibly TP-sharded local slice)

        Returns:
            stats: [B, d_stats] tensor of scalar statistics per batch element.
        """
        B = W.shape[0]
        stats_list = []
        # Flatten spatial dims for per-batch-element stats
        W_flat = W.reshape(B, -1)  # [B, d_in * d_out]

        for stat_name in self.cond_stats:
            if stat_name == "mean":
                stats_list.append(W_flat.mean(dim=1, keepdim=True))
            elif stat_name == "std":
                stats_list.append(W_flat.std(dim=1, keepdim=True))
            elif stat_name == "norm":
                stats_list.append(W_flat.norm(dim=1, keepdim=True))
            elif stat_name == "max":
                stats_list.append(W_flat.max(dim=1, keepdim=True).values)
            elif stat_name == "min":
                stats_list.append(W_flat.min(dim=1, keepdim=True).values)

        stats = torch.cat(stats_list, dim=1)  # [B, d_stats]
        # Log1p normalization for numerical stability:
        # Compresses large values (e.g. norm~100) to ~4.6 while preserving
        # small values and sign information. Prevents FiLM net from receiving
        # inputs with wildly different magnitudes.
        stats = torch.sign(stats) * torch.log1p(torch.abs(stats))
        return stats

    def _all_reduce_z(self, z: torch.Tensor) -> torch.Tensor:
        """All-reduce z across TP group using autograd-safe Function.

        Uses _AllReduceSumForward which does all-reduce SUM in forward and
        identity in backward, correctly supporting gradient computation
        through the collective operation.
        """
        return _all_reduce_sum_forward(z, self.tp_group)

    def compress_and_mlp(self, W: torch.Tensor, proj_type: str = "full") -> torch.Tensor:
        """Phase 1: Compress W and run MLP to get z_tilde.

        Used by cross-projection attention mode (two-pass). Returns z_tilde
        which can be further processed by cross-attention before decompression.

        Args:
            W: Weight matrix (same as forward()).
            proj_type: "full", "colwise", or "rowwise".

        Returns:
            z_tilde: [B, k, k] — compressed and MLP-transformed representation.
        """
        B = W.shape[0]

        # --- Compute conditioning stats (before compression) ---
        if self.conditioning != "none" and len(self.cond_stats) > 0:
            stats = self._compute_stats(W)  # [B, d_stats]

        # --- Compress: compute z = L^T @ W @ R -> [B, k, k] ---
        if proj_type == "full":
            z = torch.einsum('ik,bio,oj->bkj', self.L, W, self.R)
        elif proj_type == "colwise":
            d_out_local = W.shape[2]
            s = self.tp_rank * d_out_local
            e = s + d_out_local
            R_local = self.R[s:e, :]
            z = torch.einsum('ik,bio,oj->bkj', self.L, W, R_local)
            if self.tp_world > 1:
                z = self._all_reduce_z(z)
        elif proj_type == "rowwise":
            d_in_local = W.shape[1]
            s = self.tp_rank * d_in_local
            e = s + d_in_local
            L_local = self.L[s:e, :]
            z = torch.einsum('ik,bio,oj->bkj', L_local, W, self.R)
            if self.tp_world > 1:
                z = self._all_reduce_z(z)
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        # --- MLP with residual (+ optional conditioning) ---
        z_flat = z.reshape(B, -1)  # [B, k*k]

        if self.conditioning == "film" and len(self.cond_stats) > 0:
            film_params = self.film_net(stats)
            gamma = film_params[:, :self.k * self.k]
            beta = film_params[:, self.k * self.k:]
            z_modulated = gamma * z_flat + beta
            mlp_input = z_modulated
        elif self.conditioning == "concat" and len(self.cond_stats) > 0:
            mlp_input = torch.cat([z_flat, stats], dim=1)
        else:
            mlp_input = z_flat

        z_tilde = z + self.mlp(mlp_input).reshape(B, self.k, self.k)  # [B, k, k]
        return z_tilde

    def decompress(self, W: torch.Tensor, z_tilde: torch.Tensor, proj_type: str = "full") -> torch.Tensor:
        """Phase 2: Decompress z_tilde back to W space and produce W_tilde.

        Used by cross-projection attention mode (two-pass). Takes z_tilde
        (possibly modified by cross-attention) and produces the final W_tilde.

        Args:
            W: Original weight matrix (for residual and diagonal branch).
            z_tilde: [B, k, k] — from compress_and_mlp() or after cross-attn.
            proj_type: "full", "colwise", or "rowwise".

        Returns:
            W_tilde: Same shape as W.
        """
        # --- Decompress: delta_W = L_dec @ z_tilde @ R_dec^T ---
        L_dec = self.L_dec if self.asymmetric else self.L
        R_dec = self.R_dec if self.asymmetric else self.R

        if proj_type == "full":
            delta_W = torch.einsum('ik,bkj,oj->bio', L_dec, z_tilde, R_dec)
        elif proj_type == "colwise":
            d_out_local = W.shape[2]
            s = self.tp_rank * d_out_local
            e = s + d_out_local
            R_dec_local = R_dec[s:e, :]
            delta_W = torch.einsum('ik,bkj,oj->bio', L_dec, z_tilde, R_dec_local)
        elif proj_type == "rowwise":
            d_in_local = W.shape[1]
            s = self.tp_rank * d_in_local
            e = s + d_in_local
            L_dec_local = L_dec[s:e, :]
            delta_W = torch.einsum('ik,bkj,oj->bio', L_dec_local, z_tilde, R_dec)
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        # --- Residual (low-rank branch) ---
        return W + delta_W

    def forward(self, W: torch.Tensor, proj_type: str = "full") -> torch.Tensor:
        """Transform W (possibly TP-sharded) via Compress-MLP-Decompress.

        This is the standard single-pass forward. When cross_projection_attn
        is enabled, _SingleLayerTransform uses compress_and_mlp() + decompress()
        instead of this method.

        Args:
            W: Weight matrix.
                PP mode (proj_type="full"): [B, d_in_full, d_out_full]
                TP Colwise (proj_type="colwise"): [B, d_in_full, d_out_local]
                TP Rowwise (proj_type="rowwise"): [B, d_in_local, d_out_full]
            proj_type: "full" (PP), "colwise" (TP col-sharded), "rowwise" (TP row-sharded).

        Returns:
            W_tilde: Same shape as W.
        """
        z_tilde = self.compress_and_mlp(W, proj_type)
        return self.decompress(W, z_tilde, proj_type)


class _SingleLayerTransform(nn.Module):
    """Transform for a single LLM layer: one independent CompressMLP per projection.

    When cross_projection_attn is enabled, uses a two-pass approach:
      Pass 1: All projections run compress_and_mlp() to get z_tilde.
      Cross-Attn: Stack all z_tilde and apply CrossProjectionAttn.
      Pass 2: All projections run decompress() with the cross-attended z_tilde.
    """

    def __init__(
        self,
        proj_dims: Dict[str, Tuple[int, int]],
        k: int,
        mlp_ratio: int,
        activation: str,
        tp_mode: bool,
        tp_rank: int,
        tp_world: int,
        tp_group,
        # Enhancement options
        asymmetric: bool = False,
        conditioning: str = "none",
        cond_stats: Optional[List[str]] = None,
        # --- Enhancement H: Cross-Projection Attention ---
        cross_projection_attn: bool = False,
        cross_attn_num_heads: int = 4,
    ):
        super().__init__()
        self.tp_mode = tp_mode
        self.cross_projection_attn = cross_projection_attn

        # One independent CompressMLP per projection key
        self.transforms = nn.ModuleDict()
        for proj_key, (d_in, d_out) in proj_dims.items():
            self.transforms[proj_key] = CompressMLP(
                d_in_full=d_in,
                d_out_full=d_out,
                k=k,
                mlp_ratio=mlp_ratio,
                activation=activation,
                tp_mode=tp_mode,
                tp_rank=tp_rank,
                tp_world=tp_world,
                tp_group=tp_group,
                asymmetric=asymmetric,
                conditioning=conditioning,
                cond_stats=cond_stats,
            )

        # --- Enhancement H: Cross-Projection Attention ---
        if self.cross_projection_attn:
            # Sorted projection keys for deterministic ordering
            self._proj_keys_sorted = sorted(self.transforms.keys())
            self.cross_attn = CrossProjectionAttn(
                d_model=k * k,
                num_heads=cross_attn_num_heads,
            )

    def forward(self, layer_wdict: dict) -> dict:
        """Transform a single layer's wdict."""
        if self.cross_projection_attn:
            return self._forward_cross_attn(layer_wdict)
        else:
            return self._transform_recursive(layer_wdict, path_keys=[])

    def _forward_cross_attn(self, layer_wdict: dict) -> dict:
        """Two-pass forward with cross-projection attention.

        Pass 1: Collect all leaf (W, proj_type) pairs and run compress_and_mlp.
        Cross-Attn: Stack z_tildes, apply CrossProjectionAttn.
        Pass 2: Decompress each projection with the cross-attended z_tilde.
        """
        # --- Pass 1: Collect leaves and compute z_tilde for each ---
        leaves = {}  # proj_name -> {"W": tensor, "C": tensor, "proj_type": str}
        self._collect_leaves(layer_wdict, path_keys=[], leaves=leaves)

        # Compute z_tilde for each projection (in sorted order)
        z_tildes = {}  # proj_name -> [B, k, k]
        for proj_name in self._proj_keys_sorted:
            if proj_name in leaves and proj_name in self.transforms:
                info = leaves[proj_name]
                z_tildes[proj_name] = self.transforms[proj_name].compress_and_mlp(
                    info["W"], proj_type=info["proj_type"]
                )

        # --- Cross-Projection Attention ---
        if len(z_tildes) > 0:
            # Stack: [B, num_projs, k*k]
            proj_names_ordered = [p for p in self._proj_keys_sorted if p in z_tildes]
            B = next(iter(z_tildes.values())).shape[0]
            k = next(iter(z_tildes.values())).shape[1]
            Z = torch.stack(
                [z_tildes[p].reshape(B, -1) for p in proj_names_ordered], dim=1
            )  # [B, num_projs, k²]

            # Apply cross-attention
            Z_cross = self.cross_attn(Z)  # [B, num_projs, k²]

            # Unstack back
            for i, proj_name in enumerate(proj_names_ordered):
                z_tildes[proj_name] = Z_cross[:, i, :].reshape(B, k, k)

        # --- Pass 2: Decompress each projection ---
        result = self._decompress_recursive(layer_wdict, path_keys=[], z_tildes=z_tildes)
        return result

    def _collect_leaves(self, d: dict, path_keys: list, leaves: dict):
        """Recursively collect leaf nodes (W tensors) from wdict."""
        if d is None:
            return
        if "W" in d:
            proj_name = path_keys[-1] if path_keys else ""
            if self.tp_mode:
                if proj_name in _COLWISE_PROJS:
                    proj_type = "colwise"
                elif proj_name in _ROWWISE_PROJS:
                    proj_type = "rowwise"
                else:
                    proj_type = "full"
            else:
                proj_type = "full"
            leaves[proj_name] = {"W": d["W"], "C": d.get("C", None), "proj_type": proj_type}
            return
        for key, value in d.items():
            if isinstance(value, dict):
                self._collect_leaves(value, path_keys + [key], leaves)

    def _decompress_recursive(self, d: dict, path_keys: list, z_tildes: dict) -> dict:
        """Recursively decompress wdict nodes using pre-computed z_tildes."""
        if d is None:
            return None
        if "W" in d:
            proj_name = path_keys[-1] if path_keys else ""
            W = d["W"]
            if self.tp_mode:
                if proj_name in _COLWISE_PROJS:
                    proj_type = "colwise"
                elif proj_name in _ROWWISE_PROJS:
                    proj_type = "rowwise"
                else:
                    proj_type = "full"
            else:
                proj_type = "full"

            if proj_name in self.transforms and proj_name in z_tildes:
                W_tilde = self.transforms[proj_name].decompress(
                    W, z_tildes[proj_name], proj_type=proj_type
                )
            else:
                W_tilde = W
            return {"W": W_tilde, "C": d.get("C", None)}

        result = {}
        for key, value in d.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._decompress_recursive(value, path_keys + [key], z_tildes)
            else:
                result[key] = value
        return result

    def _transform_recursive(self, d: dict, path_keys: list) -> dict:
        """Recursively transform wdict nodes (standard single-pass mode)."""
        if d is None:
            return None

        # Leaf node: has "W" key
        if "W" in d:
            proj_name = path_keys[-1] if path_keys else ""
            W = d["W"]

            # Determine proj_type for TP mode
            if self.tp_mode:
                if proj_name in _COLWISE_PROJS:
                    proj_type = "colwise"
                elif proj_name in _ROWWISE_PROJS:
                    proj_type = "rowwise"
                else:
                    proj_type = "full"
            else:
                proj_type = "full"

            # Look up the CompressMLP by proj_name
            if proj_name in self.transforms:
                W_tilde = self.transforms[proj_name](W, proj_type=proj_type)
            else:
                W_tilde = W

            # C (bias) is NOT transformed
            return {"W": W_tilde, "C": d.get("C", None)}

        # Non-leaf: recurse into sub-dicts
        result = {}
        for key, value in d.items():
            if value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._transform_recursive(value, path_keys + [key])
            else:
                result[key] = value
        return result


class CompressedMLPTransform(nn.Module):
    """Per-layer, per-projection CompressMLP transform.

    Creates an independent CompressMLP for each (layer, projection) pair.
    This allows each layer to learn its own optimal transform, reflecting
    the fact that different layers have different weight distributions and
    functional roles.

    Architecture:
        layers[layer_idx].transforms[proj_key] = CompressMLP(...)

    Enhancement options (from cfg):
      - asymmetric: bool, whether to use separate L_dec/R_dec for decompression.
      - conditioning: "none" | "film" | "concat"
        FiLM/concat conditioning on W's global statistics (log1p-normalized).
      - cond_stats: list of stat names, e.g. ["mean", "std", "norm"]
      - cross_projection_attn: bool, whether to enable cross-projection attention.
      - cross_attn_num_heads: int, number of attention heads for cross-projection.

    Args:
        cfg: Config dict with keys: k, mlp_ratio, activation,
             asymmetric, conditioning, cond_stats,
             cross_projection_attn, cross_attn_num_heads.
        proj_dims: Dict mapping proj_key -> (d_in_full, d_out_full).
        num_layers: Number of LLM layers.
        tp_mode: Whether operating in TP mode.
        tp_rank: TP rank.
        tp_world: TP world size.
        tp_group: TP process group.
    """

    def __init__(
        self,
        cfg: dict,
        proj_dims: Dict[str, Tuple[int, int]],
        num_layers: int,
        tp_mode: bool = False,
        tp_rank: int = 0,
        tp_world: int = 1,
        tp_group=None,
    ):
        super().__init__()
        self.k = cfg.get("k", 16)
        self.mlp_ratio = cfg.get("mlp_ratio", 4)
        self.activation = cfg.get("activation", "gelu")
        self.num_layers = num_layers
        self.tp_mode = tp_mode

        # Enhancement options
        self.asymmetric = cfg.get("asymmetric", False)
        self.conditioning = cfg.get("conditioning", "none")
        self.cond_stats = cfg.get("cond_stats", [])
        self.cross_projection_attn = cfg.get("cross_projection_attn", False)
        self.cross_attn_num_heads = cfg.get("cross_attn_num_heads", 4)

        # Create one _SingleLayerTransform per layer
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_SingleLayerTransform(
                proj_dims=proj_dims,
                k=self.k,
                mlp_ratio=self.mlp_ratio,
                activation=self.activation,
                tp_mode=tp_mode,
                tp_rank=tp_rank,
                tp_world=tp_world,
                tp_group=tp_group,
                asymmetric=self.asymmetric,
                conditioning=self.conditioning,
                cond_stats=self.cond_stats,
                cross_projection_attn=self.cross_projection_attn,
                cross_attn_num_heads=self.cross_attn_num_heads,
            ))

        total_instances = num_layers * len(proj_dims)
        enhancements = []
        if self.asymmetric:
            enhancements.append("asymmetric")
        if self.conditioning != "none":
            enhancements.append(f"conditioning={self.conditioning}({self.cond_stats})")
        if self.cross_projection_attn:
            enhancements.append(f"cross_projection_attn(heads={self.cross_attn_num_heads})")
        enh_str = ", ".join(enhancements) if enhancements else "none"
        logger.info(
            f"[CompressedMLPTransform] Created {total_instances} CompressMLP instances "
            f"({num_layers} layers x {len(proj_dims)} projections), "
            f"k={self.k}, mlp_ratio={self.mlp_ratio}, tp_mode={tp_mode}, "
            f"enhancements=[{enh_str}]"
        )

    def forward(self, layer_wdict: dict, layer_idx: int) -> dict:
        """Transform a single layer's wdict using that layer's dedicated transforms.

        Args:
            layer_wdict: The per-layer wdict (already indexed by layer_idx).
            layer_idx: The layer index, used to select the correct per-layer transform.

        Returns:
            Transformed wdict with the same structure.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            # Layer index out of range (e.g. linear_attention layer with no wdict)
            return layer_wdict
        return self.layers[layer_idx](layer_wdict)
