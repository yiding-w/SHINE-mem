import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import logging
from typing import Optional

from utils.mymodel import (
    load_m2p_transformer_for_pipeline,
    build_layer_stage_mapping,
    get_extra_component_stages,
)
from utils.myparallel import (
    get_pipeline_config,
    pipeline_recv,
    pipeline_send,
    PipelineRecv,
    pipeline_send_with_grad,
    is_main_process_per_node,
)

logger = logging.getLogger(__name__)


class Hypernetwork(nn.Module):
    """
    Standalone hypernetwork module that holds all trainable non-LLM parameters
    **and** the complete forward logic (including pipeline-parallel communication).

    This includes:
      - ``m2p_transformer`` — the memory-to-parameter transformer (loaded internally).
      - ``layer_pos_emb``  — learnable positional embedding (effective_L, 1, H).
      - ``token_pos_emb``  — learnable positional embedding (1, effective_M, H).

    Supported memory_method values:
      - ``"only_full_4for1"``: (B, L, M, H) → (B, L/4, M*4, H).
        effective_L = L/4, effective_M = M*4.
      - ``"only_full_1for1"``: (B, L_fa, M, H) → (B, L_fa, M, H).
        Only full_attention layers' memory states are used (no merging).
        effective_L = L_fa, effective_M = M.

    The class is designed to be **self-contained**: given the m2p_transformer
    config and a few external values from the LLM, it loads the transformer,
    computes all pipeline metadata, and builds per-layer forward functions
    internally.

    By keeping every trainable parameter inside this single ``nn.Module``,
    saving / loading a checkpoint is as simple as::

        torch.save(model.hypernetwork.state_dict(), path)
        model.hypernetwork.load_state_dict(torch.load(path))

    Args:
        m2p_transformer_cfg: Hydra DictConfig (or dict) for the m2p_transformer,
            with ``init`` (TransformerModel kwargs), ``device_map``, and
            ``layer_types`` keys.
        num_llm_layers: number of LLM layers (L dimension).
        num_mem_token: number of memory tokens per layer (M dimension).
        dtype: tensor dtype (from the LLM parameters).
    """

    def __init__(
        self,
        m2p_transformer_cfg,
        num_llm_layers: int,
        num_mem_token: int,
        dtype: torch.dtype,
        compile_mode: Optional[str] = None,
        memory_method: str = "only_full_4for1",
        num_full_attn_layers: Optional[int] = None,
        activation_checkpointing: bool = False,
    ):
        super().__init__()

        self._memory_method = memory_method
        self._activation_checkpointing = activation_checkpointing
        if memory_method not in ("only_full_4for1", "only_full_1for1"):
            raise ValueError(
                f"Unsupported memory_method '{memory_method}'. "
                f"Supported: 'only_full_4for1', 'only_full_1for1'."
            )

        if memory_method == "only_full_4for1":
            if num_llm_layers % 4 != 0:
                raise ValueError(
                    f"num_llm_layers ({num_llm_layers}) must be divisible by 4 "
                    f"for the (L/4, M*4) reshape in the hypernetwork."
                )
        elif memory_method == "only_full_1for1":
            if num_full_attn_layers is None:
                raise ValueError(
                    "num_full_attn_layers must be provided when memory_method='only_full_1for1'."
                )
            self._num_full_attn_layers = num_full_attn_layers

        # ---- Load m2p_transformer with pipeline parallel ----
        self.m2p_transformer = load_m2p_transformer_for_pipeline(
            m2p_transformer_cfg=m2p_transformer_cfg,
            dtype=torch.bfloat16,
            freeze=False,
            compile_mode=compile_mode,
        )

        # ---- Parse config ----
        from omegaconf import OmegaConf
        if hasattr(m2p_transformer_cfg, "_metadata"):
            full_cfg = OmegaConf.to_container(m2p_transformer_cfg, resolve=True)
        elif isinstance(m2p_transformer_cfg, dict):
            full_cfg = dict(m2p_transformer_cfg)
        else:
            full_cfg = dict(m2p_transformer_cfg)
        init_cfg = full_cfg.get("init", full_cfg)
        self._m2p_init_cfg = init_cfg

        hidden_size = self.m2p_transformer.hidden_size

        # Store config values for validation and reshape
        self._num_llm_layers = num_llm_layers
        self._num_mem_token = num_mem_token
        self._hidden_size = hidden_size

        # Compute the effective L and M dimensions after reshape
        if memory_method == "only_full_4for1":
            self._effective_L = num_llm_layers // 4
            self._effective_M = num_mem_token * 4
        elif memory_method == "only_full_1for1":
            self._effective_L = num_full_attn_layers
            self._effective_M = num_mem_token

        # ---- Pipeline metadata (computed internally) ----
        parallel_cfg = get_pipeline_config()
        self._my_stage = parallel_cfg["stage"]
        self._my_device = parallel_cfg["device"]
        self._dtype = dtype

        device_map_cfg = full_cfg.get("device_map", None)
        if device_map_cfg is not None:
            self._m2p_layer_to_stage = build_layer_stage_mapping(device_map_cfg)
            m2p_extra_stages = get_extra_component_stages(device_map_cfg)
            self._m2p_num_layers = max(self._m2p_layer_to_stage.keys()) + 1

            # Stage transitions
            self._m2p_stage_transitions = []
            for i in range(self._m2p_num_layers - 1):
                src = self._m2p_layer_to_stage[i]
                dst = self._m2p_layer_to_stage[i + 1]
                if src != dst:
                    self._m2p_stage_transitions.append((i, src, dst))

            # Norm and rotary stages
            self._m2p_norm_stage = m2p_extra_stages.get(
                "norm", self._m2p_layer_to_stage[self._m2p_num_layers - 1]
            )
            self._m2p_rotary_stage = m2p_extra_stages.get(
                "rotary_emb", self._m2p_layer_to_stage[self._m2p_num_layers - 1]
            )

            # mem_gather_target_stage = stage that owns the first m2p layer
            self._mem_gather_target_stage = self._m2p_layer_to_stage[
                min(self._m2p_layer_to_stage.keys())
            ]
        else:
            # No device map — single stage
            n_layers = init_cfg.get("num_hidden_layers", 8)
            self._m2p_layer_to_stage = {i: self._my_stage for i in range(n_layers)}
            self._m2p_num_layers = n_layers
            self._m2p_stage_transitions = []
            self._m2p_norm_stage = self._my_stage
            self._m2p_rotary_stage = self._my_stage
            self._mem_gather_target_stage = self._my_stage

        if is_main_process_per_node():
            logger.info(
                f"[Hypernetwork] Pipeline metadata: "
                f"num_layers={self._m2p_num_layers}, "
                f"my_stage={self._my_stage}, "
                f"norm@{self._m2p_norm_stage}, "
                f"rotary@{self._m2p_rotary_stage}, "
                f"mem_gather_target@{self._mem_gather_target_stage}, "
                f"transitions={self._m2p_stage_transitions}"
            )

        # ---- Ensure m2p_transformer rotary_emb is available on this stage ----
        # The pipeline loader only materialises rotary_emb on the stage that
        # owns it (according to device_map.extra.rotary_emb).  All other
        # stages get an empty nn.Module() placeholder with no forward().
        # We only need a functional rotary_emb on stages that own at least
        # one m2p layer (those stages call h_forward / v_forward which
        # invokes rotary_emb).
        _has_m2p_layers = self._my_stage in set(self._m2p_layer_to_stage.values())
        m2p_rotary = self.m2p_transformer.rotary_emb
        m2p_inv_freq = getattr(m2p_rotary, "inv_freq", None)
        need_recreate = _has_m2p_layers and (
            # Case 1: inv_freq exists but is on meta device (skeleton only)
            (m2p_inv_freq is not None and m2p_inv_freq.device.type == "meta")
            # Case 2: empty nn.Module() placeholder (no inv_freq at all)
            or m2p_inv_freq is None
        )
        if need_recreate:
            from hypernetwork.m2p_transformer import RotaryEmbedding
            head_dim = init_cfg.get("head_dim", None)
            if head_dim is None:
                head_dim = init_cfg["hidden_size"] // init_cfg["num_attention_heads"]
            self.m2p_transformer.rotary_emb = RotaryEmbedding(
                head_dim=head_dim,
                max_position_embeddings=init_cfg.get("max_position_embeddings", 2048),
                rope_theta=init_cfg.get("rope_theta", 10000.0),
                device=self._my_device,
            )
            if is_main_process_per_node():
                logger.info(
                    f"[Hypernetwork] Re-created m2p_transformer rotary_emb "
                    f"on stage {self._my_stage} (device={self._my_device})"
                )

        # ---- Learnable 2D positional embeddings ----
        # Shape depends on memory_method:
        #   only_full_4for1: (B, L, M, H) → (B, L/4, M*4, H)
        #     layer_pos_emb: (L/4, 1, H), token_pos_emb: (1, M*4, H)
        #   only_full_1for1: (B, L_fa, M, H) → (B, L_fa, M, H)
        #     layer_pos_emb: (L_fa, 1, H), token_pos_emb: (1, M, H)
        # Zero-init so the model starts as if no positional bias is added.
        # Place on the same device as m2p_transformer layer 0 (the first m2p layer).
        _first_m2p_stage = self._m2p_layer_to_stage[min(self._m2p_layer_to_stage.keys())]
        _pos_emb_device = torch.device(f"cuda:{_first_m2p_stage}")
        self.layer_pos_emb = nn.Parameter(
            torch.zeros(self._effective_L, 1, hidden_size, device=_pos_emb_device, dtype=dtype)
        )
        self.token_pos_emb = nn.Parameter(
            torch.zeros(1, self._effective_M, hidden_size, device=_pos_emb_device, dtype=dtype)
        )

        # ---- Parse layer_types and build per-layer forward callables ----
        layer_types = full_cfg.get("layer_types", None) or init_cfg.get("layer_types", None)
        if layer_types is None:
            layer_types = ["h"] * self._m2p_num_layers
            if is_main_process_per_node():
                logger.warning(
                    f"[Hypernetwork] No layer_types in config, "
                    f"defaulting to all 'h' (horizontal attention)."
                )
        self._build_layer_forwards(layer_types)

        # Backward anchors for pipeline_backward() — populated by
        # _run_m2p_layers_with_grad, consumed by ModelHypernetwork.pipeline_backward.
        self._backward_anchors = []

    # ------------------------------------------------------------------
    # Init-time: build per-layer forward functions
    # ------------------------------------------------------------------

    def _build_layer_forwards(self, layer_types: list):
        """
        Build a list of per-layer forward callables from layer_types config.

        Supported layer types:
          - ``"h"`` (horizontal): attend over M*4 tokens.
          - ``"v"`` (vertical): attend over L/4 tokens.

        The bound callables are stored in ``self._m2p_layer_forwards``.
        """
        if len(layer_types) != self._m2p_num_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) != "
                f"m2p_num_layers ({self._m2p_num_layers})."
            )

        self._layer_types = layer_types
        self._m2p_layer_forwards = []

        for i, lt in enumerate(layer_types):
            m2p_layer = self.m2p_transformer.layers[i]
            if lt == "h":
                self._m2p_layer_forwards.append(self._make_h_forward(i, m2p_layer))
            elif lt == "v":
                self._m2p_layer_forwards.append(self._make_v_forward(i, m2p_layer))
            else:
                raise ValueError(
                    f"Unknown m2p_transformer layer_type '{lt}' at index {i}. "
                    f"Supported types: 'h' (horizontal), 'v' (vertical)."
                )

        if is_main_process_per_node():
            logger.info(
                f"[Hypernetwork] m2p_transformer layer_types: {layer_types}"
            )

    # ------------------------------------------------------------------
    # Init-time: create per-layer forward closures
    # ------------------------------------------------------------------

    def _make_h_forward(self, layer_idx: int, m2p_layer):
        """
        Return a callable for horizontal attention (attend over M*4 tokens).

        Input:  (B, L/4, M*4, H)
        Reshape to (B*L/4, M*4, H), run decoder layer, reshape back.
        Output: (B, L/4, M*4, H)

        Note: Position embeddings (token_pos_emb + layer_pos_emb) are added
        once before the first m2p layer in forward_no_grad / forward_with_grad,
        not repeated in every layer.
        """
        def h_forward(
            memory_states: torch.Tensor,
            position_embeddings: tuple,
            attention_mask: Optional[torch.Tensor],
        ) -> torch.Tensor:
            B, L4, M4, H = memory_states.shape

            # (B, L/4, M*4, H) → (B*L/4, M*4, H)
            hs = memory_states.reshape(B * L4, M4, H)

            # Generate position_embeddings for horizontal attention.
            # After reshape the sequence dimension is M*4 (tokens within a layer group).
            h_pos_ids = torch.arange(M4, device=hs.device).unsqueeze(0)
            h_pos_emb = self.m2p_transformer.rotary_emb(hs, h_pos_ids)

            hs = m2p_layer(
                hs,
                position_embeddings=h_pos_emb,
                attention_mask=attention_mask,
            )

            # (B*L/4, M*4, H) → (B, L/4, M*4, H)
            hs = hs.reshape(B, L4, M4, H)
            return hs
        return h_forward

    def _make_v_forward(self, layer_idx: int, m2p_layer):
        """
        Return a callable for vertical attention (attend over L/4 tokens).

        Input:  (B, L/4, M*4, H)
        Transpose to (B, M*4, L/4, H), reshape to (B*M*4, L/4, H),
        run decoder layer, reshape and transpose back.
        Output: (B, L/4, M*4, H)

        Note: Position embeddings (token_pos_emb + layer_pos_emb) are added
        once before the first m2p layer in forward_no_grad / forward_with_grad,
        not repeated in every layer.
        """
        def v_forward(
            memory_states: torch.Tensor,
            position_embeddings: tuple,
            attention_mask: Optional[torch.Tensor],
        ) -> torch.Tensor:
            B, L4, M4, H = memory_states.shape

            # (B, L/4, M*4, H) → transpose → (B, M*4, L/4, H) → (B*M*4, L/4, H)
            hs = memory_states.transpose(1, 2).reshape(B * M4, L4, H)

            # Generate position_embeddings for vertical attention.
            # After reshape the sequence dimension is L/4 (layers within a token group).
            v_pos_ids = torch.arange(L4, device=hs.device).unsqueeze(0)
            v_pos_emb = self.m2p_transformer.rotary_emb(hs, v_pos_ids)

            hs = m2p_layer(
                hs,
                position_embeddings=v_pos_emb,
                attention_mask=attention_mask,
            )

            # (B*M*4, L/4, H) → (B, M*4, L/4, H) → transpose → (B, L/4, M*4, H)
            hs = hs.reshape(B, M4, L4, H).transpose(1, 2)
            return hs
        return v_forward

    # ------------------------------------------------------------------
    # Validation and reshape helpers
    # ------------------------------------------------------------------

    def _validate_memory_states(self, memory_states: torch.Tensor):
        """
        Validate that memory_states shape matches expected dimensions.

        For only_full_4for1: expects (B, L, M, H) where L = num_llm_layers.
        For only_full_1for1: expects (B, L_fa, M, H) where L_fa = num_full_attn_layers.

        Args:
            memory_states: (B, L, M, H) tensor.

        Raises:
            ValueError: if dimensions do not match config.
        """
        B, L, M, H = memory_states.shape
        if self._memory_method == "only_full_4for1":
            if L != self._num_llm_layers:
                raise ValueError(
                    f"memory_states L={L} != expected num_llm_layers={self._num_llm_layers}"
                )
        elif self._memory_method == "only_full_1for1":
            if L != self._num_full_attn_layers:
                raise ValueError(
                    f"memory_states L={L} != expected num_full_attn_layers={self._num_full_attn_layers}"
                )
        if M != self._num_mem_token:
            raise ValueError(
                f"memory_states M={M} != expected num_mem_token={self._num_mem_token}"
            )
        if H != self._hidden_size:
            raise ValueError(
                f"memory_states H={H} != expected hidden_size={self._hidden_size}"
            )

    def _initial_reshape(self, memory_states: torch.Tensor) -> torch.Tensor:
        """
        Reshape input memory_states based on memory_method.

        For only_full_4for1:
            (B, L, M, H) → (B, L/4, M*4, H)
        For only_full_1for1:
            (B, L_fa, M, H) → (B, L_fa, M, H)  (no reshape needed)

        Args:
            memory_states: (B, L, M, H) tensor

        Returns:
            reshaped_states: (B, effective_L, effective_M, H) tensor
        """
        if self._memory_method == "only_full_4for1":
            B, L, M, H = memory_states.shape

            # Check that L is divisible by 4
            if L % 4 != 0:
                raise ValueError(f"L dimension ({L}) must be divisible by 4 for initial reshape")

            # (B, L, M, H) → (B, L/4, 4, M, H) → transpose → (B, L/4, M, 4, H) → reshape → (B, L/4, M*4, H)
            L4 = L // 4
            hs = memory_states.reshape(B, L4, 4, M, H)
            hs = hs.transpose(2, 3)  # (B, L4, M, 4, H)
            hs = hs.reshape(B, L4, M * 4, H)
            return hs

        elif self._memory_method == "only_full_1for1":
            # No reshape needed — already (B, L_fa, M, H)
            return memory_states

    # ------------------------------------------------------------------
    # Core forward: run m2p layers (no grad)
    # ------------------------------------------------------------------

    def _run_m2p_layers_no_grad(
        self,
        memory_states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Run m2p_transformer layers across pipeline stages WITHOUT gradient.

        Args:
            memory_states: (B, L/4, M*4, H) tensor after initial reshape.
                On stages that don't own the first m2p layer, this may be
                a dummy tensor — ``batch_size`` is used for shape instead.
            batch_size: batch size B (known on all stages).

        Returns:
            memory_states: (B, L/4, M*4, H) after all m2p_transformer layers and norm.
        """
        L4 = self._effective_L
        M4 = self._effective_M
        H = self._hidden_size
        B = batch_size
        pending_send = None

        for layer_idx in range(self._m2p_num_layers):
            layer_stage = self._m2p_layer_to_stage[layer_idx]

            # --- Receive from previous stage if needed ---
            if layer_idx > 0:
                prev_stage = self._m2p_layer_to_stage[layer_idx - 1]
                if prev_stage != layer_stage and layer_stage == self._my_stage:
                    memory_states = self._sync_recv_hidden(
                        shape=(B, L4, M4, H),
                        src_stage=prev_stage,
                    )

            # --- Run layer if it's on this stage ---
            if layer_stage == self._my_stage:
                # position_embeddings are generated inside h_forward / v_forward
                # based on the actual sequence length after reshape.
                memory_states = self._m2p_layer_forwards[layer_idx](
                    memory_states, None, None,
                )

            # --- Async send to next stage if needed ---
            if layer_idx < self._m2p_num_layers - 1:
                next_stage = self._m2p_layer_to_stage[layer_idx + 1]
                if layer_stage != next_stage and layer_stage == self._my_stage:
                    if pending_send is not None:
                        pending_send.wait()
                    # Use blocking send to avoid NCCL WorkNCCL objects
                    # accumulating (memory leak).
                    pipeline_send(
                        memory_states.contiguous(), next_stage
                    )
                    pending_send = None

        if pending_send is not None:
            pending_send.wait()

        # --- Transfer to m2p norm stage if needed ---
        last_layer_stage = self._m2p_layer_to_stage[self._m2p_num_layers - 1]
        if last_layer_stage != self._m2p_norm_stage:
            if last_layer_stage == self._my_stage:
                pipeline_send(memory_states.contiguous(), self._m2p_norm_stage)
            if self._m2p_norm_stage == self._my_stage:
                memory_states = self._sync_recv_hidden(
                    shape=(B, L4, M4, H),
                    src_stage=last_layer_stage,
                )

        # --- Apply final norm on the norm stage ---
        if self._my_stage == self._m2p_norm_stage:
            orig_shape = memory_states.shape
            memory_states = memory_states.reshape(-1, M4, H)
            if self.m2p_transformer.last_norm_type == "gated":
                gate = self.m2p_transformer.norm_gate_proj(memory_states)
                memory_states = self.m2p_transformer.norm(memory_states, gate)
            elif self.m2p_transformer.last_norm_type == "normal":
                memory_states = self.m2p_transformer.norm(memory_states)
            # else: "none" — identity, no-op
            memory_states = memory_states.reshape(orig_shape)

        return memory_states

    # ------------------------------------------------------------------
    # Core forward: run m2p layers (with grad)
    # ------------------------------------------------------------------

    def _run_m2p_layers_with_grad(
        self,
        memory_states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Run m2p_transformer layers across pipeline stages WITH gradient.

        Uses PipelineAsyncSend / PipelineRecv for autograd-aware communication.

        Args:
            memory_states: (B, L/4, M*4, H) tensor after initial reshape.
                On stages that don't own the first m2p layer, this may be
                a dummy tensor — ``batch_size`` is used for shape instead.
            batch_size: batch size B (known on all stages).

        Returns:
            memory_states: (B, L/4, M*4, H) after all m2p_transformer layers and norm.
        """
        L4 = self._effective_L
        M4 = self._effective_M
        H = self._hidden_size
        B = batch_size
        pending_send_tensor = None

        for layer_idx in range(self._m2p_num_layers):
            layer_stage = self._m2p_layer_to_stage[layer_idx]

            # --- Receive from previous stage if needed ---
            if layer_idx > 0:
                prev_stage = self._m2p_layer_to_stage[layer_idx - 1]
                if prev_stage != layer_stage and layer_stage == self._my_stage:
                    placeholder = torch.empty(0, device=self._my_device, requires_grad=True)
                    memory_states = PipelineRecv.apply(
                        placeholder, prev_stage, (B, L4, M4, H),
                        self._dtype, self._my_device, 0,
                    )

            # --- Run layer if it's on this stage ---
            if layer_stage == self._my_stage:
                # position_embeddings are generated inside h_forward / v_forward
                # based on the actual sequence length after reshape.
                if self._activation_checkpointing:
                    memory_states = torch_checkpoint(
                        self._m2p_layer_forwards[layer_idx],
                        memory_states, None, None,
                        use_reentrant=False,
                    )
                else:
                    memory_states = self._m2p_layer_forwards[layer_idx](
                        memory_states, None, None,
                    )

            # --- Send to next stage if needed ---
            if layer_idx < self._m2p_num_layers - 1:
                next_stage = self._m2p_layer_to_stage[layer_idx + 1]
                if layer_stage != next_stage and layer_stage == self._my_stage:
                    pending_send_tensor = pipeline_send_with_grad(
                        memory_states, next_stage
                    )

        if pending_send_tensor is not None:
            # Save anchor only if the tensor has gradient.
            if pending_send_tensor.requires_grad and pending_send_tensor.grad_fn is not None:
                self._backward_anchors.append(pending_send_tensor)

        # --- Transfer to m2p norm stage if needed ---
        last_layer_stage = self._m2p_layer_to_stage[self._m2p_num_layers - 1]
        if last_layer_stage != self._m2p_norm_stage:
            if last_layer_stage == self._my_stage:
                out = pipeline_send_with_grad(
                    memory_states, self._m2p_norm_stage
                )
                # Save anchor only if the tensor has gradient.
                if out.requires_grad and out.grad_fn is not None:
                    self._backward_anchors.append(out)
            if self._m2p_norm_stage == self._my_stage:
                placeholder = torch.empty(0, device=self._my_device, requires_grad=True)
                memory_states = PipelineRecv.apply(
                    placeholder, last_layer_stage, (B, L4, M4, H),
                    self._dtype, self._my_device, 0,
                )

        # --- Apply final norm on the norm stage ---
        if self._my_stage == self._m2p_norm_stage:
            orig_shape = memory_states.shape
            memory_states = memory_states.reshape(-1, M4, H)
            if self.m2p_transformer.last_norm_type == "gated":
                gate = self.m2p_transformer.norm_gate_proj(memory_states)
                memory_states = self.m2p_transformer.norm(memory_states, gate)
            elif self.m2p_transformer.last_norm_type == "normal":
                memory_states = self.m2p_transformer.norm(memory_states)
            # else: "none" — identity, no-op
            memory_states = memory_states.reshape(orig_shape)

        return memory_states

    # ------------------------------------------------------------------
    # Pipeline communication helpers
    # ------------------------------------------------------------------

    def _sync_recv_hidden(
        self,
        shape: tuple,
        src_stage: int,
        tag: int = 0,
    ) -> torch.Tensor:
        """Synchronously receive hidden_states from src_stage (no grad)."""
        buf = torch.empty(shape, dtype=self._dtype, device=self._my_device)
        pipeline_recv(buf, src_stage, tag)
        return buf

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_no_grad(
        self,
        memory_states: torch.Tensor,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Forward through the hypernetwork **without gradient**.

        Takes memory_states collected from the LLM and processes them through
        the m2p_transformer layers.

        Args:
            memory_states: (B, L, M, H) tensor on the mem_gather_target stage.
                On other stages this may be a dummy tensor.
            batch_size: batch size B (must be provided on ALL stages so that
                non-owning stages can allocate recv buffers).

        Returns:
            On the m2p_norm stage:
                memory_states: (B, L/4, M*4, H) — processed memory states.
            On other stages:
                None
        """
        if self._my_stage == self._mem_gather_target_stage:
            self._validate_memory_states(memory_states)
            memory_states = self._initial_reshape(memory_states)
            # Add 2D positional embeddings once on the gather stage (where
            # the first m2p layer lives) before entering the layer loop.
            memory_states = memory_states + self.token_pos_emb + self.layer_pos_emb

        memory_states = self._run_m2p_layers_no_grad(memory_states, batch_size)

        if self._my_stage == self._m2p_norm_stage:
            return memory_states
        return None

    def forward_with_grad(
        self,
        memory_states: torch.Tensor,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Forward through the hypernetwork **with gradient**.

        Gradients flow back through the pipeline via PipelineAsyncSend /
        PipelineRecv autograd functions.

        Args:
            memory_states: (B, L, M, H) tensor on the mem_gather_target stage.
                On other stages this may be a dummy tensor.
            batch_size: batch size B (must be provided on ALL stages so that
                non-owning stages can allocate recv buffers).

        Returns:
            On the m2p_norm stage:
                memory_states: (B, L/4, M*4, H) — processed memory states.
            On other stages:
                None
        """
        if self._my_stage == self._mem_gather_target_stage:
            self._validate_memory_states(memory_states)
            memory_states = self._initial_reshape(memory_states)
            # Add 2D positional embeddings once on the gather stage (where
            # the first m2p layer lives) before entering the layer loop.
            memory_states = memory_states + self.token_pos_emb + self.layer_pos_emb

        memory_states = self._run_m2p_layers_with_grad(memory_states, batch_size)

        if self._my_stage == self._m2p_norm_stage:
            return memory_states
        return None
