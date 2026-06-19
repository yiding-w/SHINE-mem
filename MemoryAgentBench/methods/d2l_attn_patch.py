"""Patch doc-to-lora to use sdpa/eager when flash-attn is not installed."""

from __future__ import annotations

import os

_D2L_ATTN_PATCH_APPLIED = False


def apply_d2l_attn_patch(attn_impl: str = "sdpa") -> None:
    """Force attn_impl for base model, ctx encoder, and HyperLoRA Idefics2Perceiver."""
    global _D2L_ATTN_PATCH_APPLIED
    if _D2L_ATTN_PATCH_APPLIED:
        return
    _D2L_ATTN_PATCH_APPLIED = True

    os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = attn_impl
    os.environ["HF_ATTN_IMPLEMENTATION"] = attn_impl

    import transformers.modeling_utils as modeling_utils

    def _set_config_attn(config, impl: str) -> None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(config, attr):
                value = getattr(config, attr)
                if value in (None, "flash_attention_2"):
                    setattr(config, attr, impl)

    @classmethod
    def _check_and_enable_flash_attn_2(cls, config, *args, **kwargs):
        _set_config_attn(config, attn_impl)
        return config

    orig_autoset = modeling_utils.PreTrainedModel._autoset_attn_implementation

    @classmethod
    def _autoset_attn_implementation(cls, config, *args, **kwargs):
        _set_config_attn(config, attn_impl)
        try:
            result = orig_autoset(config, *args, **kwargs)
            _set_config_attn(result if result is not None else config, attn_impl)
            return result if result is not None else config
        except ImportError:
            _set_config_attn(config, attn_impl)
            return config

    modeling_utils.PreTrainedModel._check_and_enable_flash_attn_2 = _check_and_enable_flash_attn_2
    modeling_utils.PreTrainedModel._autoset_attn_implementation = _autoset_attn_implementation

    # doc-to-lora Perceiver hardcodes flash_attention_2 in Idefics2PerceiverConfig.
    try:
        from ctx_to_lora.modeling.idefics2 import Idefics2PerceiverConfig

        if not getattr(Idefics2PerceiverConfig, "_d2l_attn_patched", False):
            Idefics2PerceiverConfig._d2l_attn_patched = True
            _orig_cfg_init = Idefics2PerceiverConfig.__init__

            def _patched_cfg_init(self, *args, **kwargs):
                kwargs["attn_implementation"] = attn_impl
                _orig_cfg_init(self, *args, **kwargs)
                _set_config_attn(self, attn_impl)

            Idefics2PerceiverConfig.__init__ = _patched_cfg_init  # type: ignore[method-assign]
    except ImportError:
        pass


def patch_d2l_state_dict_attn(state_dict: dict, attn_impl: str = "sdpa") -> None:
    hypernet_config = state_dict.get("hypernet_config")
    if hypernet_config is None:
        return
    agg_config = getattr(hypernet_config, "aggregator_config", None)
    if agg_config is not None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(agg_config, attr):
                setattr(agg_config, attr, attn_impl)
