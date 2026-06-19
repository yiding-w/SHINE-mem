"""Patch doc-to-lora to use sdpa/eager when flash-attn is not installed."""

from __future__ import annotations

import os
from typing import Any

_D2L_ATTN_PATCH_APPLIED = False


def flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_d2l_attn(agent_config: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Return (use_flash_attn, attn_implementation) for doc-to-lora loading."""
    agent_config = agent_config or {}

    if os.environ.get("D2L_FORCE_SDPA", "").strip() in ("1", "true", "yes"):
        return False, "sdpa"

    use_flash = agent_config.get("use_flash_attn")
    if use_flash is None:
        env_flash = os.environ.get("D2L_USE_FLASH_ATTN", "").strip().lower()
        if env_flash in ("0", "false", "no"):
            use_flash = False
        elif env_flash in ("1", "true", "yes"):
            use_flash = True
        else:
            use_flash = flash_attn_available()

    attn_impl = (
        agent_config.get("attn_implementation")
        or os.environ.get("D2L_ATTN_IMPLEMENTATION")
        or os.environ.get("ATTN_IMPLEMENTATION")
    )

    if use_flash and flash_attn_available():
        return True, "flash_attention_2"

    if attn_impl in ("flash_attention_2", "flash_attn"):
        if flash_attn_available():
            return True, "flash_attention_2"
        print("[d2l_attn] flash_attention_2 requested but flash_attn not importable; falling back to sdpa", flush=True)
        return False, "sdpa"

    return False, str(attn_impl or "sdpa")


def apply_d2l_attn_patch(attn_impl: str = "sdpa") -> None:
    """Force sdpa/eager when flash-attn is unavailable. No-op for flash_attention_2."""
    global _D2L_ATTN_PATCH_APPLIED
    if attn_impl in ("flash_attention_2", "flash_attn"):
        os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "flash_attention_2"
        os.environ["HF_ATTN_IMPLEMENTATION"] = "flash_attention_2"
        return
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
    if attn_impl in ("flash_attention_2", "flash_attn"):
        return
    hypernet_config = state_dict.get("hypernet_config")
    if hypernet_config is None:
        return
    agg_config = getattr(hypernet_config, "aggregator_config", None)
    if agg_config is not None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(agg_config, attr):
                setattr(agg_config, attr, attn_impl)
