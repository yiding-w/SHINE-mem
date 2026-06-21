"""
W-Transform package for detach_state wdict.

Provides pluggable transform modules that can be applied to wdict
before injecting into the LLM forward pass.

Usage:
    from utils.mytransform import create_transform
    module = create_transform(cfg, model_cfg, tp_mode=True, ...)
"""
from utils.mytransform.create_transform import create_transform

__all__ = ["create_transform"]
