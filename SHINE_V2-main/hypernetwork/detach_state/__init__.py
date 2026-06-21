"""
DetachState package.

Factory function ``create_detach_state(cfg, *, mode, **kwargs)`` returns the
appropriate DetachState subclass based on ``cfg.type``.
"""

from hypernetwork.detach_state.base import BaseDetachState
from hypernetwork.detach_state.empty import EmptyDetachState


def create_detach_state(cfg, *, mode: str, **kwargs) -> BaseDetachState:
    """
    Factory function: create a DetachState instance based on cfg.type.

    Args:
        cfg: Configuration dict/DictConfig with at least a "type" key.
        mode: REQUIRED. "pp" or "tp" — determines which subclass to use.
              type="full" + mode="pp" → FullDetachState
              type="full" + mode="tp" → FullTPDetachState
        **kwargs: Additional keyword arguments forwarded to the DetachState
                  subclass constructor (e.g. local_batch_size, micro_batch_size,
                  pipeline topology info for FullDetachState, or tp_rank,
                  tp_world, tp_process_group for FullTPDetachState).

    Returns:
        A BaseDetachState subclass instance.

    Raises:
        ValueError: If type is unknown.
    """
    type_name = str(cfg.get("type", "empty"))

    if type_name == "empty":
        return EmptyDetachState(cfg)
    elif type_name == "full":
        if mode == "tp":
            from hypernetwork.detach_state.full_tp import FullTPDetachState
            return FullTPDetachState(cfg, **kwargs)
        from hypernetwork.detach_state.full import FullDetachState
        return FullDetachState(cfg, **kwargs)
    else:
        raise ValueError(
            f"Unknown DetachState type '{type_name}'. "
            f"Supported: 'empty', 'full'."
        )
