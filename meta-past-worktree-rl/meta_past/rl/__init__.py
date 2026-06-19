"""RL training stack."""

from __future__ import annotations

import os


_RANK = int(os.environ.get("RANK", "0"))
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def set_phase(phase: str) -> None:
    """Update this process's title so ``ps`` / ``nvitop`` show the current
    pipeline phase (``hyper``, ``wake``, ``push``, ``sample``, ``sleep``,
    ``rescore``, ``optim``, ``heldout``, ...).

    Format: ``rl r{rank}/{phase}`` so all ranks group together visually.
    No-op if ``setproctitle`` isn't installed.
    """
    try:
        from setproctitle import setproctitle
    except ImportError:
        return
    if _WORLD_SIZE > 1:
        setproctitle(f"rl r{_RANK}/{phase}")
    else:
        setproctitle(f"rl/{phase}")
