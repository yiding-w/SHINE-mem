"""Dataset adapters: each module returns a list of ``EvalItem`` via
``load(*, limit=None, **kwargs)`` and reports its default scorer +
which eval modes make sense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..items import EvalItem


@dataclass
class DatasetSpec:
    name: str
    bucket: str                            # "A", "B", "C"
    load: Callable[..., list[EvalItem]]    # (*, limit=..., **kwargs) -> items
    scorer: str                            # key into scoring.SCORERS
    default_modes: tuple[str, ...]         # subset of ("shine", "icl", "zero")
    notes: str = ""
    # Per-task scoring kwargs (e.g. n_options for MCQ).
    scorer_kwargs: dict[str, Any] = None


REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> None:
    if spec.name in REGISTRY:
        raise ValueError(f"duplicate dataset registration: {spec.name}")
    REGISTRY[spec.name] = spec


# Eager-import all adapters so registry is populated on package import.
# Bucket A
from . import squad           # noqa: E402,F401
from . import musique         # noqa: E402,F401
from . import hotpotqa        # noqa: E402,F401
from . import twowikimulti    # noqa: E402,F401
from . import drop            # noqa: E402,F401
from . import narrativeqa     # noqa: E402,F401
from . import pubmedqa        # noqa: E402,F401
from . import boolq           # noqa: E402,F401
from . import triviaqa        # noqa: E402,F401
from . import newsqa          # noqa: E402,F401
# Bucket B
from . import bbh             # noqa: E402,F401
from . import mmlu            # noqa: E402,F401
from . import mmlu_pro        # noqa: E402,F401
from . import agieval         # noqa: E402,F401
from . import arc             # noqa: E402,F401
from . import openbookqa      # noqa: E402,F401
from . import commonsenseqa   # noqa: E402,F401
from . import hellaswag       # noqa: E402,F401
from . import piqa            # noqa: E402,F401
from . import gsm8k           # noqa: E402,F401
from . import strategyqa      # noqa: E402,F401
from . import babi            # noqa: E402,F401
from . import bigbench_non_hard  # noqa: E402,F401
from . import natural_instr   # noqa: E402,F401
from . import truthfulqa_mc   # noqa: E402,F401
# Bucket C
from . import gsm8k_zeroshot  # noqa: E402,F401
from . import mmlu_zeroshot   # noqa: E402,F401
from . import humaneval       # noqa: E402,F401
from . import truthfulqa_gen  # noqa: E402,F401
