#!/usr/bin/env python3
"""
Deprecated: use δ-mem benchmark_compare with --no-skip-shine instead.

  python -m deltamem.eval.benchmark_compare \\
    --skip-base --skip-delta --skip-lora --no-skip-shine \\
    --shine-agent-config ... --external-memory-agent-bench-root ...

See MemoryAgentBench/docs/DELTA_MEM_HGX001.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    mab_root = Path(__file__).resolve().parents[1]
    shine_root = mab_root.parent
    delta_root = shine_root / "third_party" / "delta-Mem"
    agent_cfg = mab_root / "configs/agent_conf/SHINE_Agents/SHINE_agent_qwen3_8b.yaml"
    out = shine_root / "outputs/delta_mem_qwen3_8b/shine_memory_agent_bench.json"

    if not delta_root.is_dir():
        print("Clone delta-Mem first: bash MemoryAgentBench/bash_files/sh/setup_delta_mem_hgx001.sh", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "deltamem.eval.benchmark_compare",
        "--model-path",
        "/ceph/home/muhan01/huggingfacemodels/Qwen3-8B",
        "--device",
        "cuda:0",
        "--external-memory-agent-bench-root",
        str(mab_root),
        "--shine-root",
        str(shine_root),
        "--shine-agent-config",
        str(agent_cfg),
        "--tasks",
        "memory_agent_bench",
        "--skip-base",
        "--skip-delta",
        "--skip-lora",
        "--no-skip-shine",
        "--output-json",
        str(out),
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = f"{delta_root}:{shine_root}:{mab_root}:{env.get('PYTHONPATH', '')}"
    print("Redirecting to:", " ".join(cmd), file=sys.stderr)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
