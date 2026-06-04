# SHINE integration (upstream: HUST-AI-HYZ/MemoryAgentBench)

This directory is vendored from [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) with local patches for SHINE + Qwen3-8B evaluation.

**SHINE-specific changes:**

- `methods/shine_runner.py` — load SHINE hypernetwork, HF local baseline
- `agent.py` — `SHINE_agent`, `HF_local_long_context_agent`
- `utils/templates.py` — template fallback for local agents
- `configs/agent_conf/SHINE_Agents/`, `Local_HF_Agents/`
- `bash_files/sh/run_shine_mab.sh`, `bash_files/configs/shine_mab_eval.txt`
- `docs/SHINE_MAB_SETUP.md`

Run from this directory; set `SHINE_ROOT` to the parent repo (SHINE training code).
