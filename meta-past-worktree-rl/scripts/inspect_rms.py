"""Print the RMS of every perturbable tensor (m2p weight-only scope)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.shine_adapter import ShineHypernet


def main():
    home = Path.home()
    net = ShineHypernet(
        ckpt_dir=str(home / "huggingfacemodels" / "SHINE-ift_mqa_1qa"),
        device="cuda:0",
        backbone=str(home / "huggingfacemodels" / "Qwen3-8B"),
        lora_r=8, metalora_r=128,
    )
    params = net.all_perturbable_params(
        include_metalora=False, include_mem_tokens=False, exclude_bias=True
    )
    rows = []
    for n, p in params:
        rms = float(p.detach().float().pow(2).mean().sqrt().item())
        rows.append((n, rms, p.numel()))
    rows.sort(key=lambda x: x[1])

    print(f"{'name':60s}  {'RMS':>10s}  {'numel':>10s}")
    for n, r, k in rows:
        print(f"{n:60s}  {r:10.3e}  {k:10d}")
    print()
    for thresh in (0.001, 0.01, 0.05, 0.1, 0.3):
        kept = sum(1 for _, r, _ in rows if r >= thresh)
        print(f"min_rms >= {thresh}:  {kept}/{len(rows)} tensors kept")


if __name__ == "__main__":
    main()
