# Legacy MemoryTest Comparisons

These scripts are the earlier capacity, distractor, and no-LoRA comparison experiments. They were moved out of the `MemoryTest` root so the new split, upper-bound, post-training, and evaluation entry points can stay separate.

Run them from the repository root:

```bash
python -m MemoryTest.comparisons.compare_update_capacity --merge-method sum
python -m MemoryTest.comparisons.compare_distractor_effect --merge-method sum
python -m MemoryTest.comparisons.compare_density_budget_effect --merge-method sum
python -m MemoryTest.comparisons.compare_baselines
```

The comparison scripts still default to `MemoryTest/json_data/semantic_facts.json` and read facts from the head of that file, matching the LoRA upper-bound default `--selection-mode head`.

Capacity source data generation now lives at:

```bash
python -m MemoryTest.prepare_data.generate_capacity_data
```
