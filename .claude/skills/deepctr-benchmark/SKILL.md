---
name: deepctr-benchmark
description: >-
  Run the DeepCTR model benchmark suite (benchmarks/) to validate and compare
  CTR models end-to-end on real or bundled data, producing a sorted leaderboard.
  Use when asked to benchmark, compare, validate, or measure DeepCTR models
  (single-task CTR, multitask, or sequence), or to re-run / extend the
  benchmark after a code change.
---

# DeepCTR benchmark suite

A self-contained harness (`benchmarks/`) that trains DeepCTR models and writes a
ranked leaderboard. Three tracks: `single` (21 single-task CTR models, Criteo),
`multitask` (SharedBottom/ESMM/MMOE/PLE, Census-Income), `sequence`
(DIN/BST/DSIN, DIEN skipped on TF≥2.0). Full docs: `benchmarks/README.md`.
Curated past results: `benchmarks/RESULTS.md`.

## Always do these two things first

1. **Force CPU** with `CUDA_VISIBLE_DEVICES=""`. The GPU on this host has a
   CUDA/PTX mismatch — every model fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
   on GPU. The harness isolates each model, so on GPU you get an all-`failed`
   leaderboard rather than a crash; the fix is always CPU.
2. **Use legacy Keras**: `TF_USE_LEGACY_KERAS=1` (DeepCTR targets the Keras 2
   API; on TF≥2.16 that means `tf-keras`). The package sets this itself, but set
   it explicitly when invoking pytest.

Run long sweeps in the background and poll the log — a full 21-model single-task
sweep on real data is several minutes to ~30 min on CPU.

## Data on this host

Real data lives in `benchmarks/data/` (git-ignored). If absent, regenerate or
re-download (see README; the built-in Criteo DAC URL is dead — anonymous HF
mirrors rate-limit, so a `HF_TOKEN` is needed for `reczoo/Criteo_x1`).

| File | Rows | Use |
| ---- | ---- | --- |
| `benchmarks/data/criteo_x1_500k.csv` | 500,000 | single-task, meaningful AUC |
| `benchmarks/data/criteo_x1_200k.csv` | 200,000 | single-task, faster |
| `benchmarks/data/census-income.data` (+ `.test`) | 199,523 (+99,762) | multitask, official split |
| `examples/criteo_sample.txt` | 200 | smoke only — AUC is noise |

## Commands

```bash
# Fast smoke test (1 epoch, 2 models/track, bundled data) — verify the harness
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track all --quick

# Single-task, real Criteo 500k — the headline comparison
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track single \
  --data-path benchmarks/data/criteo_x1_500k.csv \
  --epochs 1 --batch-size 1024 --val-split 0 --seed 2020

# Multitask, real Census with its official train/test partition
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track multitask \
  --data-path benchmarks/data/census-income.data --epochs 3 --batch-size 1024

# Subset / exclude slow models; e.g. drop the interaction-heavy ones to save time
#   --models DeepFM,DCN,xDeepFM     run only these
#   --exclude FiBiNET,ONN,DeepFEFM,FwFM,FGCNN   ONN alone is 77M params / ~245s
```

Leaderboards print to stdout and write `benchmarks/results/<track>_<dataset>.csv`
and `.md`. To preserve a run, copy the numbers into `benchmarks/RESULTS.md`
(the `results/` dir is git-ignored).

## Train / test split — get this right

- **Default = random split**, `--test-size 0.2`, fixed `--seed`. Correct for
  Criteo_x1 / DAC, which is anonymised and shuffled with no timestamp (so a
  random split is leakage-free and standard).
- **`--temporal-split` (+ optional `--time-col COL`)**: chronological hold-out —
  the most recent `test_size` fraction becomes the test set, so no future row
  leaks into training. Sorts by `--time-col` when present, else trusts file
  order. Use only for time-ordered logs; the bundled Criteo has no timestamp.
- **Census official split**: when `--data-path` is `census-income.data` and a
  sibling `census-income.test` exists, the loader uses that official partition
  automatically (encoders fit on the union). Don't reshuffle it.
- **`--val-split`** carves validation from *train* but is **not wired to any
  early-stopping/checkpoint** — it does not influence training or model
  selection. For short fixed-epoch runs set `--val-split 0` so training uses the
  full train split; only keep it if you add an EarlyStopping callback.

## Interpreting results

- Reported AUC/LogLoss are on the held-out **test** set (never seen in `fit`).
- Need enough data for a real ranking: on 200-row bundled data AUC ≈ 0.5 noise;
  on 200k–500k the deep interaction models (DeepFEFM/FiBiNET/DeepFM/xDeepFM)
  separate from simpler ones. More epochs raise absolute AUC but rarely flip the
  top group.
- Always report the split and epoch count alongside the leaderboard — a ranking
  is only comparable within the same data/split/epochs.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" TF_USE_LEGACY_KERAS=1 python -m pytest tests/benchmark_test.py -q
```

Covers each track on tiny data plus the split guarantees (temporal hold-out
keeps the most-recent rows as test; Census uses the official `.test`).
