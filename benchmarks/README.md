# DeepCTR benchmark suite

A small, self-contained harness that trains and compares DeepCTR models
end-to-end and writes a sorted **leaderboard**. It covers three tracks:

| Track | Models | Dataset | Metrics |
| --- | --- | --- | --- |
| `single` | 21 single-task CTR models | Criteo (binary) | AUC, LogLoss |
| `multitask` | SharedBottom, ESMM, MMOE, PLE | Census-Income (2 tasks) | per-task AUC/LogLoss, mean_AUC |
| `sequence` | DIN, BST, DSIN (DIEN¹) | synthetic / MovieLens | AUC, LogLoss |

¹ DIEN is **skipped on TensorFlow ≥ 2.0** — its GRU/AUGRU layers depend on
legacy TF1 private RNN APIs (its own unit test skips on TF2).

## Install

The core `deepctr` library stays dependency-lean, so the benchmark has a few
extras of its own. Install TensorFlow first (per the project README), then:

```bash
pip install -r benchmarks/requirements.txt      # scikit-learn, pandas
```

**TensorFlow ≥ 2.16 (Keras 3) users:** DeepCTR targets the Keras 2 API, so the
suite runs under *legacy Keras* — exactly like the project CI. Install the
matching `tf-keras` and the suite sets `TF_USE_LEGACY_KERAS=1` for you:

```bash
pip install tf-keras~=2.19.0     # match your TensorFlow minor version
```

(On TF 1.15 / 2.x-with-Keras-2 nothing extra is needed.)

## Quick start

```bash
# Fast self-test: 1 epoch, 2 models per track, bundled/tiny data
python -m benchmarks.benchmark --track single --quick

# Full single-task leaderboard on the bundled 200-row Criteo sample
python -m benchmarks.benchmark --track single

# All three tracks
python -m benchmarks.benchmark --track all
```

Each run prints a leaderboard and writes `benchmarks/results/<track>_<dataset>.csv`
and `.md` (git-ignored).

## Datasets

### Criteo (single-task)

The bundled `examples/criteo_sample.txt` has only 200 rows — fine to verify the
pipeline, **not** for meaningful AUC. For real numbers:

```bash
# Try the Criteo "DAC" ~100k-row sample (downloads once into benchmarks/data/)
python -m benchmarks.benchmark --track single --download --epochs 5

# ...or point at any full Criteo file (TSV, no header: label + I1..I13 + C1..C26)
python -m benchmarks.benchmark --track single --data-path /path/to/train.txt --epochs 5

# High-cardinality full data: hash sparse features instead of label-encoding
python -m benchmarks.benchmark --track single --data-path train.txt --hash --max-rows 1000000
```

The downloader is best-effort: the historical Criteo URL is fragile, so if it is
unreachable the loader prints manual-download instructions and falls back to the
bundled sample (it never hard-fails). Override the URL with `--download-url`.

### Census-Income (multitask)

Uses the bundled `examples/census-income.sample`; pass the full UCI
Census-Income KDD `.data` file via `--data-path`. Two derived binary tasks:
`label_income` (>50k) and `label_marital` (never-married).

> When `--data-path` points at `census-income.data` and its sibling
> `census-income.test` exists, the loader uses the dataset's **official
> train/test partition** (199,523 / 99,762 rows) instead of a random split —
> the canonical evaluation. Encoders are fit on the union so test-only
> categories don't crash the run. The bundled sample has no `.test` sibling and
> falls back to a random split.

> ESMM assumes sequentially-dependent CTR→CTCVR tasks; on Census it still runs,
> but its numbers are not directly comparable to the other multitask models.

### Sequence

`--seq-source synthetic` (default) deterministically generates user behavior
sequences with an **injected, learnable signal** (the label depends on whether
the target item is in the user's history and on the user's favourite category).
Synthetic AUC is modest with default model sizes — its purpose is to confirm all
sequence models build/train/predict and to compare their **cost** (params, train
time). For real predictive comparison use `--seq-source movielens --data-path
ml-1m/ratings...` (the bundled 200-row MovieLens file is only a smoke test);
Amazon-Electronics is the canonical heavy benchmark and is left as an extension.

## Common flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--track` | `single` | `single` / `multitask` / `sequence` / `all` |
| `--models a,b` | all | run only these models |
| `--exclude a,b` | – | skip these models |
| `--epochs` | 5 | training epochs |
| `--batch-size` | 256 | |
| `--embedding-dim` | 8 | uniform sparse embedding size |
| `--val-split` / `--test-size` | 0.2 / 0.2 | |
| `--temporal-split` / `--time-col` | off / – | single-task: chronological hold-out (most recent `--test-size` as test) instead of a random split — for time-ordered logs, to avoid leakage. Sorts by `--time-col` when given, else trusts file order. (The bundled/Criteo_x1 data is anonymised & shuffled with no timestamp, so it stays on the random split.) |
| `--seed` | 2020 | seeds python/numpy/TF |
| `--quick` | off | 1 epoch, 2 models/track, small data |
| `--source` / `--download` / `--data-path` | bundled | Criteo source |
| `--seq-source` / `--n-samples` / `--sess-max-count` / `--sess-len` | synthetic / 2000 / 3 / 4 | sequence options |
| `--output-dir` | `benchmarks/results` | leaderboard output |

## Output

The leaderboard ranks `status=ok` models by the headline metric (AUC desc /
mean_AUC desc / RMSE asc), with skipped/failed models listed below:

```
| Rank | Model   | Status | AUC    | LogLoss | Params  | Train(s) | Note |
| ---- | ------- | ------ | ------ | ------- | ------- | -------- | ---- |
| 1    | DCNMix  | ok     | 0.6022 | 0.7255  | 124,000 | 8.8      |      |
| 2    | DeepFM  | ok     | 0.5842 | 0.7297  | 118,564 | 6.8      |      |
| -    | DIEN    | skipped|        |         |         |          | DIEN's GRU/AUGRU ... unsupported on TF>=2.0 |
```

Every model runs in isolation (`clear_session()` + `try/except`), so one failing
model never aborts the sweep — it is recorded as `failed` with the error.

## Model-specific notes

- **AFM, CCPM, EDCN** are sparse-only (`support_dense=False`); the registry
  drops `DenseFeat` from their inputs automatically.
- **FLEN** needs ≥2 field groups; the registry round-robins the sparse features
  into synthetic groups (Criteo has no field taxonomy).
- **PNN** has no linear part; **MLR** uses the feature columns as both its region
  and base inputs.
- **ONN, FiBiNET, DeepFEFM, FwFM** build many pairwise/field interactions and are
  noticeably slower to construct than the rest.

## Extending

Add a model by registering a builder in `benchmarks/registry.py`:

```python
SINGLE_TASK_MODELS["MyModel"] = lambda lin, dnn, task: MyModel(lin, dnn, task=task)
```

Datasets live in `benchmarks/datasets/` and return the typed containers in
`benchmarks/common.py` (`SingleTaskData` / `MultiTaskData` / `SequenceData`).

## Test

```bash
TF_USE_LEGACY_KERAS=1 pytest tests/benchmark_test.py -q
```
