# DeepCTR benchmark results

Real-data leaderboards produced by `benchmarks/benchmark.py`. The raw
`benchmarks/results/*.{csv,md}` are git-ignored (generated); this file is the
curated, version-controlled snapshot.

> **Environment note.** All runs below were executed **CPU-only**
> (`CUDA_VISIBLE_DEVICES=""`) because the GPU on the run host had a CUDA/PTX
> mismatch. TensorFlow 2.19 + `tf-keras` 2.19 (legacy Keras,
> `TF_USE_LEGACY_KERAS=1`). Absolute AUCs are not tuned maxima — the value is in
> the **relative ranking and cost** under a fixed, leakage-free split.

## Single-task CTR — real Criteo `criteo_x1_2m.csv` (current headline)

2,000,000 samples sliced from the full Criteo_x1 `train.csv` · **random split,
seed=2020 → 1,600,000 train / 400,000 test** (`--val-split 0`). 1 epoch, batch
1024, embedding-dim 8.

| Rank | Model | AUC | LogLoss | Params | Train(s) |
| ---- | ----- | --- | ------- | ------ | -------- |
| 1 | DeepFEFM | 0.7914 | 0.4541 | 7.5M | 332 |
| 2 | FiBiNET | 0.7913 | 0.4542 | 8.7M | 188 |
| 3 | PNN | 0.7901 | 0.4552 | 6.7M | 72 |
| 4 | ONN | 0.7900 | 0.4556 | 163.8M | 1304 |
| 5 | DeepFM | 0.7893 | 0.4558 | 7.4M | 60 |
| 6 | xDeepFM | 0.7892 | 0.4560 | 7.7M | 163 |
| 7 | FwFM | 0.7887 | 0.4563 | 7.4M | 307 |
| 8 | FNN | 0.7886 | 0.4565 | 6.6M | 55 |
| 9 | DCN | 0.7885 | 0.4564 | 7.4M | 71 |
| 10 | WDL | 0.7885 | 0.4565 | 7.4M | 61 |
| 11 | FLEN | 0.7884 | 0.4565 | 7.4M | 68 |
| 12 | AutoInt | 0.7882 | 0.4570 | 7.4M | 129 |
| 13 | DCNMix | 0.7879 | 0.4569 | 7.5M | 186 |
| 14 | NFM | 0.7872 | 0.4575 | 7.3M | 56 |
| 15 | FGCNN | 0.7807 | 0.4629 | 15.8M | 500 |
| 16 | DIFM | 0.7795 | 0.4639 | 7.4M | 116 |
| 17 | IFM | 0.7784 | 0.4647 | 7.4M | 58 |
| 18 | MLR | 0.7753 | 0.4681 | 6.5M | 44 |
| 19 | CCPM | 0.7579 | 0.4807 | 7.3M | 90 |
| 20 | EDCN | 0.7543 | 0.4819 | 7.6M | 82 |
| 21 | AFM | 0.7495 | 0.4861 | 7.3M | 93 |

**Cross-scale (same random split, 1 epoch unless noted)**

| Model | 200k (3ep) | 500k (1ep) | 2M (1ep) |
| ----- | ---------- | ---------- | -------- |
| DeepFM | 0.745 | 0.7794 | **0.7893** |
| DeepFEFM | (excluded) | 0.7817 | **0.7914** |
| FiBiNET | (excluded) | 0.7797 | **0.7913** |

- **Data volume is the biggest lever**: DeepFM gains ~0.01 AUC from 500k→2M,
  more than swapping architectures buys — the top 14 models sit within 0.787–0.791.
- DeepFEFM / FiBiNET stay #1/#2; **PNN climbs to #3** at scale.
- **ONN catches up on AUC at 2M (0.7900) but is absurdly costly**: 163.8M params
  and ~22 min (22× DeepFM's 60s). Not worth it.

## Single-task CTR — real Criteo `criteo_x1_500k.csv`

500,000 samples · **random split, seed=2020 → 400,000 train / 100,000 test**
(no validation carve-out: `--val-split 0`, since 1 epoch + no early stopping
makes a validation set dead weight). 1 epoch, batch 1024, embedding-dim 8.
The Criteo_x1 (Kaggle DAC) data is anonymised and shuffled with no timestamp,
so a random split is the standard, leakage-free choice.

| Rank | Model | AUC | LogLoss | Params | Train(s) |
| ---- | ----- | --- | ------- | ------ | -------- |
| 1 | DeepFEFM | 0.7817 | 0.4664 | 3.6M | 130.4 |
| 2 | FiBiNET | 0.7797 | 0.4680 | 4.8M | 104.8 |
| 3 | DeepFM | 0.7794 | 0.4682 | 3.5M | 13.2 |
| 4 | xDeepFM | 0.7793 | 0.4683 | 3.8M | 39.1 |
| 5 | FwFM | 0.7791 | 0.4684 | 3.5M | 103.1 |
| 6 | DCN | 0.7790 | 0.4687 | 3.5M | 16.3 |
| 7 | PNN | 0.7790 | 0.4687 | 3.2M | 14.8 |
| 8 | WDL | 0.7790 | 0.4685 | 3.5M | 13.7 |
| 9 | FNN | 0.7789 | 0.4686 | 3.1M | 10.0 |
| 10 | FLEN | 0.7789 | 0.4686 | 3.5M | 15.3 |
| 11 | AutoInt | 0.7787 | 0.4688 | 3.5M | 32.5 |
| 12 | DCNMix | 0.7786 | 0.4690 | 3.6M | 47.1 |
| 13 | NFM | 0.7779 | 0.4695 | 3.5M | 15.8 |
| 14 | ONN | 0.7602 | 0.4955 | 77.0M | 244.7 |
| 15 | DIFM | 0.7525 | 0.4879 | 3.6M | 29.5 |
| 16 | MLR | 0.7499 | 0.4935 | 3.0M | 26.6 |
| 17 | FGCNN | 0.7495 | 0.4896 | 8.4M | 122.6 |
| 18 | IFM | 0.7447 | 0.4927 | 3.5M | 14.4 |
| 19 | CCPM | 0.7445 | 0.4934 | 3.4M | 20.6 |
| 20 | EDCN | 0.7395 | 0.4970 | 3.7M | 18.0 |
| 21 | AFM | 0.7282 | 0.5092 | 3.4M | 19.7 |

**Takeaways**

- Explicit feature-interaction deep models (**DeepFEFM, FiBiNET, DeepFM,
  xDeepFM**) lead once there is enough data. An earlier 200k / 3-epoch run that
  excluded the slow interaction models had MLR/NFM on top — a small-data
  artifact, not a real ranking.
- The top 13 models sit in a tight 0.778–0.782 band (1 epoch is not converged),
  but are clearly separated from the tail (AFM 0.728, EDCN 0.740).
- **Cost outlier: ONN** — 77M params and 245s to train for only 0.76 AUC.
  DeepFEFM/FiBiNET/FwFM/FGCNN cost ~100–130s (≈10× DeepFM's 13s); only
  DeepFEFM/FiBiNET earn it with a top-of-board AUC.

## Multitask — real UCI Census-Income (official split)

**Official train/test partition: 199,523 train / 99,762 test** (loader uses the
shipped `census-income.test` when present, not a random reshuffle). 3 epochs,
batch 1024. Tasks: `label_income` (>50k), `label_marital` (never-married).

| Rank | Model | income AUC | marital AUC | mean_AUC | Params | Train(s) |
| ---- | ----- | ---------- | ----------- | -------- | ------ | -------- |
| 1 | SharedBottom | 0.9465 | 0.9943 | 0.9704 | 116K | 8.1 |
| 2 | MMOE | 0.9465 | 0.9942 | 0.9704 | 308K | 11.0 |
| 3 | PLE | 0.9462 | 0.9943 | 0.9703 | 424K | 17.3 |
| 4 | ESMM | 0.5632 | 0.9751 | 0.7692 | 211K | 8.5 |

**Takeaways**

- On Census the three general multitask architectures are statistically
  indistinguishable (mean_AUC 0.9703–0.9704); the task is easy enough that the
  extra capacity of MMOE/PLE buys nothing here. SharedBottom wins on cost.
- **ESMM is far behind** (income AUC 0.56) — expected: it assumes
  sequentially-dependent CTR→CTCVR tasks, which Census does not satisfy. Its
  numbers are not directly comparable.

## Sequence — real MovieLens-25M `ml25m_seq_2m.csv` (current headline)

**2,284,710** behavior samples built from the first 2.3M ratings of MovieLens-25M
(per user, sorted by timestamp: target = current movie, history = up to 12 prior
movies, label = rating ≥ 4, cate = first genre). 29,273 items · random split
(seed=2020, test_size=0.2). 1 epoch, batch 1024, embedding-dim 8, **CPU**.

| Rank | Model | AUC | LogLoss | Params | Train(s) |
| ---- | ----- | --- | ------- | ------ | -------- |
| 1 | DIN | **0.8021** | 0.5408 | 419,194 | 16.3 |
| 2 | DSIN | 0.7951 | 0.5471 | 436,635 | 77.2 |
| 3 | BST | 0.7943 | 0.5487 | 419,162 | 45.0 |
| - | DIEN | skipped | — | — | — | GRU/AUGRU need legacy TF1 RNN APIs; unsupported on TF≥2.0 |

- On real sequences all three models land at **0.79–0.80 AUC** — a genuine signal,
  unlike the ~0.5 synthetic track (which only checks build/train/predict + cost).
- **DIN is the value pick**: top AUC *and* fastest (16s). DSIN edges out BST but
  costs ~4.7× the train time — its session structure earns nothing at 1 epoch.
- MovieLens-25M has no user demographics, so `gender` (a minor sparse feature) is
  synthesized deterministically from `user_id`; the sequence signal is fully real.

## Sequence — synthetic

DIN / BST / DSIN build, train and predict; **DIEN is skipped on TF≥2.0** (its
GRU/AUGRU layers need legacy TF1 private RNN APIs). Synthetic AUC is ~0.5 by
design — the synthetic track verifies the models run and compares their cost,
not predictive power. Use `--seq-source movielens`/Amazon for real numbers.

## Reproduce

See `.claude/skills/deepctr-benchmark` for the exact commands, or:

```bash
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track single \
  --data-path benchmarks/data/criteo_x1_500k.csv \
  --epochs 1 --batch-size 1024 --val-split 0 --seed 2020

CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track multitask \
  --data-path benchmarks/data/census-income.data --epochs 3 --batch-size 1024

# Sequence on real MovieLens-25M (build ml25m_seq_2m.csv first — see the skill)
CUDA_VISIBLE_DEVICES="" python -m benchmarks.benchmark --track sequence \
  --seq-source movielens --data-path benchmarks/data/ml25m_seq_2m.csv \
  --epochs 1 --batch-size 1024 --embedding-dim 8
```
