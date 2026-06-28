"""Fast smoke test for the benchmark suite (benchmarks/).

Runs the real track runners on bundled / tiny-synthetic data for one epoch with
two models each, asserting metrics are produced and leaderboards are written.
Kept lean so it adds little to CI time.
"""
import os

# DeepCTR needs the Keras 2 API; on TF>=2.16 that means legacy Keras. Set it
# before TensorFlow is imported (mirrors benchmarks/__init__.py and CI).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from benchmarks.benchmark import build_parser, run_multitask, run_sequence, run_single


def _args(track, tmp_path, **overrides):
    argv = ["--track", track, "--quick", "--epochs", "1",
            "--embedding-dim", "4", "--output-dir", str(tmp_path)]
    args = build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_single_task_track(tmp_path):
    results = run_single(_args("single", tmp_path))
    assert any(r.status == "ok" and "AUC" in r.metrics for r in results)
    assert (tmp_path / "single_criteo_sample.csv").exists()
    assert (tmp_path / "single_criteo_sample.md").exists()


def test_multitask_track(tmp_path):
    results = run_multitask(_args("multitask", tmp_path))
    assert any(r.status == "ok" and "mean_AUC" in r.metrics for r in results)
    assert (tmp_path / "multitask_census.csv").exists()


def test_sequence_track(tmp_path):
    results = run_sequence(_args("sequence", tmp_path))
    by_model = {r.model: r for r in results}
    # DIN must train and produce an AUC; DIEN is cleanly skipped on TF>=2.0.
    assert by_model["DIN"].status == "ok" and "AUC" in by_model["DIN"].metrics
    assert (tmp_path / "sequence_synthetic_seq.csv").exists()


def test_temporal_split_holds_out_most_recent_rows(tmp_path):
    """--temporal-split must hold out the most recent rows (by --time-col),
    never random ones, so no future information leaks into training."""
    import numpy as np
    import pandas as pd

    from benchmarks.datasets.criteo import COLUMNS, load_criteo

    n = 100
    df = pd.DataFrame({c: [0] * n for c in COLUMNS})
    df["ts"] = list(range(n))           # ascending time
    df["label"] = [0] * 80 + [1] * 20   # the 20 most-recent rows are the positives
    # Shuffle file order so a correct split must rely on `ts`, not row position.
    df = df.iloc[np.random.RandomState(0).permutation(n)].reset_index(drop=True)
    path = tmp_path / "criteo_ts.csv"
    df.to_csv(path, index=False)

    data = load_criteo(data_path=str(path), temporal_split=True, time_col="ts", test_size=0.2)

    assert len(data.train_y) == 80 and len(data.test_y) == 20
    # Most-recent 20% (ts 80..99 -> all positives) must be exactly the test set.
    assert set(data.test_y.flatten().tolist()) == {1}
    assert set(data.train_y.flatten().tolist()) == {0}


def _census_row(income, marital):
    from benchmarks.datasets.census import COLUMN_NAMES

    vals = ["0"] * len(COLUMN_NAMES)
    vals[COLUMN_NAMES.index("marital_stat")] = marital
    vals[COLUMN_NAMES.index("income_50k")] = income
    return ",".join(vals)


def test_census_uses_official_test_partition(tmp_path):
    """When a sibling .test file exists, load_census must use that official
    partition rather than reshuffling a random split out of .data."""
    from benchmarks.datasets.census import load_census

    train_rows = [_census_row(" 50000+.", " Never married")] * 10 + \
                 [_census_row(" - 50000.", " Married")] * 10          # 20 train rows
    test_rows = [_census_row(" - 50000.", " Married")] * 7            # 7 test rows
    (tmp_path / "mini.data").write_text("\n".join(train_rows) + "\n")
    (tmp_path / "mini.test").write_text("\n".join(test_rows) + "\n")

    data = load_census(data_path=str(tmp_path / "mini.data"))

    # Official split -> exactly the .test rows (a random 0.2 split would give 5).
    assert len(data.train_y[0]) == 20
    assert len(data.test_y[0]) == 7


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
