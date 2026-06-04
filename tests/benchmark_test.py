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


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
