"""Shared utilities for the DeepCTR benchmark suite: reproducibility helpers,
the data/result containers passed between modules, and leaderboard writers.
"""
from __future__ import absolute_import, division, print_function

import csv
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .metrics import HIGHER_IS_BETTER


# --------------------------------------------------------------------------- #
# Containers passed from dataset loaders to the track runners.
# --------------------------------------------------------------------------- #
@dataclass
class SingleTaskData:
    name: str
    task: str  # 'binary' or 'regression'
    linear_feature_columns: list
    dnn_feature_columns: list
    feature_names: list
    train_input: dict
    train_y: np.ndarray
    test_input: dict
    test_y: np.ndarray
    target: str = "label"


@dataclass
class MultiTaskData:
    name: str
    dnn_feature_columns: list
    feature_names: list
    train_input: dict
    train_y: List[np.ndarray]
    test_input: dict
    test_y: List[np.ndarray]
    task_types: tuple
    task_names: tuple


@dataclass
class SequenceData:
    name: str
    feature_columns: list
    behavior_feature_list: list
    train_input: dict
    train_y: np.ndarray
    test_input: dict
    test_y: np.ndarray
    # >0 only for DSIN-style sessionized data; the number of sessions.
    sess_max_count: int = 0


@dataclass
class ModelResult:
    model: str
    status: str  # 'ok' | 'failed' | 'skipped'
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Optional[int] = None
    train_seconds: Optional[float] = None
    note: str = ""


# --------------------------------------------------------------------------- #
# Reproducibility.
# --------------------------------------------------------------------------- #
def set_seeds(seed=2020):
    """Seed python, numpy and TensorFlow RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover - tf always present in practice
        pass


def reset_session():
    """Clear the Keras backend between models so a 20+ model sweep doesn't
    accumulate graph state / leak memory."""
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:  # pragma: no cover
        pass


def split_dict_xy(x, y, test_size=0.2, seed=2020):
    """Index-based train/test split for a dict-of-arrays input and a 1-D ``y``.

    Used by the sequence track, whose features are built directly as numpy
    arrays rather than a DataFrame. Every value in ``x`` has length ``len(y)``.
    """
    y = np.asarray(y)
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(1, int(round(n * (1 - test_size))))
    tr, te = idx[:cut], idx[cut:]
    if len(te) == 0:  # degenerate tiny-sample case: reuse train as test
        te = tr
    train_x = {k: np.asarray(v)[tr] for k, v in x.items()}
    test_x = {k: np.asarray(v)[te] for k, v in x.items()}
    return train_x, y[tr], test_x, y[te]


# --------------------------------------------------------------------------- #
# Leaderboard rendering / persistence.
# --------------------------------------------------------------------------- #
def _metric_keys(results):
    keys = []
    for r in results:
        for k in r.metrics:
            if k not in keys:
                keys.append(k)
    return keys


def _is_nan(v):
    return isinstance(v, float) and np.isnan(v)


def sort_results(results, primary_metric):
    """Return results with ``status=='ok'`` first, ranked by ``primary_metric``
    (direction from HIGHER_IS_BETTER), then skipped/failed rows."""
    higher = HIGHER_IS_BETTER.get(primary_metric, True)

    def key(r):
        v = r.metrics.get(primary_metric)
        if v is None or _is_nan(v):
            return (1, 0.0)  # undefined metric -> bottom of the ok group
        return (0, -v if higher else v)

    ok = sorted([r for r in results if r.status == "ok"], key=key)
    rest = [r for r in results if r.status != "ok"]
    return ok + rest


def _fmt(v):
    if v is None:
        return ""
    if _is_nan(v):
        return "nan"
    if isinstance(v, float):
        return "%.4f" % v
    return str(v)


def format_markdown(results, primary_metric):
    metric_keys = _metric_keys(results)
    headers = ["Rank", "Model", "Status"] + metric_keys + ["Params", "Train(s)", "Note"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    rank = 0
    for r in results:
        if r.status == "ok":
            rank += 1
            rank_s = str(rank)
        else:
            rank_s = "-"
        row = [rank_s, r.model, r.status]
        row += [_fmt(r.metrics.get(k)) for k in metric_keys]
        row.append("" if r.params is None else "{:,}".format(r.params))
        row.append("" if r.train_seconds is None else "%.1f" % r.train_seconds)
        row.append(r.note or "")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_results(results, out_dir, track, dataset, primary_metric):
    """Write ``<track>_<dataset>.csv`` and ``.md`` to ``out_dir``; return paths."""
    os.makedirs(out_dir, exist_ok=True)
    base = "%s_%s" % (track, dataset)
    metric_keys = _metric_keys(results)

    csv_path = os.path.join(out_dir, base + ".csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["model", "status"] + metric_keys + ["params", "train_seconds", "note"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "model": r.model,
                "status": r.status,
                "params": r.params,
                "train_seconds": None if r.train_seconds is None else round(r.train_seconds, 2),
                "note": r.note,
            }
            for k in metric_keys:
                row[k] = r.metrics.get(k)
            w.writerow(row)

    md_path = os.path.join(out_dir, base + ".md")
    with open(md_path, "w") as f:
        f.write("# DeepCTR benchmark - %s track on `%s`\n\n" % (track, dataset))
        f.write(format_markdown(results, primary_metric) + "\n")

    return csv_path, md_path


def print_leaderboard(results, primary_metric, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(format_markdown(results, primary_metric))
    print("")
