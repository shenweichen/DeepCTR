"""Criteo dataset loader for the single-task (binary CTR) track.

Three sources, one preprocessing path (ported from
``examples/run_classification_criteo.py`` and ``..._hash.py``):

* ``bundled``  - the 200-row ``examples/criteo_sample.txt`` (offline, deterministic).
* ``download`` - the Criteo "DAC" ~100k-row sample, fetched once into
  ``benchmarks/data/``; falls back to the bundled sample if the network/URL fails.
* ``data_path``- any full Criteo file (``train.txt`` / ``dac_sample.txt``: TSV,
  no header, ``label`` + I1..I13 + C1..C26).
"""
from __future__ import absolute_import, division, print_function

import os
import tarfile
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import DenseFeat, SparseFeat, get_feature_names

from ..common import SingleTaskData

_HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "examples")
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")

DENSE_FEATURES = ["I%d" % i for i in range(1, 14)]
SPARSE_FEATURES = ["C%d" % i for i in range(1, 27)]
COLUMNS = ["label"] + DENSE_FEATURES + SPARSE_FEATURES

# Criteo's historical public DAC-sample mirror (criteo-labs S3) is offline, and
# there is no stable drop-in auto-download anymore (the data now lives as large
# multi-file datasets on Hugging Face / Criteo AI Lab). So there is no default
# auto-download: use --data-path for real data, or pass your own tar.gz mirror
# via --download-url. The loader degrades gracefully to the bundled sample.
DEFAULT_DAC_URL = None


def _read_bundled():
    return pd.read_csv(os.path.join(EXAMPLES_DIR, "criteo_sample.txt"))


def _read_criteo_file(path):
    """Read a Criteo file, auto-detecting separator and header."""
    with open(path) as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    if first.lower().startswith("label"):
        return pd.read_csv(path, sep=sep)
    return pd.read_csv(path, sep=sep, header=None, names=COLUMNS)


def _print_manual_instructions():
    print(
        "[criteo] No public auto-download is available (Criteo's DAC-sample mirror "
        "is offline).\n"
        "         For a real benchmark, fetch a Criteo dataset and pass it via "
        "--data-path, e.g.:\n"
        "           - Criteo_x1 split:  https://huggingface.co/datasets/reczoo/Criteo_x1\n"
        "           - Full click logs:  https://huggingface.co/datasets/criteo/CriteoClickLogs\n"
        "           - Criteo AI Lab:    https://ailab.criteo.com/ressources/\n"
        "         then:  python -m benchmarks.benchmark --track single --data-path <file.csv>\n"
        "         (or pass a working tar.gz mirror via --download-url)."
    )


def _download_dac_sample(url, dest_dir):
    """Download + extract ``dac_sample.txt`` into ``dest_dir``; return its path.

    Cached: a second call reuses the extracted file. Raises on failure (the
    caller handles fallback).
    """
    os.makedirs(dest_dir, exist_ok=True)
    target_txt = os.path.join(dest_dir, "dac_sample.txt")
    if os.path.exists(target_txt):
        return target_txt

    tar_path = os.path.join(dest_dir, "dac_sample.tar.gz")
    print("[criteo] downloading DAC sample from %s ..." % url)
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("dac_sample.txt"))
        member.name = os.path.basename(member.name)  # strip any leading path
        tar.extract(member, dest_dir)
    return target_txt


def load_criteo(source="bundled", data_path=None, embedding_dim=8, use_hash=False,
                hash_buckets=1000, test_size=0.2, seed=2020, max_rows=None,
                download_url=DEFAULT_DAC_URL, temporal_split=False, time_col=None):
    """Load and preprocess Criteo into a :class:`SingleTaskData` (task='binary').

    By default the train/test split is random (the Kaggle DAC / Criteo_x1 data
    is anonymised and shuffled, with no timestamp, so this is leakage-free and
    standard). For logs that *do* carry time, pass ``temporal_split=True`` to
    hold out the most recent ``test_size`` fraction instead: rows are sorted by
    ``time_col`` when given, otherwise the file's existing order is trusted, and
    the tail becomes the test set -- so no future row can leak into training.
    """
    if data_path is not None:
        df = _read_criteo_file(data_path)
        name = "criteo_custom"
    elif source == "download":
        path = None
        if download_url:
            try:
                path = _download_dac_sample(download_url, DATA_DIR)
            except Exception as exc:  # network down, URL moved, bad archive, ...
                print("[criteo] download failed (%s); using bundled sample instead." % exc)
                _print_manual_instructions()
        else:
            print("[criteo] --source download requested but no download URL is "
                  "configured (the historical DAC-sample mirror is offline).")
            _print_manual_instructions()
        if path:
            df = _read_criteo_file(path)
            name = "criteo_dac100k"
        else:
            df, name = _read_bundled(), "criteo_sample"
    else:
        df, name = _read_bundled(), "criteo_sample"

    if max_rows:
        df = df.head(int(max_rows)).copy()

    df[SPARSE_FEATURES] = df[SPARSE_FEATURES].fillna("-1")
    df[DENSE_FEATURES] = df[DENSE_FEATURES].fillna(0)

    if use_hash:
        # Hash the raw string values inside the model (no label encoding).
        for feat in SPARSE_FEATURES:
            df[feat] = df[feat].astype(str)
        sparse_cols = [
            SparseFeat(feat, hash_buckets, embedding_dim=embedding_dim, use_hash=True, dtype="string")
            for feat in SPARSE_FEATURES
        ]
    else:
        for feat in SPARSE_FEATURES:
            df[feat] = LabelEncoder().fit_transform(df[feat])
        sparse_cols = [
            SparseFeat(feat, vocabulary_size=int(df[feat].max()) + 1, embedding_dim=embedding_dim)
            for feat in SPARSE_FEATURES
        ]

    df[DENSE_FEATURES] = MinMaxScaler((0, 1)).fit_transform(df[DENSE_FEATURES])
    dense_cols = [DenseFeat(feat, 1) for feat in DENSE_FEATURES]

    feature_columns = sparse_cols + dense_cols
    feature_names = get_feature_names(feature_columns)

    if temporal_split:
        # Chronological hold-out: the most recent rows become the test set so no
        # future information leaks into training. Sort by time_col when present;
        # otherwise assume the file is already in chronological order.
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col, kind="mergesort")  # stable
        elif time_col:
            print("[criteo] --time-col %r not found; assuming the file is already "
                  "in chronological order." % time_col)
        cut = max(1, int(round(len(df) * (1 - test_size))))
        train, test = df.iloc[:cut], df.iloc[cut:]
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=seed)
    train_input = {n: train[n].values for n in feature_names}
    test_input = {n: test[n].values for n in feature_names}

    return SingleTaskData(
        name=name,
        task="binary",
        linear_feature_columns=feature_columns,
        dnn_feature_columns=feature_columns,
        feature_names=feature_names,
        train_input=train_input,
        # 2-D targets: DeepCTR's PredictionLayer emits shape (None, 1) and
        # Keras 3 (TF>=2.16) requires target rank to match output rank.
        train_y=train["label"].values.reshape(-1, 1),
        test_input=test_input,
        test_y=test["label"].values.reshape(-1, 1),
        target="label",
    )
