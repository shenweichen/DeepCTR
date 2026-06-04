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

# Historical Criteo Labs mirror of the 100k-row DAC sample. May be offline; the
# loader degrades gracefully to the bundled sample when it is.
DEFAULT_DAC_URL = "https://s3-eu-west-1.amazonaws.com/criteo-labs/dac_sample.tar.gz"


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


def _print_manual_instructions(url):
    print(
        "[criteo] To benchmark on real data, download the Criteo DAC sample "
        "manually and pass it via --data-path:\n"
        "           curl -L -o dac_sample.tar.gz '%s'\n"
        "           tar -xzf dac_sample.tar.gz   # -> dac_sample.txt\n"
        "         (or point --data-path at any full Criteo train.txt)." % url
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
                download_url=DEFAULT_DAC_URL):
    """Load and preprocess Criteo into a :class:`SingleTaskData` (task='binary')."""
    if data_path is not None:
        df = _read_criteo_file(data_path)
        name = "criteo_custom"
    elif source == "download":
        path = None
        try:
            path = _download_dac_sample(download_url, DATA_DIR)
        except Exception as exc:  # network down, URL moved, bad archive, ...
            print("[criteo] download failed (%s); using bundled sample instead." % exc)
            _print_manual_instructions(download_url)
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
