"""Sequence-track data for DIN / DIEN / DSIN / BST.

A single behavior generator produces per-sample ``(user, gender, target item,
target cate, pay_score, history[(item, cate)...], label)`` tuples; two
formatters turn those into the exact feature layouts the models expect:

* ``_to_din_data``  -> ``hist_item_id`` / ``hist_cate_id`` + ``seq_length``
  (DIN, BST, DIEN), behavior list ``["item_id", "cate_id"]``.
* ``_to_dsin_data`` -> ``sess_k_item`` / ``sess_k_cate_id`` + ``sess_length``
  (DSIN), behavior list ``["item", "cate_id"]``.

Two sources:

* ``synthetic`` (default): deterministic generator with an *injected* learnable
  signal (label depends on user-cate preference and whether the target item is
  in the history), so AUC is meaningfully > 0.5 and models are comparable.
* ``movielens``: real sequences built from a MovieLens file (per user, sorted by
  timestamp, target = current movie, history = prior movies, label = rating>=4,
  cate = first genre). Tiny on the bundled 200-row sample; real on MovieLens-1M
  via ``data_path``.
"""
from __future__ import absolute_import, division, print_function

import os

import numpy as np

from deepctr.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names

from ..common import SequenceData, split_dict_xy

_HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "examples")


# --------------------------------------------------------------------------- #
# Behavior generators -> parallel lists + (n_users, n_items, n_cates).
# --------------------------------------------------------------------------- #
def _generate_synthetic(n_samples, n_users, n_items, n_cates, max_hist, seed):
    rng = np.random.RandomState(seed)
    item2cate = rng.randint(1, n_cates + 1, size=n_items + 1)
    item2cate[0] = 0  # 0 is the padding/mask id
    pref = rng.rand(n_users, n_cates + 1)  # latent user x cate preference
    fav_cate = pref.argmax(axis=1)  # each user's most-preferred category

    users, genders, items, cates, scores, histories, ys = [], [], [], [], [], [], []
    for _ in range(n_samples):
        u = int(rng.randint(0, n_users))
        g = int(rng.randint(0, 2))
        length = int(rng.randint(1, max_hist + 1))
        hist_items = rng.randint(1, n_items + 1, size=length)
        hist = [(int(it), int(item2cate[it])) for it in hist_items]
        # Half the time the candidate is drawn from the user's own history, so
        # "is the target in the behavior sequence" is a strong, balanced signal
        # that attention-based models are designed to capture.
        if rng.rand() < 0.5:
            tgt = int(rng.choice(hist_items))
        else:
            tgt = int(rng.randint(1, n_items + 1))
        tcate = int(item2cate[tgt])
        in_hist = 1.0 if tgt in hist_items else 0.0
        logit = -1.2 + 2.6 * in_hist + 1.3 * (tcate == fav_cate[u]) + 0.2 * rng.randn()
        p = 1.0 / (1.0 + np.exp(-logit))
        users.append(u)
        genders.append(g)
        items.append(tgt)
        cates.append(tcate)
        scores.append(float(rng.rand()))
        histories.append(hist)
        ys.append(int(rng.rand() < p))
    return users, genders, items, cates, scores, histories, ys, (n_users, n_items, n_cates)


def _generate_movielens(path, max_hist):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(path)
    df = df.sort_values(["user_id", "timestamp"])
    df["movie_idx"] = LabelEncoder().fit_transform(df["movie_id"]) + 1  # 0 = mask
    df["cate"] = LabelEncoder().fit_transform(df["genres"].apply(lambda g: str(g).split("|")[0])) + 1
    df["user_idx"] = LabelEncoder().fit_transform(df["user_id"])
    df["gender_idx"] = (df["gender"] == "M").astype(int)

    n_users = int(df["user_idx"].max()) + 1
    n_items = int(df["movie_idx"].max())
    n_cates = int(df["cate"].max())

    users, genders, items, cates, scores, histories, ys = [], [], [], [], [], [], []
    for _, grp in df.groupby("user_idx"):
        seq = list(zip(grp["movie_idx"], grp["cate"], grp["rating"], grp["gender_idx"], grp["user_idx"]))
        for i in range(1, len(seq)):
            it, ct, rt, gd, u = seq[i]
            hist = [(int(a), int(b)) for a, b, _, _, _ in seq[max(0, i - max_hist):i]]
            if not hist:
                continue
            users.append(int(u))
            genders.append(int(gd))
            items.append(int(it))
            cates.append(int(ct))
            scores.append(0.0)
            histories.append(hist)
            ys.append(int(rt >= 4))
    return users, genders, items, cates, scores, histories, ys, (n_users, n_items, n_cates)


# --------------------------------------------------------------------------- #
# Formatters -> SequenceData.
# --------------------------------------------------------------------------- #
def _to_din_data(samples, maxlen, embedding_dim, test_size, seed, name):
    users, genders, items, cates, scores, histories, ys, (n_users, n_items, n_cates) = samples
    n = len(ys)
    hist_item = np.zeros((n, maxlen), dtype="int64")
    hist_cate = np.zeros((n, maxlen), dtype="int64")
    seq_length = np.zeros((n,), dtype="int64")
    for i, hist in enumerate(histories):
        hist = hist[-maxlen:]
        seq_length[i] = len(hist)
        for j, (it, ct) in enumerate(hist):
            hist_item[i, j] = it
            hist_cate[i, j] = ct

    feature_columns = [
        SparseFeat("user", n_users, embedding_dim=embedding_dim),
        SparseFeat("gender", 2, embedding_dim=embedding_dim),
        SparseFeat("item_id", n_items + 1, embedding_dim=embedding_dim),
        SparseFeat("cate_id", n_cates + 1, embedding_dim=embedding_dim),
        DenseFeat("pay_score", 1),
        VarLenSparseFeat(
            SparseFeat("hist_item_id", n_items + 1, embedding_dim=embedding_dim, embedding_name="item_id"),
            maxlen=maxlen, length_name="seq_length"),
        VarLenSparseFeat(
            SparseFeat("hist_cate_id", n_cates + 1, embedding_dim=embedding_dim, embedding_name="cate_id"),
            maxlen=maxlen, length_name="seq_length"),
    ]
    feature_dict = {
        "user": np.array(users), "gender": np.array(genders),
        "item_id": np.array(items), "cate_id": np.array(cates),
        "pay_score": np.array(scores, dtype="float32"),
        "hist_item_id": hist_item, "hist_cate_id": hist_cate, "seq_length": seq_length,
    }
    x = {k: feature_dict[k] for k in get_feature_names(feature_columns)}
    train_x, train_y, test_x, test_y = split_dict_xy(x, np.array(ys).reshape(-1, 1), test_size, seed)
    return SequenceData(name, feature_columns, ["item_id", "cate_id"],
                        train_x, train_y, test_x, test_y, sess_max_count=0)


def _to_dsin_data(samples, sess_len, sess_max_count, embedding_dim, test_size, seed, name):
    users, genders, items, cates, scores, histories, ys, (n_users, n_items, n_cates) = samples
    n = len(ys)
    sess_item = [np.zeros((n, sess_len), dtype="int64") for _ in range(sess_max_count)]
    sess_cate = [np.zeros((n, sess_len), dtype="int64") for _ in range(sess_max_count)]
    sess_length = np.zeros((n,), dtype="int64")
    for i, hist in enumerate(histories):
        chunks = [hist[k:k + sess_len] for k in range(0, len(hist), sess_len)][:sess_max_count]
        sess_length[i] = len(chunks)
        for s, chunk in enumerate(chunks):
            for j, (it, ct) in enumerate(chunk):
                sess_item[s][i, j] = it
                sess_cate[s][i, j] = ct

    feature_columns = [
        SparseFeat("user", n_users, embedding_dim=embedding_dim),
        SparseFeat("gender", 2, embedding_dim=embedding_dim),
        SparseFeat("item", n_items + 1, embedding_dim=embedding_dim),
        SparseFeat("cate_id", n_cates + 1, embedding_dim=embedding_dim),
        DenseFeat("pay_score", 1),
    ]
    feature_dict = {
        "user": np.array(users), "gender": np.array(genders),
        "item": np.array(items), "cate_id": np.array(cates),
        "pay_score": np.array(scores, dtype="float32"),
    }
    for k in range(sess_max_count):
        feature_columns += [
            VarLenSparseFeat(
                SparseFeat("sess_%d_item" % k, n_items + 1, embedding_dim=embedding_dim, embedding_name="item"),
                maxlen=sess_len),
            VarLenSparseFeat(
                SparseFeat("sess_%d_cate_id" % k, n_cates + 1, embedding_dim=embedding_dim, embedding_name="cate_id"),
                maxlen=sess_len),
        ]
        feature_dict["sess_%d_item" % k] = sess_item[k]
        feature_dict["sess_%d_cate_id" % k] = sess_cate[k]

    x = {k: feature_dict[k] for k in get_feature_names(feature_columns)}
    x["sess_length"] = sess_length  # DSIN reads this directly (not a feature column)
    train_x, train_y, test_x, test_y = split_dict_xy(x, np.array(ys).reshape(-1, 1), test_size, seed)
    return SequenceData(name, feature_columns, ["item", "cate_id"],
                        train_x, train_y, test_x, test_y, sess_max_count=sess_max_count)


def make_sequence_datasets(source="synthetic", n_samples=2000, n_users=100, n_items=50,
                           n_cates=8, max_hist=8, sess_max_count=3, sess_len=4,
                           embedding_dim=8, test_size=0.2, seed=2020, data_path=None):
    """Build both sequence views from one behavior source.

    Returns ``(din_data, dsin_data)``: ``din_data`` feeds DIN/BST/DIEN,
    ``dsin_data`` feeds DSIN.
    """
    if source == "movielens":
        path = data_path or os.path.join(EXAMPLES_DIR, "movielens_sample.txt")
        samples = _generate_movielens(path, max_hist=sess_max_count * sess_len)
        name = "movielens_seq"
    else:
        samples = _generate_synthetic(n_samples, n_users, n_items, n_cates,
                                      max_hist=sess_max_count * sess_len, seed=seed)
        name = "synthetic_seq"

    din_data = _to_din_data(samples, max_hist, embedding_dim, test_size, seed, name)
    dsin_data = _to_dsin_data(samples, sess_len, sess_max_count, embedding_dim, test_size, seed, name)
    return din_data, dsin_data
