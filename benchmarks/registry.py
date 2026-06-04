"""Model registry: maps a model name to a builder that hides each model's
constructor quirks behind a uniform call. The track runners in ``benchmark.py``
just look models up here.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from packaging import version

from deepctr.feature_column import DenseFeat, SparseFeat
from deepctr.models import (AFM, BST, CCPM, DCN, DIEN, DIFM, DIN, DSIN, EDCN, FGCNN, FLEN, FNN,
                            FwFM, IFM, MLR, MMOE, NFM, ONN, PLE, PNN, WDL, AutoInt, DCNMix,
                            DeepFEFM, DeepFM, ESMM, FiBiNET, SharedBottom, xDeepFM)

_TF2 = version.parse(tf.__version__) >= version.parse("2.0.0")
_TF28 = version.parse(tf.__version__) >= version.parse("2.8.0")


class SkipModel(Exception):
    """Raised by a builder when a model is unsupported in the current env
    (recorded as status='skipped' rather than 'failed')."""


def _sparse_only(feature_columns):
    """Drop DenseFeat. AFM / CCPM / EDCN declare support_dense=False and only
    accept sparse (embedding) features."""
    return [fc for fc in feature_columns if not isinstance(fc, DenseFeat)]


def _regroup(feature_columns, n_groups=3):
    """Round-robin SparseFeat into ``n_groups`` field groups. FLEN's field-wise
    bi-interaction needs >=2 groups; the generic loader puts everything in one.
    The grouping is synthetic (no field taxonomy in Criteo)."""
    out, gi = [], 0
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            out.append(fc._replace(group_name="g%d" % (gi % n_groups)))
            gi += 1
        else:
            out.append(fc)
    return out


# --------------------------------------------------------------------------- #
# Single-task: builder(linear_fc, dnn_fc, task) -> tf.keras.Model
# --------------------------------------------------------------------------- #
SINGLE_TASK_MODELS = {
    "DeepFM": lambda lin, dnn, task: DeepFM(lin, dnn, task=task),
    "xDeepFM": lambda lin, dnn, task: xDeepFM(lin, dnn, task=task),
    "DCN": lambda lin, dnn, task: DCN(lin, dnn, task=task),
    "DCNMix": lambda lin, dnn, task: DCNMix(lin, dnn, task=task),
    "AutoInt": lambda lin, dnn, task: AutoInt(lin, dnn, task=task),
    "FiBiNET": lambda lin, dnn, task: FiBiNET(lin, dnn, task=task),
    "AFM": lambda lin, dnn, task: AFM(_sparse_only(lin), _sparse_only(dnn), task=task),  # sparse-only
    "NFM": lambda lin, dnn, task: NFM(lin, dnn, task=task),
    "PNN": lambda lin, dnn, task: PNN(dnn, task=task),  # DNN-only, no linear part
    "WDL": lambda lin, dnn, task: WDL(lin, dnn, task=task),
    "FNN": lambda lin, dnn, task: FNN(lin, dnn, task=task),
    "FwFM": lambda lin, dnn, task: FwFM(lin, dnn, task=task),
    "CCPM": lambda lin, dnn, task: CCPM(_sparse_only(lin), _sparse_only(dnn), task=task),  # sparse-only
    "ONN": lambda lin, dnn, task: ONN(lin, dnn, task=task),
    "FGCNN": lambda lin, dnn, task: FGCNN(lin, dnn, task=task),
    "FLEN": lambda lin, dnn, task: FLEN(_regroup(lin), _regroup(dnn), task=task),  # needs >=2 field groups
    "IFM": lambda lin, dnn, task: IFM(lin, dnn, task=task),
    "DIFM": lambda lin, dnn, task: DIFM(lin, dnn, task=task),
    "DeepFEFM": lambda lin, dnn, task: DeepFEFM(lin, dnn, task=task),
    "MLR": lambda lin, dnn, task: MLR(lin, lin, task=task),  # region == base
    "EDCN": lambda lin, dnn, task: EDCN(_sparse_only(lin), _sparse_only(dnn), task=task),  # sparse-only
}

# --------------------------------------------------------------------------- #
# Multitask: builder(dnn_fc, task_types, task_names) -> tf.keras.Model
# --------------------------------------------------------------------------- #
MULTITASK_MODELS = {
    "SharedBottom": lambda dnn, types, names: SharedBottom(dnn, task_types=types, task_names=names),
    "ESMM": lambda dnn, types, names: ESMM(dnn, task_types=types, task_names=names),
    "MMOE": lambda dnn, types, names: MMOE(dnn, task_types=types, task_names=names),
    "PLE": lambda dnn, types, names: PLE(dnn, task_types=types, task_names=names),
}


# --------------------------------------------------------------------------- #
# Sequence: each entry says which data view it needs and how to build it.
# --------------------------------------------------------------------------- #
def _behavior_embedding_dim(data):
    target = data.behavior_feature_list[0]
    for fc in data.feature_columns:
        if getattr(fc, "name", None) == target:
            return fc.embedding_dim
    return 8


def _behavior_embedding_sum(data):
    """Total embedding width of the behavior features (DSIN concatenates them)."""
    total = 0
    for fc in data.feature_columns:
        if getattr(fc, "name", None) in data.behavior_feature_list:
            total += fc.embedding_dim
    return total


def _largest_divisor_leq(n, cap):
    for h in range(min(n, cap), 0, -1):
        if n % h == 0:
            return h
    return 1


def _build_din(data, task):
    # Dice activation has serialization/run issues on modern TF; the DIN test
    # itself switches to 'sigmoid' on TF>=2.8.
    att_activation = "sigmoid" if _TF28 else "dice"
    return DIN(data.feature_columns, data.behavior_feature_list,
               att_activation=att_activation, task=task)


def _build_bst(data, task):
    emb = _behavior_embedding_dim(data)
    head = _largest_divisor_leq(emb, 8)  # Transformer needs emb_dim % head == 0
    return BST(data.feature_columns, data.behavior_feature_list, att_head_num=head, task=task)


def _build_dien(data, task):
    if _TF2:
        raise SkipModel("DIEN's GRU/AUGRU layers depend on legacy TF1 private RNN "
                        "APIs; unsupported on TensorFlow>=2.0 (its own test skips on TF2).")
    return DIEN(data.feature_columns, data.behavior_feature_list, gru_type="GRU", task=task)


def _build_dsin(data, task):
    # DSIN requires sum(behavior embedding dims) == att_embedding_size * att_head_num.
    hist_emb = _behavior_embedding_sum(data)
    head = _largest_divisor_leq(hist_emb, 8)
    att_emb = hist_emb // head
    return DSIN(data.feature_columns, data.behavior_feature_list,
                sess_max_count=data.sess_max_count,
                att_embedding_size=att_emb, att_head_num=head, task=task)


# view: which SequenceData to feed ('din' for DIN/BST/DIEN, 'dsin' for DSIN)
SEQUENCE_MODELS = {
    "DIN": {"view": "din", "build": _build_din},
    "BST": {"view": "din", "build": _build_bst},
    "DIEN": {"view": "din", "build": _build_dien},
    "DSIN": {"view": "dsin", "build": _build_dsin},
}


def select(names, include=None, exclude=None):
    """Filter an ordered list of model names by --models / --exclude."""
    out = list(names)
    if include:
        inc = set(include)
        out = [n for n in out if n in inc]
    if exclude:
        exc = set(exclude)
        out = [n for n in out if n not in exc]
    return out
