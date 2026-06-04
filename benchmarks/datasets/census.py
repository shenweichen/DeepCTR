"""Census-Income loader for the multitask track (ported from
``examples/run_mtl.py``).

Two binary tasks are derived: ``label_income`` (>50k) and ``label_marital``
(never-married). DNN-only feature columns (multitask models take no linear
part). ``data_path`` accepts the full UCI Census-Income KDD ``.data`` file,
which shares the bundled sample's headerless, comma-separated format.
"""
from __future__ import absolute_import, division, print_function

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import DenseFeat, SparseFeat, get_feature_names

from ..common import MultiTaskData

_HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "examples")

COLUMN_NAMES = [
    "age", "class_worker", "det_ind_code", "det_occ_code", "education", "wage_per_hour", "hs_college",
    "marital_stat", "major_ind_code", "major_occ_code", "race", "hisp_origin", "sex", "union_member",
    "unemp_reason", "full_or_part_emp", "capital_gains", "capital_losses", "stock_dividends",
    "tax_filer_stat", "region_prev_res", "state_prev_res", "det_hh_fam_stat", "det_hh_summ",
    "instance_weight", "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
    "num_emp", "fam_under_18", "country_father", "country_mother", "country_self", "citizenship",
    "own_or_self", "vet_question", "vet_benefits", "weeks_worked", "year", "income_50k",
]

SPARSE_FEATURES = [
    "class_worker", "det_ind_code", "det_occ_code", "education", "hs_college", "major_ind_code",
    "major_occ_code", "race", "hisp_origin", "sex", "union_member", "unemp_reason",
    "full_or_part_emp", "tax_filer_stat", "region_prev_res", "state_prev_res", "det_hh_fam_stat",
    "det_hh_summ", "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
    "fam_under_18", "country_father", "country_mother", "country_self", "citizenship",
    "vet_question",
]

TASK_NAMES = ("label_income", "label_marital")
TASK_TYPES = ("binary", "binary")


def load_census(data_path=None, embedding_dim=4, test_size=0.2, seed=2020):
    """Load and preprocess Census-Income into a :class:`MultiTaskData`.

    The UCI Census-Income (KDD) distribution ships an *official* train/test
    partition: ``census-income.data`` (train) and ``census-income.test`` (test).
    When ``data_path`` points at the ``.data`` file and its sibling ``.test``
    exists, that official split is used instead of a random one -- this is the
    canonical evaluation for the dataset and avoids reshuffling the two together.
    Encoders are still fit on the union so categories that appear only in the
    test partition don't crash the run. The bundled sample has no ``.test``
    sibling and falls back to a random split.
    """
    train_path = data_path or os.path.join(EXAMPLES_DIR, "census-income.sample")

    official_test = None
    if data_path and data_path.endswith(".data"):
        sibling = data_path[: -len(".data")] + ".test"
        if os.path.exists(sibling):
            official_test = sibling

    if official_test:
        train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
        test_df = pd.read_csv(official_test, header=None, names=COLUMN_NAMES)
        n_train = len(train_df)
        data = pd.concat([train_df, test_df], ignore_index=True)
        print("[census] using official train/test split (%d train + %d test rows)"
              % (n_train, len(test_df)))
    else:
        data = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
        n_train = None

    data["label_income"] = data["income_50k"].map({" - 50000.": 0, " 50000+.": 1})
    data["label_marital"] = data["marital_stat"].apply(lambda x: 1 if x == " Never married" else 0)
    data.drop(labels=["income_50k", "marital_stat"], axis=1, inplace=True)

    columns = data.columns.values.tolist()
    dense_features = [c for c in columns if c not in SPARSE_FEATURES and c not in TASK_NAMES]

    data[SPARSE_FEATURES] = data[SPARSE_FEATURES].fillna("-1")
    data[dense_features] = data[dense_features].fillna(0)
    data[dense_features] = MinMaxScaler((0, 1)).fit_transform(data[dense_features])
    for feat in SPARSE_FEATURES:
        data[feat] = LabelEncoder().fit_transform(data[feat])

    feature_columns = (
        [SparseFeat(feat, int(data[feat].max()) + 1, embedding_dim=embedding_dim) for feat in SPARSE_FEATURES]
        + [DenseFeat(feat, 1) for feat in dense_features]
    )
    feature_names = get_feature_names(feature_columns)

    if n_train is not None:
        train, test = data.iloc[:n_train], data.iloc[n_train:]
    else:
        train, test = train_test_split(data, test_size=test_size, random_state=seed)
    train_input = {n: train[n].values for n in feature_names}
    test_input = {n: test[n].values for n in feature_names}

    return MultiTaskData(
        name="census",
        dnn_feature_columns=feature_columns,
        feature_names=feature_names,
        train_input=train_input,
        train_y=[train[t].values.reshape(-1, 1) for t in TASK_NAMES],
        test_input=test_input,
        test_y=[test[t].values.reshape(-1, 1) for t in TASK_NAMES],
        task_types=TASK_TYPES,
        task_names=TASK_NAMES,
    )
