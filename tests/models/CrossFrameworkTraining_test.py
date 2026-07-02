# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from deepctr.models import DeepFM
from deepctr.initializers import export_deepfm_parameters, load_deepfm_parameters
from ..utils import get_test_data


def test_default_initialization_is_semantically_reproducible():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=2, sequence_feature=[]
    )
    tf.keras.backend.clear_session()
    first = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026
    )
    tf.keras.backend.clear_session()
    second = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026
    )
    for first_weight, second_weight in zip(first.get_weights(), second.get_weights()):
        np.testing.assert_array_equal(first_weight, second_weight)


def test_portable_parameters_round_trip_between_models():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=2, sequence_feature=[]
    )
    tf.keras.backend.clear_session()
    source = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026
    )
    target = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2027
    )
    load_deepfm_parameters(target, export_deepfm_parameters(source))
    for source_weight, target_weight in zip(source.get_weights(), target.get_weights()):
        np.testing.assert_array_equal(source_weight, target_weight)
