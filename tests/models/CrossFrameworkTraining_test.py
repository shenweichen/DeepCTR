# -*- coding: utf-8 -*-
import numpy as np
import pytest
import tensorflow as tf

from deepctr.models import DeepFM
from deepctr.initializers import export_deepfm_parameters, load_deepfm_parameters
from deepctr.training import (
    NumpyBatchSequence,
    TensorFlowAdam,
    compile_cross_framework,
    fit_cross_framework,
)
from ..utils import get_test_data


def test_numpy_batch_sequence_matches_seeded_epoch_permutations():
    values = np.arange(10)
    sequence = NumpyBatchSequence(
        {"value": values}, values, batch_size=3, seed=2026, shuffle=True
    )
    random = np.random.default_rng(2026)
    first_expected = random.permutation(10)
    first_actual = np.concatenate([
        sequence[index][1] for index in range(len(sequence))
    ]).reshape(-1)
    np.testing.assert_array_equal(first_actual, first_expected)
    sequence.on_epoch_end()
    second_expected = random.permutation(10)
    second_actual = np.concatenate([
        sequence[index][1] for index in range(len(sequence))
    ]).reshape(-1)
    np.testing.assert_array_equal(second_actual, second_expected)
    assert sequence[0][1].shape[1] == 1


def test_tensorflow_adam_aggregates_duplicate_sparse_indices():
    variable = tf.Variable(np.zeros((3, 1), dtype="float32"))
    gradient = tf.IndexedSlices(
        values=tf.constant([[1.0], [2.0]], dtype=tf.float32),
        indices=tf.constant([0, 0], dtype=tf.int32),
        dense_shape=tf.constant([3, 1], dtype=tf.int32),
    )
    optimizer = TensorFlowAdam()
    optimizer.apply_gradients([variable], [gradient])
    assert variable.numpy()[0, 0] < 0
    np.testing.assert_array_equal(variable.numpy()[1:], np.zeros((2, 1)))


def _trained_predictions(x, y, feature_columns):
    tf.keras.backend.clear_session()
    model = DeepFM(
        feature_columns,
        feature_columns,
        dnn_hidden_units=(8,),
        dnn_dropout=0.0,
        l2_reg_linear=0.0,
        l2_reg_embedding=0.0,
        l2_reg_dnn=0.0,
        seed=2026,
    )
    compile_cross_framework(model)
    fit_cross_framework(
        model, x, y, batch_size=16, epochs=2, seed=2026, verbose=0
    )
    return model.predict(x, batch_size=32, verbose=0)


def test_cross_framework_training_is_reproducible():
    x, y, feature_columns = get_test_data(
        96, sparse_feature_num=2, dense_feature_num=2, sequence_feature=[]
    )
    first = _trained_predictions(x, y, feature_columns)
    second = _trained_predictions(x, y, feature_columns)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_portable_initialization_is_semantically_reproducible():
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


def test_deepfm_defaults_to_portable_initialization():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1, sequence_feature=[]
    )
    model = DeepFM(feature_columns, feature_columns)
    assert model.initialization_profile == "cross_framework"


def test_native_initialization_remains_available():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1, sequence_feature=[]
    )
    model = DeepFM(
        feature_columns, feature_columns, initialization_profile="native"
    )
    assert model.initialization_profile == "native"


def test_portable_parameters_round_trip_between_models():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=2, sequence_feature=[]
    )
    tf.keras.backend.clear_session()
    source = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026,
        initialization_profile="cross_framework"
    )
    target = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2027
    )
    load_deepfm_parameters(target, export_deepfm_parameters(source))
    for source_weight, target_weight in zip(source.get_weights(), target.get_weights()):
        np.testing.assert_array_equal(source_weight, target_weight)


def test_unknown_initialization_profile_is_rejected():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1, sequence_feature=[]
    )
    with pytest.raises(ValueError, match="initialization_profile"):
        DeepFM(
            feature_columns, feature_columns,
            initialization_profile="unknown"
        )
