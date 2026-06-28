"""Exact mask and causal-invariance contracts for OneTrans."""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from deepctr.layers.onetrans import OneTransLayer
from tests.correctness import assert_invariant, assign_named_weights


def _expected_mask():
    return np.asarray([
        # seq_length = 2; layout is [feature_0, feature_1, seq_0, seq_1, seq_2]
        [[1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 0]],
        # seq_length = 1
        [[1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0]],
    ], dtype="float32")


def test_attention_mask_matches_structured_contract():
    layer = OneTransLayer(num_feat_tokens=2, seq_len_max=3,
                          att_embedding_size=2, head_num=1,
                          use_ffn=False, use_layer_norm=False)
    lengths = tf.constant([[2], [1]], dtype=tf.int32)
    actual = layer._build_mask(lengths, 5)
    assert_allclose(actual.numpy(), _expected_mask(), rtol=0, atol=0)


def test_causal_output_is_invariant_to_future_token_changes():
    tokens = np.asarray([[[0.1, 0.2], [0.3, -0.1],
                          [0.5, 0.4], [0.7, -0.2], [0.9, 0.6]]], dtype="float32")
    lengths = np.asarray([[3]], dtype="int32")
    layer = OneTransLayer(num_feat_tokens=2, seq_len_max=3,
                          att_embedding_size=2, head_num=1,
                          use_ffn=False, use_layer_norm=False, dropout_rate=0.0)
    layer([tf.constant(tokens), tf.constant(lengths)], training=False)
    identity = np.eye(2, dtype="float32")
    assign_named_weights(layer, {"W_Q": identity, "W_K": identity, "W_V": identity})

    def change_future(values):
        values[0][0, 4, :] = np.asarray([100.0, -100.0], dtype="float32")
        return values

    # seq_0 is token index 2. Its causal receptive field excludes seq_2.
    assert_invariant(layer, [tokens, lengths], change_future,
                     output_selector=lambda output: output[:, 2, :])
