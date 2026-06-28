"""Equation and gradient contracts for WuKong's composite interaction block."""
from __future__ import absolute_import, division, print_function

import numpy as np

from deepctr.layers.wukong import WuKongLayer
from tests.correctness import assert_finite_gradients, assert_forward_matches


def _reference(inputs, weights):
    x = inputs
    batch_size, _token_count, embedding_dim = x.shape
    interaction = np.matmul(x, np.swapaxes(x, 1, 2)).reshape(batch_size, -1)
    hidden = np.maximum(np.matmul(interaction, weights["fmb_w1"]) + weights["fmb_b1"], 0.0)
    fmb = np.matmul(hidden, weights["fmb_w2"]) + weights["fmb_b2"]
    fmb = fmb.reshape(batch_size, 1, embedding_dim)

    mean = np.mean(fmb, axis=-1, keepdims=True)
    variance = np.mean(np.square(fmb - mean), axis=-1, keepdims=True)
    fmb = (fmb - mean) / np.sqrt(variance + 1e-9)
    fmb = fmb * weights["gamma"] + weights["beta"]

    lcb = np.einsum("lk,bkd->bld", weights["lcb_w"], x)
    return np.concatenate([fmb, lcb], axis=1) + x


def test_reference_and_gradients_for_composite_interaction_block():
    x = np.asarray([[[0.2, 0.7], [0.5, -0.3], [0.8, 0.1]],
                    [[-0.4, 0.6], [0.9, 0.2], [0.3, -0.5]]], dtype="float32")
    layer = WuKongLayer(num_fmb_tokens=1, fmb_hidden_dim=2, dropout_rate=0.0)
    weights = {
        "fmb_w1": np.arange(18, dtype="float32").reshape(9, 2) / 30.0 - 0.2,
        "fmb_b1": np.asarray([0.2, 0.1], dtype="float32"),
        "fmb_w2": np.asarray([[0.4, -0.2], [0.1, 0.5]], dtype="float32"),
        "fmb_b2": np.asarray([0.05, -0.1], dtype="float32"),
        "lcb_w": np.asarray([[0.3, -0.2, 0.5], [-0.4, 0.6, 0.2]], dtype="float32"),
        "gamma": np.asarray([1.2, 0.8], dtype="float32"),
        "beta": np.asarray([0.1, -0.1], dtype="float32"),
    }

    assert_forward_matches(layer, x, _reference, weights=weights,
                           rtol=2e-5, atol=2e-5)
    assert_finite_gradients(layer, x)
