"""Equation and gradient contracts for FinalMLP's reusable core layer."""
from __future__ import absolute_import, division, print_function

import numpy as np

from deepctr.layers.finalmlp import InteractionAggregation
from tests.correctness import (assert_finite_gradients, assert_forward_matches,
                               assert_numerical_gradient)


def _reference(inputs, weights):
    y1, y2 = inputs
    linear = weights["bias"] + np.matmul(y1, weights["w1"]) + np.matmul(y2, weights["w2"])
    heads = weights["W_bilinear"].shape[0]
    y1_heads = y1.reshape((-1, heads, y1.shape[-1] // heads))
    y2_heads = y2.reshape((-1, heads, y2.shape[-1] // heads))
    bilinear = np.einsum("bhi,hiok,bhk->bo", y1_heads, weights["W_bilinear"], y2_heads)
    return linear + bilinear


def test_reference_and_gradients_for_bilinear_interaction():
    y1 = np.asarray([[0.2, -0.3, 0.5, 0.7],
                     [0.1, 0.4, -0.2, 0.8]], dtype="float32")
    y2 = np.asarray([[0.6, -0.1, 0.9, 0.3],
                     [-0.5, 0.2, 0.4, 0.7]], dtype="float32")
    layer = InteractionAggregation(num_heads=2, output_dim=2)
    weights = {
        "w1": np.arange(8, dtype="float32").reshape(4, 2) / 10.0,
        "w2": np.arange(8, 16, dtype="float32").reshape(4, 2) / 20.0,
        "bias": np.asarray([0.1, -0.2], dtype="float32"),
        "W_bilinear": np.arange(16, dtype="float32").reshape(2, 2, 2, 2) / 25.0,
    }

    assert_forward_matches(layer, [y1, y2], _reference, weights=weights)
    assert_finite_gradients(layer, [y1, y2])
    assert_numerical_gradient(layer, [y1[:1], y2[:1]], rtol=3e-2, atol=3e-3)
