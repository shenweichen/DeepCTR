# -*- coding:utf-8 -*-
"""Portable, semantic-name-based model initialization."""

import hashlib
import math

import numpy as np


def _semantic_seed(seed, semantic):
    payload = "{}:{}".format(seed, semantic).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _normal(semantic, shape, seed, stddev):
    random = np.random.Generator(np.random.PCG64(_semantic_seed(seed, semantic)))
    return random.normal(0.0, stddev, size=shape).astype("float32")


def _glorot_normal(semantic, shape, seed):
    fan_in, fan_out = shape[0], shape[1]
    return _normal(
        semantic,
        shape,
        seed,
        math.sqrt(2.0 / float(fan_in + fan_out)),
    )


def _deepfm_semantic(name, shape):
    name = name.split(":", 1)[0]
    layer, variable = name.split("/", 1)
    if layer.startswith("linear0sparse_seq_emb_") and variable == "embeddings":
        return "linear_embedding:" + layer[len("linear0sparse_seq_emb_"):], "embedding"
    if layer.startswith("linear0sparse_emb_") and variable == "embeddings":
        return "linear_embedding:" + layer[len("linear0sparse_emb_"):], "embedding"
    if layer.startswith("sparse_seq_emb_") and variable == "embeddings":
        return "embedding:" + layer[len("sparse_seq_emb_"):], "embedding"
    if layer.startswith("sparse_emb_") and variable == "embeddings":
        return "embedding:" + layer[len("sparse_emb_"):], "embedding"
    if layer.startswith("linear") and variable == "linear_kernel":
        return "linear_dense", "embedding"
    if layer.startswith("dnn") and variable.startswith("kernel"):
        return "dnn_kernel:" + variable[len("kernel"):], "glorot"
    if layer.startswith("dnn") and variable.startswith("bias"):
        return "dnn_bias:" + variable[len("bias"):], "zero"
    if layer.startswith("dense") and variable == "kernel":
        return "dnn_output", "glorot"
    if layer.startswith("prediction_layer") and variable == "global_bias":
        return "output_bias", "zero"
    raise ValueError("unsupported DeepFM parameter for portable initialization: {}".format(name))


def initialize_deepfm_parameters(model, seed=1024, init_std=0.0001):
    """Initialize DeepFM parameters identically across supported frameworks."""
    for weight in model.trainable_weights:
        shape = tuple(int(value) for value in weight.shape)
        semantic, distribution = _deepfm_semantic(weight.name, shape)
        if distribution == "embedding":
            value = _normal(semantic, shape, seed, init_std)
        elif distribution == "glorot":
            value = _glorot_normal(semantic, shape, seed)
        else:
            value = np.zeros(shape, dtype="float32")
        weight.assign(value)


def export_deepfm_parameters(model):
    """Return canonical NumPy DeepFM parameters for framework transfer."""
    values = {}
    for weight in model.trainable_weights:
        semantic, _ = _deepfm_semantic(
            weight.name, tuple(int(value) for value in weight.shape)
        )
        values[semantic] = weight.numpy().copy()
    return values


def load_deepfm_parameters(model, parameters):
    """Load canonical NumPy DeepFM parameters from another framework."""
    expected = {
        _deepfm_semantic(
            weight.name, tuple(int(value) for value in weight.shape)
        )[0]
        for weight in model.trainable_weights
    }
    if set(parameters) != expected:
        raise ValueError(
            "portable parameter coverage mismatch; missing={} unknown={}".format(
                sorted(expected - set(parameters)),
                sorted(set(parameters) - expected),
            )
        )
    for weight in model.trainable_weights:
        semantic, _ = _deepfm_semantic(
            weight.name, tuple(int(value) for value in weight.shape)
        )
        value = np.asarray(parameters[semantic])
        if tuple(value.shape) != tuple(int(item) for item in weight.shape):
            raise ValueError("shape mismatch for {}".format(semantic))
        weight.assign(value)
    return model
