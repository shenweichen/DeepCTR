# -*- coding:utf-8 -*-
"""Custom layer for the FinalMLP model: multi-head bilinear interaction
aggregation that fuses the two MLP streams into the prediction logit.

Reference:
    Mao K, et al. FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction.
    AAAI 2023. (https://arxiv.org/abs/2304.00902)
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class InteractionAggregation(Layer):
    """Multi-head bilinear interaction aggregation.

    Given the two stream outputs ``y1`` (B, d1) and ``y2`` (B, d2), produces an
    ``output_dim`` logit::

        out = b + y1 W1 + y2 W2 + sum_h (y1_h^T W_h y2_h)

    where each stream is split into ``num_heads`` heads for the bilinear term.

    Input shape
        - list of two 2D tensors ``[(B, d1), (B, d2)]``.
    Output shape
        - ``(B, output_dim)``.
    """

    def __init__(self, num_heads=1, output_dim=1, seed=1024, **kwargs):
        super(InteractionAggregation, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.seed = seed

    def build(self, input_shape):
        if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 2):
            raise ValueError("InteractionAggregation expects a list of two inputs")
        self.d1 = int(input_shape[0][-1])
        self.d2 = int(input_shape[1][-1])
        if self.d1 % self.num_heads or self.d2 % self.num_heads:
            raise ValueError("stream dims (%d, %d) must be divisible by num_heads (%d)"
                             % (self.d1, self.d2, self.num_heads))
        self.h1 = self.d1 // self.num_heads
        self.h2 = self.d2 // self.num_heads

        self.w1 = self.add_weight(name="w1", shape=(self.d1, self.output_dim),
                                  initializer="glorot_uniform")
        self.w2 = self.add_weight(name="w2", shape=(self.d2, self.output_dim),
                                  initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(self.output_dim,),
                                    initializer="zeros")
        self.W = self.add_weight(
            name="W_bilinear",
            shape=(self.num_heads, self.h1, self.output_dim, self.h2),
            initializer="glorot_uniform")
        super(InteractionAggregation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        y1, y2 = inputs
        out = self.bias + tf.matmul(y1, self.w1) + tf.matmul(y2, self.w2)  # (B, o)
        y1_h = tf.reshape(y1, [-1, self.num_heads, self.h1])               # (B, H, h1)
        y2_h = tf.reshape(y2, [-1, self.num_heads, self.h2])               # (B, H, h2)
        bilinear = tf.einsum("bhi,hiok,bhk->bo", y1_h, self.W, y2_h)       # (B, o)
        return out + bilinear

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)

    def get_config(self):
        config = {"num_heads": self.num_heads, "output_dim": self.output_dim, "seed": self.seed}
        base = super(InteractionAggregation, self).get_config()
        base.update(config)
        return base
