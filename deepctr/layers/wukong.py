# -*- coding:utf-8 -*-
"""Custom layer for the WuKong model: one stackable interaction layer made of a
Factorization-Machine Block (FMB) and a Linear Compression Block (LCB), joined
by a residual connection.

Reference:
    Zhang B, et al. Wukong: Towards a Scaling Law for Large-Scale Recommendation.
    ICML 2024. (https://arxiv.org/abs/2403.02545)
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout

from .normalization import LayerNormalization


class WuKongLayer(Layer):
    """One WuKong interaction layer over a token matrix ``X`` of shape (B, k, d).

    * FMB: explicit pairwise interactions ``X X^T`` (B, k, k) → MLP → reshaped to
      ``(B, n_fmb, d)`` and layer-normed.
    * LCB: a learned linear recombination of the input embeddings → ``(B, n_lcb, d)``.
    * Output: ``concat([FMB, LCB], axis=1) + X`` (residual), shape ``(B, k, d)``,
      so layers stack uniformly.

    Input / output shape: ``(batch_size, k, d)``.
    """

    def __init__(self, num_fmb_tokens=None, fmb_hidden_dim=128, dropout_rate=0.0,
                 seed=1024, **kwargs):
        super(WuKongLayer, self).__init__(**kwargs)
        self.num_fmb_tokens = num_fmb_tokens
        self.fmb_hidden_dim = fmb_hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.dropout = Dropout(dropout_rate, seed=seed)
        self.ln = LayerNormalization()

    def build(self, input_shape):
        self.k = int(input_shape[1])
        self.d = int(input_shape[2])
        # split output tokens so n_fmb + n_lcb == k (keeps the layer stackable)
        self.n_fmb = self.num_fmb_tokens if self.num_fmb_tokens else max(1, self.k // 2)
        self.n_fmb = min(self.n_fmb, self.k)
        self.n_lcb = self.k - self.n_fmb

        # FMB: flatten(X X^T) (k*k) -> hidden -> n_fmb*d
        self.fmb_w1 = self.add_weight(name="fmb_w1", shape=(self.k * self.k, self.fmb_hidden_dim),
                                      initializer="glorot_uniform")
        self.fmb_b1 = self.add_weight(name="fmb_b1", shape=(self.fmb_hidden_dim,),
                                      initializer="zeros")
        self.fmb_w2 = self.add_weight(name="fmb_w2", shape=(self.fmb_hidden_dim, self.n_fmb * self.d),
                                      initializer="glorot_uniform")
        self.fmb_b2 = self.add_weight(name="fmb_b2", shape=(self.n_fmb * self.d,),
                                      initializer="zeros")
        # LCB: (n_lcb, k) linear recombination of input embeddings
        if self.n_lcb > 0:
            self.lcb_w = self.add_weight(name="lcb_w", shape=(self.n_lcb, self.k),
                                         initializer="glorot_uniform")
        super(WuKongLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x = inputs  # (B, k, d)
        # ── FMB ──
        inter = tf.matmul(x, x, transpose_b=True)            # (B, k, k)
        flat = tf.reshape(inter, [-1, self.k * self.k])      # (B, k*k)
        h = tf.nn.relu(tf.matmul(flat, self.fmb_w1) + self.fmb_b1)
        h = self.dropout(h, training=training)
        h = tf.matmul(h, self.fmb_w2) + self.fmb_b2          # (B, n_fmb*d)
        fmb = tf.reshape(h, [-1, self.n_fmb, self.d])        # (B, n_fmb, d)
        fmb = self.ln(fmb)

        # ── LCB ──
        if self.n_lcb > 0:
            lcb = tf.einsum("lk,bkd->bld", self.lcb_w, x)    # (B, n_lcb, d)
            out = tf.concat([fmb, lcb], axis=1)              # (B, k, d)
        else:
            out = fmb

        # ── residual ──
        return out + x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"num_fmb_tokens": self.num_fmb_tokens, "fmb_hidden_dim": self.fmb_hidden_dim,
                  "dropout_rate": self.dropout_rate, "seed": self.seed}
        base = super(WuKongLayer, self).get_config()
        base.update(config)
        return base
