# -*- coding:utf-8 -*-
"""
OneTrans Layer: Unified Feature Interaction and Sequence Modeling.

Assembles feature field tokens and behavior sequence tokens into one token
sequence, then applies a single Transformer with a mixed attention mask so
that:
  - Feature tokens attend bidirectionally to all feature tokens and all
    valid sequence tokens.
  - Sequence token at position t attends to all feature tokens and to
    sequence tokens at positions 0..t (causal), excluding padding.

Reference:
    OneTrans: Unified Feature Interaction and Sequence Modeling with One
    Transformer in Industrial Recommender Systems.
"""

import tensorflow as tf

try:
    from tensorflow.python.ops.init_ops_v2 import TruncatedNormal, Zeros, glorot_uniform
except ImportError:
    from tensorflow.python.ops.init_ops import (
        TruncatedNormal, Zeros,
        glorot_uniform_initializer as glorot_uniform,
    )

from tensorflow.keras.layers import Layer, Dropout

from .normalization import LayerNormalization
from .utils import reduce_max, softmax


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding (sinusoidal, applied only to sequence tokens)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEncoding(Layer):
    """Sinusoidal positional encoding added to sequence tokens."""

    def __init__(self, **kwargs):
        super(SinusoidalPositionEncoding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # inputs: (B, T, D)
        T = tf.shape(inputs)[1]
        D = inputs.get_shape().as_list()[-1]

        pos = tf.cast(tf.range(T), tf.float32)           # (T,)
        dim = tf.cast(tf.range(D), tf.float32)           # (D,)
        angle = pos[:, tf.newaxis] / tf.pow(
            10000.0, (2 * (dim[tf.newaxis, :] // 2)) / tf.cast(D, tf.float32)
        )                                                 # (T, D)
        sin_enc = tf.math.sin(angle[:, 0::2])
        cos_enc = tf.math.cos(angle[:, 1::2])

        # Interleave sin/cos
        pe = tf.reshape(
            tf.stack([sin_enc, cos_enc], axis=2),
            [T, -1]
        )[:, :D]                                          # (T, D)

        return inputs + pe[tf.newaxis, :, :]             # (B, T, D)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(SinusoidalPositionEncoding, self).get_config()


class TokenSlice(Layer):
    """Slice the first ``num_tokens`` positions from a ``(B, N, D)`` sequence.

    A serializable replacement for ``Lambda(lambda t: t[:, :n, :])`` — Lambda
    layers that wrap a Python ``lambda`` cannot be safely round-tripped through
    ``save_model``/``load_model``.
    """

    def __init__(self, num_tokens, **kwargs):
        super(TokenSlice, self).__init__(**kwargs)
        self.num_tokens = num_tokens

    def call(self, inputs, **kwargs):
        return inputs[:, :self.num_tokens, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_tokens, input_shape[-1])

    def get_config(self):
        config = {"num_tokens": self.num_tokens}
        base = super(TokenSlice, self).get_config()
        base.update(config)
        return base


# ─────────────────────────────────────────────────────────────────────────────
# OneTrans single transformer block
# ─────────────────────────────────────────────────────────────────────────────

class OneTransLayer(Layer):
    """One Transformer block for OneTrans.

    Processes a combined token sequence of shape ``(B, F+T, D)`` where the
    first F tokens are feature field tokens and the remaining T tokens are
    behavior sequence tokens.

    A structured attention mask enforces:
      - Feature tokens can attend to all feature tokens and all *valid*
        sequence tokens.
      - Sequence token at step t can attend to all feature tokens and to
        sequence tokens at steps 0..t (causal mask), but not to padding.

    Input shape
        - A list of two tensors:
          - ``tokens``:     ``(batch_size, F+T, embedding_size)``
          - ``seq_length``: ``(batch_size, 1)`` — number of valid seq tokens

    Output shape
        - ``(batch_size, F+T, embedding_size)``

    Arguments
        - **num_feat_tokens**: int. Number of feature field tokens F.
        - **seq_len_max**: int. Maximum sequence length T.
        - **att_embedding_size**: int. Size per attention head.
        - **head_num**: int. Number of attention heads.
        - **use_ffn**: bool. Whether to add a point-wise FFN sub-layer.
        - **ffn_expand**: int. FFN hidden size = embedding_size * ffn_expand.
        - **dropout_rate**: float.
        - **use_layer_norm**: bool.
        - **seed**: int.
    """

    def __init__(
        self,
        num_feat_tokens,
        seq_len_max,
        att_embedding_size=8,
        head_num=4,
        use_ffn=True,
        ffn_expand=4,
        dropout_rate=0.0,
        use_layer_norm=True,
        seed=1024,
        **kwargs,
    ):
        super(OneTransLayer, self).__init__(**kwargs)
        self.F = num_feat_tokens
        self.T = seq_len_max
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_ffn = use_ffn
        self.ffn_expand = ffn_expand
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.seed = seed

        # Sub-layers are created in __init__ (not build) so Keras tracks them
        # consistently for full-model save/load round-trips.
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        if self.use_layer_norm:
            self.ln1 = LayerNormalization()
            self.ln2 = LayerNormalization()

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num (%d) must equal embedding_size (%d)"
                % (self.num_units, embedding_size)
            )

        self.W_Q = self.add_weight(
            name="W_Q", shape=[embedding_size, self.num_units],
            initializer=TruncatedNormal(seed=self.seed))
        self.W_K = self.add_weight(
            name="W_K", shape=[embedding_size, self.num_units],
            initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_V = self.add_weight(
            name="W_V", shape=[embedding_size, self.num_units],
            initializer=TruncatedNormal(seed=self.seed + 2))

        if self.use_ffn:
            hidden = embedding_size * self.ffn_expand
            self.fw1 = self.add_weight(
                name="fw1", shape=[self.num_units, hidden],
                initializer=glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight(
                name="fw2", shape=[hidden, self.num_units],
                initializer=glorot_uniform(seed=self.seed + 3))

        super(OneTransLayer, self).build(input_shape)

    # ── attention mask ───────────────────────────────────────────────────────

    def _build_mask(self, seq_length, N):
        """Build the structured attention mask.

        Returns float mask of shape ``(B, N, N)`` where 1.0 = can attend.

        Layout of the N = F+T token sequence:
          [feat_0, ..., feat_{F-1}, seq_0, ..., seq_{T-1}]
        """
        F, T = self.F, self.T

        # ── valid sequence positions: (B, T) ──
        # seq_length: (B, 1)
        seq_valid = tf.sequence_mask(
            tf.squeeze(seq_length, axis=1), T, dtype=tf.float32
        )  # (B, T)

        # ── feature-to-feature block: (B, F, F) all 1s ──
        batch_size = tf.shape(seq_length)[0]
        ff_block = tf.ones([batch_size, F, F])

        # ── feature-to-seq block: (B, F, T) — valid positions ──
        # Each feature token attends to all valid seq tokens
        fs_block = tf.tile(
            seq_valid[:, tf.newaxis, :], [1, F, 1]
        )  # (B, F, T)

        # ── seq-to-feature block: (B, T, F) all 1s ──
        sf_block = tf.ones([batch_size, T, F])

        # ── seq-to-seq block: (B, T, T) causal AND valid ──
        # Causal mask: lower-triangular (including diagonal)
        causal = tf.linalg.band_part(tf.ones([T, T]), -1, 0)  # (T, T)
        # Valid column mask: don't attend to padding positions
        col_valid = tf.tile(
            seq_valid[:, tf.newaxis, :], [1, T, 1]
        )  # (B, T, T)
        ss_block = causal[tf.newaxis, :, :] * col_valid        # (B, T, T)

        # ── assemble full mask: (B, N, N) ──
        top    = tf.concat([ff_block, fs_block], axis=2)  # (B, F, N)
        bottom = tf.concat([sf_block, ss_block], axis=2)  # (B, T, N)
        mask   = tf.concat([top, bottom], axis=1)         # (B, N, N)
        return mask

    # ── forward ──────────────────────────────────────────────────────────────

    def call(self, inputs, training=None, **kwargs):
        tokens, seq_length = inputs
        # tokens: (B, N, D),  seq_length: (B, 1)
        N = self.F + self.T

        # ── multi-head self-attention ──
        Q = tf.tensordot(tokens, self.W_Q, axes=(-1, 0))  # (B, N, D)
        K = tf.tensordot(tokens, self.W_K, axes=(-1, 0))
        V = tf.tensordot(tokens, self.W_V, axes=(-1, 0))

        # Split heads: (h*B, N, d)
        Q_ = tf.concat(tf.split(Q, self.head_num, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.head_num, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.head_num, axis=2), axis=0)

        # Scaled dot-product: (h*B, N, N)
        scale = self.att_embedding_size ** 0.5
        scores = tf.matmul(Q_, K_, transpose_b=True) / scale

        # Structured mask: (B, N, N) → tile for heads: (h*B, N, N)
        mask = self._build_mask(seq_length, N)
        mask = tf.tile(mask, [self.head_num, 1, 1])

        NEG_INF = -2.0 ** 32 + 1.0
        paddings = tf.ones_like(scores) * NEG_INF
        scores = tf.where(tf.equal(mask, 1.0), scores, paddings)

        # Numerical stability + softmax
        scores -= reduce_max(scores, axis=-1, keep_dims=True)
        weights = softmax(scores)                           # (h*B, N, N)
        weights = self.dropout(weights, training=training)

        # Weighted sum: (h*B, N, d) → merge heads → (B, N, D)
        context = tf.matmul(weights, V_)
        context = tf.concat(tf.split(context, self.head_num, axis=0), axis=2)

        # Residual + LayerNorm
        out = tokens + context
        if self.use_layer_norm:
            out = self.ln1(out)

        # ── point-wise FFN ──
        if self.use_ffn:
            fw = tf.nn.relu(tf.tensordot(out, self.fw1, axes=(-1, 0)))
            fw = self.dropout(fw, training=training)
            fw = tf.tensordot(fw, self.fw2, axes=(-1, 0))
            out = out + fw
            if self.use_layer_norm:
                out = self.ln2(out)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            "num_feat_tokens": self.F,
            "seq_len_max":     self.T,
            "att_embedding_size": self.att_embedding_size,
            "head_num":        self.head_num,
            "use_ffn":         self.use_ffn,
            "ffn_expand":      self.ffn_expand,
            "dropout_rate":    self.dropout_rate,
            "use_layer_norm":  self.use_layer_norm,
            "seed":            self.seed,
        }
        base = super(OneTransLayer, self).get_config()
        base.update(config)
        return base
