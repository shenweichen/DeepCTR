# -*- coding:utf-8 -*-
"""
FinalMLP

Reference:
    [1] Mao K, Zhu J, Su L, et al. FinalMLP: An Enhanced Two-Stream MLP Model for
    CTR Prediction. AAAI 2023. (https://arxiv.org/abs/2304.00902)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Multiply

from ..feature_column import build_input_features, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.finalmlp import InteractionAggregation
from ..layers.utils import combined_dnn_input


def _feature_selection(emb, gate_units, seed):
    """Per-stream feature gating: a context MLP produces a sigmoid gate that is
    multiplied element-wise into the shared embedding, differentiating the two
    streams' inputs."""
    feat_dim = int(emb.shape[-1])
    gate = DNN(gate_units, "relu", seed=seed)(emb) if gate_units else emb
    gate = Dense(feat_dim, activation="sigmoid")(gate)
    return Multiply()([emb, gate])


def FinalMLP(linear_feature_columns, dnn_feature_columns,
             mlp1_hidden_units=(128, 64), mlp2_hidden_units=(128, 64),
             fs1_gate_units=(64,), fs2_gate_units=(64,), num_heads=1,
             l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
             dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the FinalMLP architecture (two feature-gated MLP streams fused
    by a multi-head bilinear interaction aggregation head).

    :param linear_feature_columns: accepted for a uniform builder signature; the
        input set is built from ``linear + dnn`` columns (FinalMLP itself is
        MLP-only, with no separate linear/FM logit).
    :param dnn_feature_columns: features for both MLP streams.
    :param mlp1_hidden_units / mlp2_hidden_units: layer sizes of the two streams.
    :param fs1_gate_units / fs2_gate_units: hidden sizes of each stream's feature-selection gate MLP.
    :param num_heads: number of heads in the interaction aggregation (must divide both stream output dims).
    :param task: ``"binary"`` or ``"regression"``.
    :return: A Keras model instance.
    """
    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding, seed)
    emb = combined_dnn_input(sparse_embedding_list, dense_value_list)  # (B, D)

    stream1_in = _feature_selection(emb, fs1_gate_units, seed)
    stream2_in = _feature_selection(emb, fs2_gate_units, seed + 1)

    stream1 = DNN(mlp1_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                  seed=seed)(stream1_in)
    stream2 = DNN(mlp2_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                  seed=seed + 1)(stream2_in)

    final_logit = InteractionAggregation(num_heads=num_heads, output_dim=1, seed=seed)(
        [stream1, stream2])
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
