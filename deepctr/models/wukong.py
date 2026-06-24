# -*- coding:utf-8 -*-
"""
WuKong

Reference:
    [1] Zhang B, Luo L, Chen Y, et al. Wukong: Towards a Scaling Law for
    Large-Scale Recommendation. ICML 2024. (https://arxiv.org/abs/2403.02545)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import concat_func, combined_dnn_input, add_func
from ..layers.wukong import WuKongLayer


def WuKong(linear_feature_columns, dnn_feature_columns, num_layers=3, fmb_hidden_dim=128,
           num_fmb_tokens=None, dnn_hidden_units=(128, 64), l2_reg_linear=0.00001,
           l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the WuKong architecture.

    Stacks ``num_layers`` WuKong interaction layers (Factorization-Machine Block +
    Linear Compression Block, with residuals) over the stacked field embeddings,
    then a DNN head. All sparse / var-len features must share one ``embedding_dim``
    (the field embeddings are stacked into a (B, num_fields, d) token matrix).

    :param linear_feature_columns: features for the linear part.
    :param dnn_feature_columns: features for the deep/interaction part.
    :param num_layers: number of stacked WuKong layers.
    :param fmb_hidden_dim: hidden size of each FMB's MLP.
    :param num_fmb_tokens: FMB output tokens per layer (default: half the fields).
    :param dnn_hidden_units: layer sizes of the DNN head.
    :param task: ``"binary"`` or ``"regression"``.
    :return: A Keras model instance.
    """
    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding, seed)
    if not sparse_embedding_list:
        raise ValueError("WuKong requires at least one embedding (sparse/var-len) feature.")

    # stack field embeddings into a (B, num_fields, d) token matrix
    x = concat_func(sparse_embedding_list, axis=1)
    for _ in range(num_layers):
        x = WuKongLayer(num_fmb_tokens=num_fmb_tokens, fmb_hidden_dim=fmb_hidden_dim,
                        dropout_rate=dnn_dropout, seed=seed)(x)

    interaction_out = Flatten()(x)
    dnn_input = combined_dnn_input([interaction_out], dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                  seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=False)(dnn_out)

    final_logit = add_func([linear_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
