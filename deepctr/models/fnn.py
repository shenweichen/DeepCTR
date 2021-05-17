# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Zhang W, Du T, Wang J. Deep learning over multi-field categorical data[C]//European conference on information retrieval. Springer, Cham, 2016: 45-57.(https://arxiv.org/pdf/1601.02376.pdf)
"""
import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import add_func, combined_dnn_input


def FNN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
        dnn_activation='relu', task='binary'):
    """Instantiates the Factorization-supported Neural Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear weight
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(deep_out)
    final_logit = add_func([dnn_logit, linear_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=output)
    return model
