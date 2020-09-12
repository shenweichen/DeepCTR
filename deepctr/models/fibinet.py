# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
"""

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import SENETLayer, BilinearInteraction
from ..layers.utils import concat_func, add_func, combined_dnn_input


def FiBiNET(linear_feature_columns, dnn_feature_columns, bilinear_type='interaction', reduction_ratio=3,
            dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
            l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
            task='binary'):
    """Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bilinear_type: str,bilinear function type used in Bilinear Interaction Layer,can be ``'all'`` , ``'each'`` or ``'interaction'``
    :param reduction_ratio: integer in [1,inf), reduction ratio used in SENET Layer
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    senet_embedding_list = SENETLayer(
        reduction_ratio, seed)(sparse_embedding_list)

    senet_bilinear_out = BilinearInteraction(
        bilinear_type=bilinear_type, seed=seed)(senet_embedding_list)
    bilinear_out = BilinearInteraction(
        bilinear_type=bilinear_type, seed=seed)(sparse_embedding_list)

    dnn_input = combined_dnn_input(
        [tf.keras.layers.Flatten()(concat_func([senet_bilinear_out, bilinear_out]))], dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

    final_logit = add_func([linear_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
