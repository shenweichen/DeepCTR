# -*- coding:utf-8 -*-
"""
Author:
    Harshit Pande

Reference:
    [1] Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising
    (https://arxiv.org/pdf/1806.03514.pdf)

"""

from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FwFMLayer
from ..layers.utils import concat_func, add_func, combined_dnn_input


def FwFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
         l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_field_strength=0.00001, l2_reg_dnn=0,
         seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the FwFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list if do not want DNN, the layer number and units
    in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_field_strength: float. L2 regularizer strength applied to the field pair strength parameters
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                        l2_reg_embedding, seed,
                                                                        support_group=True)

    fwfm_logit = add_func([FwFMLayer(num_fields=len(v), regularizer=l2_reg_field_strength)
                           (concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])

    final_logit_components = [linear_logit, fwfm_logit]

    if dnn_hidden_units:
        dnn_input = combined_dnn_input(list(chain.from_iterable(
            group_embedding_dict.values())), dense_value_list)
        dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                         dnn_use_bn, seed)(dnn_input)
        dnn_logit = tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)
        final_logit_components.append(dnn_logit)

    final_logit = add_func(final_logit_components)

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
