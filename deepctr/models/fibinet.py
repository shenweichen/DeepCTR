# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
"""

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, add, Flatten

from ..inputs import build_input_features, get_linear_logit, input_from_feature_columns, combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import SENETLayer, BilinearInteraction
from ..layers.utils import concat_fun


def FiBiNET(linear_feature_columns, dnn_feature_columns, embedding_size=8, bilinear_type='interaction', reduction_ratio=3, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
            l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
            task='binary'):
    """Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param bilinear_type: str,bilinear function type used in Bilinear Interaction Layer,can be ``'all'`` , ``'each'`` or ``'interaction'``
    :param reduction_ratio: integer in [1,inf), reduction ratio used in SENET Layer
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         embedding_size,
                                                                         l2_reg_embedding, init_std,
                                                                         seed)

    senet_embedding_list = SENETLayer(
        reduction_ratio, seed)(sparse_embedding_list)

    senet_bilinear_out = BilinearInteraction(
        bilinear_type=bilinear_type, seed=seed)(senet_embedding_list)
    bilinear_out = BilinearInteraction(
        bilinear_type=bilinear_type, seed=seed)(sparse_embedding_list)

    linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
                                    seed=seed, prefix='linear')

    dnn_input = combined_dnn_input(
        [Flatten()(concat_fun([senet_bilinear_out, bilinear_out]))], dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed)(dnn_input)
    dnn_logit = Dense(
        1, use_bias=False, activation=None)(dnn_out)

    if len(linear_feature_columns) > 0 and len(dnn_feature_columns) > 0:  # linear + dnn
        final_logit = add([linear_logit, dnn_logit])
    elif len(linear_feature_columns) == 0:
        final_logit = dnn_logit
    elif len(dnn_feature_columns) == 0:
        final_logit = linear_logit
    else:
        raise NotImplementedError

    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
