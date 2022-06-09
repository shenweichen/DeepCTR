# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.(https://arxiv.org/pdf/1611.00144.pdf)
"""

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Reshape, Flatten

from ..feature_column import build_input_features, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import InnerProductLayer, OutterProductLayer
from ..layers.utils import concat_func, combined_dnn_input


def PNN(dnn_feature_columns, dnn_hidden_units=(256, 128, 64), l2_reg_embedding=0.00001, l2_reg_dnn=0,
        seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False, kernel_type='mat',
        task='binary'):
    """Instantiates the Product-based Neural Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float . L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param use_inner: bool,whether use inner-product or not.
    :param use_outter: bool,whether use outter-product or not.
    :param kernel_type: str,kernel_type used in outter-product,can be ``'mat'`` , ``'vec'`` or ``'num'``
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if kernel_type not in ['mat', 'vec', 'num']:
        raise ValueError("kernel_type must be mat,vec or num")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    inner_product = Flatten()(
        InnerProductLayer()(sparse_embedding_list))
    outter_product = OutterProductLayer(kernel_type)(sparse_embedding_list)

    # ipnn deep input
    linear_signal = Reshape(
        [sum(map(lambda x: int(x.shape[-1]), sparse_embedding_list))])(concat_func(sparse_embedding_list))

    if use_inner and use_outter:
        deep_input = concat_func([linear_signal, inner_product, outter_product])
    elif use_inner:
        deep_input = concat_func([linear_signal, inner_product])
    elif use_outter:
        deep_input = concat_func([linear_signal, outter_product])
    else:
        deep_input = linear_signal

    dnn_input = combined_dnn_input([deep_input], dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    dnn_logit = Dense(1, use_bias=False)(dnn_out)

    output = PredictionLayer(task)(dnn_logit)

    model = Model(inputs=inputs_list,
                  outputs=output)
    return model
