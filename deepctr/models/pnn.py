# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.(https://arxiv.org/pdf/1611.00144.pdf)
"""

import tensorflow as tf
from ..layers.core import PredictionLayer, MLP
from ..layers.interaction import InnerProductLayer, OutterProductLayer
from ..input_embedding import preprocess_input_embedding
from ..utils import check_feature_config_dict
from ..layers.utils import concat_fun


def PNN(feature_dim_dict, embedding_size=8, hidden_size=(128, 128), l2_reg_embedding=1e-5, l2_reg_deep=0,
        init_std=0.0001, seed=1024, keep_prob=1, activation='relu',
        final_activation='sigmoid', use_inner=True, use_outter=False, kernel_type='mat', ):
    """Instantiates the Product-based Neural Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float . L2 regularizer strength applied to embedding vector
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param activation: Activation function to use in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param use_inner: bool,whether use inner-product or not.
    :param use_outter: bool,whether use outter-product or not.
    :param kernel_type: str,kernel_type used in outter-product,can be ``'mat'`` , ``'vec'`` or ``'num'``
    :return: A Keras model instance.
    """
    check_feature_config_dict(feature_dim_dict)

    if kernel_type not in ['mat', 'vec', 'num']:
        raise ValueError("kernel_type must be mat,vec or num")

    deep_emb_list, _, inputs_list = preprocess_input_embedding(feature_dim_dict, embedding_size,
                                                               l2_reg_embedding, 0, init_std,
                                                               seed, True)

    inner_product = tf.keras.layers.Flatten()(InnerProductLayer()(deep_emb_list))
    outter_product = OutterProductLayer(kernel_type)(deep_emb_list)

    # ipnn deep input
    linear_signal = tf.keras.layers.Reshape(
        [len(deep_emb_list)*embedding_size])(concat_fun(deep_emb_list))

    if use_inner and use_outter:
        deep_input = tf.keras.layers.Concatenate()(
            [linear_signal, inner_product, outter_product])
    elif use_inner:
        deep_input = tf.keras.layers.Concatenate()(
            [linear_signal, inner_product])
    elif use_outter:
        deep_input = tf.keras.layers.Concatenate()(
            [linear_signal, outter_product])
    else:
        deep_input = linear_signal

    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                   False, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)

    output = PredictionLayer(final_activation)(deep_logit)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=output)
    return model
