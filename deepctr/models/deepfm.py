# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

import tensorflow as tf
from ..input_embedding import preprocess_input_embedding
from ..layers.core import PredictionLayer, MLP
from ..layers.interaction import FM
from ..utils import check_feature_config_dict
from ..layers.utils import concat_fun


def DeepFM(feature_dim_dict, embedding_size=8,
           use_fm=True, hidden_size=(128, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_deep=0,
           init_std=0.0001, seed=1024, keep_prob=1, activation='relu', final_activation='sigmoid', use_bn=False):
    """Instantiates the DeepFM Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_fm: bool,use FM part or not
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param activation: Activation function to use in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param use_bn: bool. Whether use BatchNormalization before activation or not.in deep net
    :return: A Keras model instance.
    """
    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, linear_logit, inputs_list = preprocess_input_embedding(feature_dim_dict, embedding_size,
                                                                          l2_reg_embedding, l2_reg_linear, init_std,
                                                                          seed, True)

    fm_input = concat_fun(deep_emb_list, axis=1)
    deep_input = tf.keras.layers.Flatten()(fm_input)
    fm_out = FM()(fm_input)
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                   use_bn, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)

    if len(hidden_size) == 0 and use_fm == False:  # only linear
        final_logit = linear_logit
    elif len(hidden_size) == 0 and use_fm == True:  # linear + FM
        final_logit = tf.keras.layers.add([linear_logit, fm_out])
    elif len(hidden_size) > 0 and use_fm == False:  # linear +　Deep
        final_logit = tf.keras.layers.add([linear_logit, deep_logit])
    elif len(hidden_size) > 0 and use_fm == True:  # linear + FM + Deep
        final_logit = tf.keras.layers.add([linear_logit, fm_out, deep_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer(final_activation)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
