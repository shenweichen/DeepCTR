# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)
"""
import tensorflow as tf

from ..input_embedding import preprocess_input_embedding
from ..layers.core import PredictionLayer, MLP
from ..layers.interaction import CrossNet
from ..utils import check_feature_config_dict
from ..layers.utils import concat_fun


def DCN(feature_dim_dict, embedding_size='auto',
        cross_num=2, hidden_size=(128, 128, ), l2_reg_embedding=1e-5, l2_reg_cross=1e-5, l2_reg_deep=0,
        init_std=0.0001, seed=1024, keep_prob=1, use_bn=False, activation='relu', final_activation='sigmoid',
        ):
    """Instantiates the Deep&Cross Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive int or str,sparse feature embedding_size.If set to "auto",it will be 6*pow(cardinality,025)
    :param cross_num: positive integet,cross layer number
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param use_bn: bool. Whether use BatchNormalization before activation or not.in deep net
    :param activation: Activation function to use in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :return: A Keras model instance.

    """
    if len(hidden_size) == 0 and cross_num == 0:
        raise ValueError("Either hidden_layer or cross layer must > 0")

    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, _, inputs_list = preprocess_input_embedding(feature_dim_dict, embedding_size,
                                                                          l2_reg_embedding, 0, init_std,
                                                                          seed, False)

    deep_input = tf.keras.layers.Flatten()(concat_fun(deep_emb_list))

    if len(hidden_size) > 0 and cross_num > 0:  # Deep & Cross
        deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                       use_bn, seed)(deep_input)
        cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross)(deep_input)
        stack_out = tf.keras.layers.Concatenate()([cross_out, deep_out])
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(stack_out)
    elif len(hidden_size) > 0:  # Only Deep
        deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                       use_bn, seed)(deep_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)
    elif cross_num > 0:  # Only Cross
        cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross)(deep_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cross_out)
    else:  # Error
        raise NotImplementedError

    output = PredictionLayer(final_activation)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
