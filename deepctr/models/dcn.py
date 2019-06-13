# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)
"""
import tensorflow as tf

from ..input_embedding import preprocess_input_embedding
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import CrossNet
from ..layers.utils import concat_fun
from ..utils import check_feature_config_dict


def DCN(feature_dim_dict, embedding_size='auto',
        cross_num=2, dnn_hidden_units=(128, 128,), l2_reg_embedding=1e-5, l2_reg_cross=1e-5, l2_reg_dnn=0,
        init_std=0.0001, seed=1024, dnn_dropout=0, dnn_use_bn=False, dnn_activation='relu', task='binary',
        ):
    """Instantiates the Deep&Cross Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive int or str,sparse feature embedding_size.If set to "auto",it will be 6*pow(cardinality,025)
    :param cross_num: positive integet,cross layer number
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    if len(dnn_hidden_units) == 0 and cross_num == 0:
        raise ValueError("Either hidden_layer or cross layer must > 0")

    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, _, _, inputs_list = preprocess_input_embedding(feature_dim_dict,
                                                                  embedding_size,
                                                                  l2_reg_embedding,
                                                                  0, init_std,
                                                                  seed,
                                                                  create_linear_weight=False)

    deep_input = tf.keras.layers.Flatten()(concat_fun(deep_emb_list))

    if len(dnn_hidden_units) > 0 and cross_num > 0:  # Deep & Cross
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed)(deep_input)
        cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross)(deep_input)
        stack_out = tf.keras.layers.Concatenate()([cross_out, deep_out])
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(stack_out)
    elif len(dnn_hidden_units) > 0:  # Only Deep
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed)(deep_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)
    elif cross_num > 0:  # Only Cross
        cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross)(deep_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cross_out)
    else:  # Error
        raise NotImplementedError

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
