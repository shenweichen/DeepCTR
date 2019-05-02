# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhang W, Du T, Wang J. Deep learning over multi-field categorical data[C]//European conference on information retrieval. Springer, Cham, 2016: 45-57.(https://arxiv.org/pdf/1601.02376.pdf)
"""
import tensorflow as tf

from ..input_embedding import preprocess_input_embedding, get_linear_logit
from ..layers.core import PredictionLayer, DNN
from ..layers.utils import concat_fun
from ..utils import check_feature_config_dict


def FNN(feature_dim_dict, embedding_size=8,
        dnn_hidden_units=(128, 128),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0,
        init_std=0.0001, seed=1024, dnn_dropout=0,
        dnn_activation='relu', task='binary', ):
    """Instantiates the Factorization-supported Neural Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear weight
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, linear_emb_list, dense_input_dict, inputs_list = preprocess_input_embedding(feature_dim_dict,
                                                                                               embedding_size,
                                                                                               l2_reg_embedding,
                                                                                               l2_reg_linear, init_std,
                                                                                               seed,
                                                                                               create_linear_weight=True)

    linear_logit = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear)

    deep_input = tf.keras.layers.Flatten()(concat_fun(deep_emb_list))
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                   dnn_dropout, False, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)
    final_logit = tf.keras.layers.add([deep_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=output)
    return model
