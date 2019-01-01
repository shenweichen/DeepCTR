# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhang W, Du T, Wang J. Deep learning over multi-field categorical data[C]//European conference on information retrieval. Springer, Cham, 2016: 45-57.(https://arxiv.org/pdf/1601.02376.pdf)
"""
import tensorflow as tf

from ..layers import PredictionLayer, MLP
from ..input_embedding import get_inputs_embedding
from ..utils import  concat_fun


def FNN(feature_dim_dict, embedding_size=8,
        hidden_size=(128, 128),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_deep=0,
        init_std=0.0001, seed=1024, keep_prob=1,
        activation='relu', final_activation='sigmoid', ):
    """Instantiates the Factorization-supported Neural Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear weight
    :param l2_reg_deep: float . L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param activation: Activation function to use in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :return: A Keras model instance.
    """
    if not isinstance(feature_dim_dict,
                      dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
        raise ValueError(
            "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

    deep_emb_list, linear_logit, inputs_list = get_inputs_embedding(
        feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed)

    deep_input = tf.keras.layers.Flatten()(concat_fun(deep_emb_list))
    deep_out = MLP(hidden_size, activation, l2_reg_deep,
                   keep_prob, False, seed)(deep_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)
    final_logit = tf.keras.layers.add([deep_logit, linear_logit])
    output = PredictionLayer(final_activation)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=output)
    return model
