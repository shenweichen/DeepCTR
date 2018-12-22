# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
"""

from tensorflow.python.keras.layers import Dense, Concatenate, Flatten, add
from tensorflow.python.keras.models import Model
from ..layers import PredictionLayer, MLP
from ..utils import get_input, get_sep_embeddings


def WDL(deep_feature_dim_dict, wide_feature_dim_dict, embedding_size=8, hidden_size=(128, 128), l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_deep=0, init_std=0.0001, seed=1024, keep_prob=1, activation='relu', final_activation='sigmoid',):
    """Instantiates the Wide&Deep Learning architecture.

    :param deep_feature_dim_dict: dict,to indicate sparse field and dense field in deep part like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param wide_feature_dim_dict: dict,to indicate sparse field and dense field in wide part like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param activation: Activation function to use in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :return: A Keras model instance.
    """
    if not isinstance(deep_feature_dim_dict,
                      dict) or "sparse" not in deep_feature_dim_dict or "dense" not in deep_feature_dim_dict:
        raise ValueError(
            "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

    sparse_input, dense_input, bias_sparse_input, bias_dense_input = get_input(
        deep_feature_dim_dict, wide_feature_dim_dict)
    sparse_embedding, wide_linear_embedding = get_sep_embeddings(
        deep_feature_dim_dict, wide_feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear)

    embed_list = [sparse_embedding[i](sparse_input[i])
                  for i in range(len(sparse_input))]
    deep_input = Concatenate()(embed_list) if len(
        embed_list) > 1 else embed_list[0]
    deep_input = Flatten()(deep_input)
    if len(dense_input) > 0:
        deep_input = Concatenate()([deep_input]+dense_input)

    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                   False, seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)
    final_logit = deep_logit
    if len(wide_feature_dim_dict['dense']) + len(wide_feature_dim_dict['sparse']) > 0:
        if len(wide_feature_dim_dict['sparse']) > 0:
            bias_embed_list = [wide_linear_embedding[i](
                bias_sparse_input[i]) for i in range(len(bias_sparse_input))]
            linear_term = add(bias_embed_list) if len(
                bias_embed_list) > 1 else bias_embed_list[0]
            final_logit = add([final_logit, linear_term])
        if len(wide_feature_dim_dict['dense']) > 0:
            wide_dense_term = Dense(1, use_bias=False, activation=None)(Concatenate()(
                bias_dense_input) if len(bias_dense_input) > 1 else bias_dense_input[0])
            final_logit = add([final_logit, wide_dense_term])

    output = PredictionLayer(final_activation)(final_logit)
    model = Model(inputs=sparse_input + dense_input +
                  bias_sparse_input + bias_dense_input, outputs=output)
    return model
