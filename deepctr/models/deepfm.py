# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

from tensorflow.python.keras.layers import Dense, Concatenate, Reshape, Flatten, add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from ..utils import get_input, get_share_embeddings
from ..layers import PredictionLayer, MLP, FM


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
    if not isinstance(feature_dim_dict,
                      dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
        raise ValueError(
            "feature_dim_dict must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")
    if not isinstance(feature_dim_dict["sparse"], dict):
        raise ValueError("feature_dim_dict['sparse'] must be a dict,cur is", type(
            feature_dim_dict['sparse']))
    if not isinstance(feature_dim_dict["dense"], list):
        raise ValueError("feature_dim_dict['dense'] must be a list,cur is", type(
            feature_dim_dict['dense']))

    sparse_input, dense_input = get_input(feature_dim_dict, None)
    sparse_embedding, linear_embedding, = get_share_embeddings(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear)

    embed_list = [sparse_embedding[i](sparse_input[i])
                  for i in range(len(sparse_input))]
    linear_term = [linear_embedding[i](sparse_input[i])
                   for i in range(len(sparse_input))]
    if len(linear_term) > 1:
        linear_term = add(linear_term)
    elif len(linear_term) > 0:
        linear_term = linear_term[0]

    if len(dense_input) > 0:
        continuous_embedding_list = list(
            map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding), ),
                dense_input))
        continuous_embedding_list = list(
            map(Reshape((1, embedding_size)), continuous_embedding_list))
        embed_list += continuous_embedding_list

        dense_input_ = dense_input[0] if len(
            dense_input) == 1 else Concatenate()(dense_input)
        linear_dense_logit = Dense(
            1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg_linear))(dense_input_)
        linear_term = add([linear_dense_logit, linear_term])

    fm_input = Concatenate(axis=1)(embed_list)
    deep_input = Flatten()(fm_input)
    fm_out = FM()(fm_input)
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                   use_bn, seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)

    if len(hidden_size) == 0 and use_fm == False:  # only linear
        final_logit = linear_term
    elif len(hidden_size) == 0 and use_fm == True:  # linear + FM
        final_logit = add([linear_term, fm_out])
    elif len(hidden_size) > 0 and use_fm == False:  # linear +　Deep
        final_logit = add([linear_term, deep_logit])
    elif len(hidden_size) > 0 and use_fm == True:  # linear + FM + Deep
        final_logit = add([linear_term, fm_out, deep_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer(final_activation)(final_logit)
    model = Model(inputs=sparse_input + dense_input, outputs=output)
    return model
