# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhang L, Shen W, Li S, et al. Field-aware Neural Factorization Machine for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1902.09096, 2019.(https://arxiv.org/abs/1902.09096)
    (The original NFFM was first used by Yi Yang(yangyi868@gmail.com) in TSA competition in 2017.)
"""

import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Dense, Embedding, Lambda, add,
                                            multiply)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from ..input_embedding import (create_singlefeat_inputdict,
                               get_embedding_vec_list, get_inputs_list,
                               get_linear_logit)
from ..layers.core import DNN, PredictionLayer
from ..layers.utils import concat_fun,Hash
from ..utils import check_feature_config_dict


def NFFM(feature_dim_dict, embedding_size=4, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, dnn_dropout=0,
         init_std=0.0001, seed=1024, include_linear=True, use_bn=True, reduce_sum=False, task='binary',
         ):
    """Instantiates the Field-aware Neural Factorization Machine architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param include_linear: bool,whether include linear term or not
    :param use_bn: bool,whether use bn after ffm out or not
    :param reduce_sum: bool,whether apply reduce_sum on cross vector
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    check_feature_config_dict(feature_dim_dict)
    if 'sequence' in feature_dim_dict and len(feature_dim_dict['sequence']) > 0:
        raise ValueError("now sequence input is not supported in NFFM")#TODO:support sequence input

    sparse_input_dict, dense_input_dict = create_singlefeat_inputdict(
        feature_dim_dict)

    sparse_embedding, dense_embedding, linear_embedding = create_embedding_dict(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear, )

    embed_list = []
    for i, j in itertools.combinations(feature_dim_dict['sparse'], 2):
        i_input = sparse_input_dict[i.name]
        if i.hash_flag:
            i_input = Hash(i.dimension)(i_input)
        j_input = sparse_input_dict[j.name]
        if j.hash_flag:
            j_input = Hash(j.dimension)(j_input)

        element_wise_prod = multiply([sparse_embedding[i.name][j.name](i_input), sparse_embedding[j.name][i.name](j_input)])
        if reduce_sum:
            element_wise_prod = Lambda(lambda element_wise_prod: K.sum(
                element_wise_prod, axis=-1))(element_wise_prod)
        embed_list.append(element_wise_prod)
    for i, j in itertools.combinations(feature_dim_dict['dense'], 2):
        element_wise_prod = multiply([dense_embedding[i.name][j.name](
            dense_input_dict[i.name]), dense_embedding[j.name][i.name](dense_input_dict[j.name])])
        if reduce_sum:
            element_wise_prod = Lambda(lambda element_wise_prod: K.sum(
                element_wise_prod, axis=-1))(element_wise_prod)
        embed_list.append(
            Lambda(lambda x: K.expand_dims(x, axis=1))(element_wise_prod))

    for i in feature_dim_dict['sparse']:
        i_input = sparse_input_dict[i.name]
        if i.hash_flag:
            i_input = Hash(i.dimension)(i_input)
        for j in feature_dim_dict['dense']:
            element_wise_prod = multiply([sparse_embedding[i.name][j.name](i_input),
                                          dense_embedding[j.name][i.name](dense_input_dict[j.name])])

            if reduce_sum:
                element_wise_prod = Lambda(lambda element_wise_prod: K.sum(element_wise_prod, axis=-1))(
                    element_wise_prod)
            embed_list.append(element_wise_prod)

    ffm_out = tf.keras.layers.Flatten()(concat_fun(embed_list, axis=1))
    if use_bn:
        ffm_out = tf.keras.layers.BatchNormalization()(ffm_out)
    ffm_out = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout)(ffm_out)
    final_logit = Dense(1, use_bias=False)(ffm_out)

    linear_emb_list = get_embedding_vec_list(
        linear_embedding, sparse_input_dict, feature_dim_dict['sparse'])

    linear_logit = get_linear_logit(
        linear_emb_list, dense_input_dict, l2_reg_linear)

    if include_linear:
        final_logit = add([final_logit, linear_logit])

    output = PredictionLayer(task)(final_logit)

    inputs_list = get_inputs_list(
        [sparse_input_dict, dense_input_dict])
    model = Model(inputs=inputs_list, outputs=output)
    return model


def create_embedding_dict(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w, ):
    sparse_embedding = {j.name: {feat.name: Embedding(j.dimension, embedding_size,
                                                      embeddings_initializer=RandomNormal(
                                                          mean=0.0, stddev=0.0001, seed=seed),
                                                      embeddings_regularizer=l2(
                                                          l2_rev_V),
                                                      name='sparse_emb_' + str(j.name) + '_' + str(
                                                          i) + '-' + feat.name) for i, feat in
                                 enumerate(feature_dim_dict["sparse"] + feature_dim_dict['dense'])} for j in
                        feature_dim_dict["sparse"]}

    dense_embedding = {
        j.name: {feat.name: Dense(embedding_size, kernel_initializer=RandomNormal(mean=0.0, stddev=0.0001,
                                                                                  seed=seed), use_bias=False,
                                  kernel_regularizer=l2(l2_rev_V), name='sparse_emb_' + str(j.name) + '_' + str(
                i) + '-' + feat.name) for i, feat in
                 enumerate(feature_dim_dict["sparse"] + feature_dim_dict["dense"])} for j in feature_dim_dict["dense"]}

    linear_embedding = {feat.name: Embedding(feat.dimension, 1,
                                             embeddings_initializer=RandomNormal(
                                                 mean=0.0, stddev=init_std, seed=seed),
                                             embeddings_regularizer=l2(
                                                 l2_reg_w),
                                             name='linear_emb_' + str(i) + '-' + feat.name) for
                        i, feat in enumerate(feature_dim_dict["sparse"])}

    return sparse_embedding, dense_embedding, linear_embedding
