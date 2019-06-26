# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)

"""

import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Dense, Embedding, Lambda, add,
                                            multiply)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from ..inputs import (build_input_features,VarLenSparseFeat,
                      get_linear_logit,SparseFeat,get_dense_input,combined_dnn_input)
from ..layers.core import DNN, PredictionLayer
from ..layers.utils import concat_fun,Hash
from ..utils import check_feature_config_dict


def ONN(linear_feature_columns,dnn_feature_columns, embedding_size=4, dnn_hidden_units=(128, 128),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, dnn_dropout=0,
        init_std=0.0001, seed=1024, include_linear=True, use_bn=True, reduce_sum=False, task='binary',
        ):
    """Instantiates the Operation-aware Neural Networks  architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
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

    #check_feature_config_dict(feature_dim_dict)
    #todo 需要修改
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    if len(varlen_sparse_feature_columns)> 0:
        raise ValueError("VarLenSparseFeat is not supported in ONN now")#TODO:support sequence input


    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features,linear_feature_columns,
                                       l2_reg_linear, init_std,seed,prefix='linear')

    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),dnn_feature_columns)) if dnn_feature_columns else []
    #varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []



    sparse_embedding = {fc_j.embedding_name: {fc_i.embedding_name: Embedding(fc_j.dimension, embedding_size,
                                                      embeddings_initializer=RandomNormal(
                                                          mean=0.0, stddev=0.0001, seed=seed),
                                                      embeddings_regularizer=l2(
                                                          l2_reg_embedding),
                                                      name='sparse_emb_' + str(fc_j.embedding_name) + '_' + str(
                                                          i) + '-' + fc_i.embedding_name) for i, fc_i in
                                 enumerate(sparse_feature_columns)} for fc_j in
                        sparse_feature_columns}


    dense_value_list = get_dense_input(features,dnn_feature_columns)


    embed_list = []
    for fc_i, fc_j in itertools.combinations(sparse_feature_columns, 2):
        i_input = features[fc_i.name]
        if fc_i.use_hash:
            i_input = Hash(fc_i.dimension)(i_input)
        j_input = features[fc_j.name]
        if fc_j.use_hash:
            j_input = Hash(fc_j.dimension)(j_input)

        element_wise_prod = multiply([sparse_embedding[fc_i.name][fc_j.name](i_input), sparse_embedding[fc_j.name][fc_i.name](j_input)])
        if reduce_sum:
            element_wise_prod = Lambda(lambda element_wise_prod: K.sum(
                element_wise_prod, axis=-1))(element_wise_prod)
        embed_list.append(element_wise_prod)

    print(embed_list)
    ffm_out = tf.keras.layers.Flatten()(concat_fun(embed_list, axis=1))
    if use_bn:
        ffm_out = tf.keras.layers.BatchNormalization()(ffm_out)
    dnn_input = combined_dnn_input([ffm_out],dense_value_list)
    dnn_out = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout)(dnn_input)
    final_logit = Dense(1, use_bias=False)(dnn_out)

    if include_linear:
        final_logit = add([final_logit, linear_logit])

    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model

