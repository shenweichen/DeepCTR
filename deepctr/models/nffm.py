# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhang L, Shen W, Li S, et al. Field-aware Neural Factorization Machine for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1902.09096, 2019.(https://arxiv.org/abs/1902.09096)
"""

from tensorflow.python.keras.layers import Dense, Embedding, Concatenate, Reshape, Dropout, add, multiply, Layer, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal, Ones, RandomUniform
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
import tensorflow as tf


from ..layers.core import PredictionLayer, MLP
from ..input_embedding import create_singlefeat_inputdict,get_inputs_list,get_embedding_vec_list, get_linear_logit
from ..layers.utils import concat_fun
from ..utils import check_feature_config_dict
import itertools

# def get_embedding_idx(sp_input, field_num, feat):
#     # sp_input,sparse_embedding = inputs
#     multi_input = K.repeat_elements(sp_input, field_num, 1)
#     offset = K.cast(K.arange(0, field_num * feat.dimension, feat.dimension, dtype=tf.int32), tf.int32)
#     offset_c = K.constant(np.arange(0, field_num * feat.dimension, feat.dimension), tf.int32)
#
#     return multi_input + offset_c

def NFFM(feature_dim_dict, embedding_size=4, hidden_size=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_linear=1e-5,
         init_std=0.0001, seed=1024, final_activation='sigmoid', include_linear=True, bn=True, reduce_sum=False,
         ):
    """Instantiates the Field-aware Neural Factorization Machine architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_deep: float . L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param activation: Activation function to use in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :return: A Keras model instance.
    """
    check_feature_config_dict(feature_dim_dict)

    sparse_input_dict, dense_input_dict = create_singlefeat_inputdict(
        feature_dim_dict)

    sparse_embedding, dense_embedding,linear_embedding = get_embeddings(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear,)

    embed_list = []
    for i,j in itertools.combinations(feature_dim_dict['sparse'],2):
        element_wise_prod = multiply([sparse_embedding[i.name][j.name](sparse_input_dict[i.name]),sparse_embedding[j.name][i.name](sparse_input_dict[j.name])])
        if reduce_sum:
            element_wise_prod = Lambda(lambda element_wise_prod:K.sum(element_wise_prod,axis=-1))(element_wise_prod)
        embed_list.append(element_wise_prod)
    for i,j in itertools.combinations(feature_dim_dict['dense'],2):
        element_wise_prod = multiply([dense_embedding[i.name][j.name](dense_input_dict[i.name]),dense_embedding[j.name][i.name](dense_input_dict[j.name])])
        if reduce_sum:
            #K.expand_dims()
            element_wise_prod = Lambda(lambda element_wise_prod:K.sum(element_wise_prod,axis=-1))(element_wise_prod)
        embed_list.append(Lambda(lambda x: K.expand_dims(x,axis=1))(element_wise_prod))

    for i in feature_dim_dict['sparse']:
        for j in feature_dim_dict['dense']:
            element_wise_prod = multiply([sparse_embedding[i.name][j.name](sparse_input_dict[i.name]),
                                          dense_embedding[j.name][i.name](dense_input_dict[j.name])])

            if reduce_sum:
                element_wise_prod = Lambda(lambda element_wise_prod: K.sum(element_wise_prod, axis=-1))(
                    element_wise_prod)
            embed_list.append(element_wise_prod)


    ffm_out = tf.keras.layers.Flatten()(concat_fun(embed_list,axis=1))

    if bn:
        ffm_out = tf.keras.layers.BatchNormalization()(ffm_out)

    ffm_out = MLP(hidden_size)(ffm_out)
    final_logit = Dense(1,use_bias=False)(ffm_out)

    linear_emb_list = get_embedding_vec_list(
        linear_embedding, sparse_input_dict)

    linear_logit = get_linear_logit(
        linear_emb_list, dense_input_dict, l2_reg_linear)


    if include_linear:
        final_logit = add([final_logit, linear_logit])

    output = PredictionLayer(final_activation)(final_logit)

    inputs_list = get_inputs_list(
        [sparse_input_dict, dense_input_dict])
    model = Model(inputs=inputs_list, outputs=output)
    return model


def get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w,):

    sparse_embedding = {j.name: {feat.name: Embedding(j.dimension, embedding_size,
                                                          embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001,seed=seed),
                                                          embeddings_regularizer=l2(l2_rev_V),
                                                          name='sparse_emb_' + str(j.name) + '_' + str(
                                                              i) + '-' + feat.name) for i, feat in
                                     enumerate(feature_dim_dict["sparse"]+feature_dim_dict['dense'])} for j in feature_dim_dict["sparse"]}

    dense_embedding = {j.name: {feat.name: Dense(embedding_size,kernel_initializer=RandomNormal(mean=0.0, stddev=0.0001,
                                                                                          seed=seed),use_bias=False,kernel_regularizer=l2(l2_rev_V),name='sparse_emb_' + str(j.name) + '_' + str(
                                                          i) + '-' + feat.name) for i, feat in
                                 enumerate(feature_dim_dict["sparse"]+feature_dim_dict["dense"])} for j in feature_dim_dict["dense"]}

    linear_embedding = {feat.name: Embedding(feat.dimension, 1,
                                             embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,seed=seed),
                                             embeddings_regularizer=l2(l2_reg_w),
                                             name='linear_emb_' + str(i) + '-' + feat.name) for
                        i, feat in enumerate(feature_dim_dict["sparse"])}

    return sparse_embedding,dense_embedding, linear_embedding
