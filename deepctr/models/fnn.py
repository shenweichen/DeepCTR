# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhang, Weinan, Tianming Du, and Jun Wang. "Deep learning over multi-field categorical data." European conference on information retrieval. Springer, Cham, 2016.(https://arxiv.org/pdf/1601.02376.pdf)
"""

from tensorflow.python.keras.layers import Dense, Embedding, Concatenate, Reshape, add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2

from ..layers import PredictionLayer, MLP
from ..utils import get_input


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

    sparse_input, dense_input = get_input(feature_dim_dict, None)
    # sparse_embedding = [Embedding(feature_dim_dict["sparse"][feat], embedding_size,
    #                              embeddings_initializer=RandomNormal( mean=0.0, stddev=init_std, seed=seed),
    #   embeddings_regularizer=l2( l2_reg_embedding),name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
    #   enumerate(feature_dim_dict["sparse"])]
    sparse_embedding, linear_embedding, = get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding,
                                                         l2_reg_linear)

    embed_list = [sparse_embedding[i](sparse_input[i])
                  for i in range(len(feature_dim_dict["sparse"]))]

    linear_term = [linear_embedding[i](sparse_input[i])
                   for i in range(len(sparse_input))]
    if len(linear_term) > 1:
        linear_term = add(linear_term)
    elif len(linear_term) > 0:
        linear_term = linear_term[0]

    #linear_term = add([linear_embedding[i](sparse_input[i]) for i in range(len(feature_dim_dict["sparse"]))])
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

    num_inputs = len(dense_input) + len(sparse_input)
    deep_input = Reshape([num_inputs*embedding_size]
                         )(Concatenate()(embed_list))
    deep_out = MLP(hidden_size, activation, l2_reg_deep,
                   keep_prob, False, seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)
    final_logit = add([deep_logit, linear_term])
    output = PredictionLayer(final_activation)(final_logit)
    model = Model(inputs=sparse_input + dense_input,
                  outputs=output)
    return model


def get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    sparse_embedding = [Embedding(feature_dim_dict["sparse"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(feature_dim_dict["sparse"])]
    linear_embedding = [Embedding(feature_dim_dict["sparse"][feat], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                      seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(feature_dim_dict["sparse"])]

    return sparse_embedding, linear_embedding
