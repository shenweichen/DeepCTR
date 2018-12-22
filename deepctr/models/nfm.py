# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
"""

from tensorflow.python.keras.layers import Dense, Embedding, Concatenate, Reshape, Dropout, add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2
from ..layers import PredictionLayer, MLP, BiInteractionPooling
from ..utils import get_input, get_share_embeddings


def NFM(feature_dim_dict, embedding_size=8,
        hidden_size=(128, 128), l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_deep=0,
        init_std=0.0001, seed=1024, keep_prob=1, activation='relu', final_activation='sigmoid',
        ):
    """Instantiates the Neural Factorization Machine architecture.

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
    if not isinstance(feature_dim_dict,
                      dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
        raise ValueError(
            "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

    sparse_input, dense_input = get_input(feature_dim_dict, None)
    sparse_embedding, linear_embedding = get_share_embeddings(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear)

    embed_list = [sparse_embedding[i](sparse_input[i])
                  for i in range(len(sparse_input))]

    linear_term = [linear_embedding[i](sparse_input[i])
                   for i in range(len(sparse_input))]
    if len(linear_term) > 1:
        linear_term = add(linear_term)
    elif len(linear_term) == 1:
        linear_term = linear_term[0]
    else:
        linear_term = 0

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

    bi_out = BiInteractionPooling()(fm_input)
    bi_out = Dropout(1 - keep_prob)(bi_out)
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                   False, seed)(bi_out)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)

    final_logit = linear_term  # TODO add bias term

    if len(hidden_size) > 0:
        final_logit = add([final_logit, deep_logit])

    output = PredictionLayer(final_activation)(final_logit)
    print(output)
    model = Model(inputs=sparse_input + dense_input, outputs=output)
    return model
