# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhang W, Du T, Wang J. Deep learning over multi-field categorical data[C]//European conference on information retrieval. Springer, Cham, 2016: 45-57.(https://arxiv.org/pdf/1601.02376.pdf)
"""

from tensorflow.python.keras.layers import Dense, Concatenate, Reshape, add
from tensorflow.python.keras.models import Model

from ..layers import PredictionLayer, MLP
from ..utils import create_input_dict, create_embedding_dict, get_embedding_vec_list, get_inputs_list, get_linear_logit, embed_dense_input


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

    sparse_input, dense_input = create_input_dict(feature_dim_dict)
    sparse_embedding = create_embedding_dict(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding)
    linear_embedding = create_embedding_dict(
        feature_dim_dict, 1, init_std, seed, l2_reg_linear, prefix='linear')
    embed_list = get_embedding_vec_list(sparse_embedding, sparse_input)
    linear_term = get_embedding_vec_list(linear_embedding, sparse_input)

    embed_list = embed_dense_input(
        dense_input, embed_list, embedding_size, l2_reg_embedding)
    linear_term = get_linear_logit(linear_term, dense_input, l2_reg_linear)

    num_inputs = len(dense_input) + len(sparse_input)
    deep_input = Reshape([num_inputs*embedding_size]
                         )(Concatenate()(embed_list))
    deep_out = MLP(hidden_size, activation, l2_reg_deep,
                   keep_prob, False, seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)
    final_logit = add([deep_logit, linear_term])
    output = PredictionLayer(final_activation)(final_logit)
    inputs_list = get_inputs_list([sparse_input, dense_input])
    model = Model(inputs=inputs_list,
                  outputs=output)
    return model
