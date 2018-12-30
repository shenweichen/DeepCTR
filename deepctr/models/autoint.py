# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)

"""

from tensorflow.python.keras.layers import Dense, Embedding, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf

from ..utils import get_input_list
from ..layers import PredictionLayer, MLP, InteractingLayer


def AutoInt(feature_dim_dict, embedding_size=8, att_layer_num=3, att_embedding_size=8, att_head_num=2, att_res=True, hidden_size=(256, 256), activation='relu',
            l2_reg_deep=0, l2_reg_embedding=1e-5, use_bn=False, keep_prob=1.0, init_std=0.0001, seed=1024,
            final_activation='sigmoid',):
    """Instantiates the AutoInt Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_embedding_size: int.The embedding size in multi-head self-attention network.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param activation: Activation function to use in deep net
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param use_bn:  bool. Whether use BatchNormalization before activation or not.in deep net
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param final_activation: output activation,usually ``'sigmoid'`` or ``'linear'``
    :return: A Keras model instance.
    """

    if len(hidden_size) <= 0 and att_layer_num <= 0:
        raise ValueError("Either hidden_layer or att_layer_num must > 0")
    if not isinstance(feature_dim_dict, dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
        raise ValueError(
            "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

    sparse_input, dense_input = get_input_list(feature_dim_dict)
    sparse_embedding = get_embeddings(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding)
    embed_list = [sparse_embedding[i](sparse_input[i])
                  for i in range(len(sparse_input))]

    att_input = Concatenate(axis=1)(embed_list) if len(
        embed_list) > 1 else embed_list[0]

    for i in range(att_layer_num):
        att_input = InteractingLayer(
            att_embedding_size, att_head_num, att_res)(att_input)
    att_output = tf.keras.layers.Flatten()(att_input)

    deep_input = tf.keras.layers.Flatten()(Concatenate()(embed_list)
                                           if len(embed_list) > 1 else embed_list[0])
    if len(dense_input) > 0:
        if len(dense_input) == 1:
            continuous_list = dense_input[0]
        else:
            continuous_list = Concatenate()(dense_input)

        deep_input = Concatenate()([deep_input, continuous_list])

    if len(hidden_size) > 0 and att_layer_num > 0:  # Deep & Interacting Layer
        deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                       use_bn, seed)(deep_input)
        stack_out = Concatenate()([att_output, deep_out])
        final_logit = Dense(1, use_bias=False, activation=None)(stack_out)
    elif len(hidden_size) > 0:  # Only Deep
        deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                       use_bn, seed)(deep_input)
        final_logit = Dense(1, use_bias=False, activation=None)(deep_out)
    elif att_layer_num > 0:  # Only Interacting Layer
        final_logit = Dense(1, use_bias=False, activation=None)(att_output)
    else:  # Error
        raise NotImplementedError

    output = PredictionLayer(final_activation)(final_logit)
    model = Model(inputs=sparse_input + dense_input, outputs=output)

    return model


def get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V):
    sparse_embedding = [Embedding(feature_dim_dict["sparse"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
        mean=0.0, stddev=init_std, seed=seed),
        embeddings_regularizer=l2(l2_rev_V),
        name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
        enumerate(feature_dim_dict["sparse"])]

    return sparse_embedding
