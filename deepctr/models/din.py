# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

from collections import OrderedDict

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from ..input_embedding import get_inputs_list, create_singlefeat_inputdict, get_embedding_vec_list
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer
from ..layers.utils import concat_fun, NoMask
from ..utils import check_feature_config_dict


def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = OrderedDict()
    for i, feat in enumerate(seq_feature_list):
        user_behavior_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat)

    return sparse_input, dense_input, user_behavior_input


def DIN(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=16,
        dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40),
        att_activation="dice", att_weight_normalization=False,
        l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
    :param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    check_feature_config_dict(feature_dim_dict)

    sparse_input, dense_input, user_behavior_input = get_input(
        feature_dim_dict, seq_feature_list, hist_len_max)

    sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
                                                  embeddings_initializer=RandomNormal(
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' + str(i) + '-' + feat.name,
                                                  mask_zero=(feat.name in seq_feature_list)) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}

    query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'],
                                            seq_feature_list, seq_feature_list)

    keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input, feature_dim_dict['sparse'],
                                           seq_feature_list, seq_feature_list)

    deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'],
                                                 mask_feat_list=seq_feature_list)

    keys_emb = concat_fun(keys_emb_list)
    deep_input_emb = concat_fun(deep_input_emb_list)

    query_emb = concat_fun(query_emb_list)

    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                         weight_normalization=att_weight_normalization, supports_masking=True)([
        query_emb, keys_emb])

    deep_input_emb = Concatenate()([NoMask()(deep_input_emb), hist])
    deep_input_emb = Flatten()(deep_input_emb)
    if len(dense_input) > 0:
        deep_input_emb = Concatenate()([deep_input_emb] + list(dense_input.values()))

    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(deep_input_emb)
    final_logit = Dense(1, use_bias=False)(output)

    output = PredictionLayer(task)(final_logit)
    model_input_list = get_inputs_list([sparse_input, dense_input, user_behavior_input])

    model = Model(inputs=model_input_list, outputs=output)
    return model
