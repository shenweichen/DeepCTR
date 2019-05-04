# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
"""

from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Input, Permute, multiply)
from tensorflow.python.keras.regularizers import l2

from ..input_embedding import create_singlefeat_inputdict, get_inputs_list,get_embedding_vec_list
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer, DynamicGRU
from ..layers.utils import concat_fun
from ..utils import check_feature_config_dict


def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = OrderedDict()
    for i, feat in enumerate(seq_feature_list):
        user_behavior_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat)

    user_behavior_length = Input(shape=(1,), name='seq_length')

    return sparse_input, dense_input, user_behavior_input, user_behavior_length


def auxiliary_loss(h_states, click_seq, noclick_seq, mask, stag=None):
    #:param h_states:
    #:param click_seq:
    #:param noclick_seq: #[B,T-1,E]
    #:param mask:#[B,1]
    #:param stag:
    #:return:
    hist_len, _ = click_seq.get_shape().as_list()[1:]
    mask = tf.sequence_mask(mask, hist_len)
    mask = mask[:, 0, :]

    mask = tf.cast(mask, tf.float32)

    click_input_ = tf.concat([h_states, click_seq], -1)

    noclick_input_ = tf.concat([h_states, noclick_seq], -1)

    click_prop_ = auxiliary_net(click_input_, stag=stag)[:, :, 0]

    noclick_prop_ = auxiliary_net(noclick_input_, stag=stag)[
                    :, :, 0]  # [B,T-1]

    click_loss_ = - tf.reshape(tf.log(click_prop_),
                               [-1, tf.shape(click_seq)[1]]) * mask

    noclick_loss_ = - \
                        tf.reshape(tf.log(1.0 - noclick_prop_),
                                   [-1, tf.shape(noclick_seq)[1]]) * mask

    loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)

    return loss_


def auxiliary_net(in_, stag='auxiliary_net'):
    bn1 = tf.layers.batch_normalization(
        inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)

    dnn1 = tf.layers.dense(bn1, 100, activation=None,
                           name='f1' + stag, reuse=tf.AUTO_REUSE)

    dnn1 = tf.nn.sigmoid(dnn1)

    dnn2 = tf.layers.dense(dnn1, 50, activation=None,
                           name='f2' + stag, reuse=tf.AUTO_REUSE)

    dnn2 = tf.nn.sigmoid(dnn2)

    dnn3 = tf.layers.dense(dnn2, 1, activation=None,
                           name='f3' + stag, reuse=tf.AUTO_REUSE)

    y_hat = tf.nn.sigmoid(dnn3)

    return y_hat


def interest_evolution(concat_behavior, deep_input_item, user_behavior_length, gru_type="GRU", use_neg=False,
                       neg_concat_behavior=None, embedding_size=8, att_hidden_size=(64, 16), att_activation='sigmoid',
                       att_weight_normalization=False, ):
    if gru_type not in ["GRU", "AIGRU", "AGRU", "AUGRU"]:
        raise ValueError("gru_type error ")
    aux_loss_1 = None

    rnn_outputs = DynamicGRU(embedding_size * 2, return_sequence=True,
                             name="gru1")([concat_behavior, user_behavior_length])

    if gru_type == "AUGRU" and use_neg:
        aux_loss_1 = auxiliary_loss(rnn_outputs[:, :-1, :], concat_behavior[:, 1:, :],

                                    neg_concat_behavior[:, 1:, :],

                                    tf.subtract(user_behavior_length, 1), stag="gru")  # [:, 1:]

    if gru_type == "GRU":
        rnn_outputs2 = DynamicGRU(embedding_size * 2, return_sequence=True,
                                  name="gru2")([rnn_outputs, user_behavior_length])
        # attention_score = AttentionSequencePoolingLayer(hidden_size=att_hidden_size, activation=att_activation, weight_normalization=att_weight_normalization, return_score=True)([
        #     deep_input_item, rnn_outputs2, user_behavior_length])
        # outputs = Lambda(lambda x: tf.matmul(x[0], x[1]))(
        #     [attention_score, rnn_outputs2])
        # hist = outputs
        hist = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                             weight_normalization=att_weight_normalization, return_score=False)([
            deep_input_item, rnn_outputs2, user_behavior_length])

    else:  # AIGRU AGRU AUGRU

        scores = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                               weight_normalization=att_weight_normalization, return_score=True)([
            deep_input_item, rnn_outputs, user_behavior_length])

        if gru_type == "AIGRU":
            hist = multiply([rnn_outputs, Permute([2, 1])(scores)])
            final_state2 = DynamicGRU(embedding_size * 2, gru_type="GRU", return_sequence=False, name='gru2')(
                [hist, user_behavior_length])
        else:  # AGRU AUGRU
            final_state2 = DynamicGRU(embedding_size * 2, gru_type=gru_type, return_sequence=False,
                                      name='gru2')([rnn_outputs, user_behavior_length, Permute([2, 1])(scores)])
        hist = final_state2
    return hist, aux_loss_1


def DIEN(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=16,
         gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(200, 80),
         dnn_activation='relu',
         att_hidden_units=(64, 16), att_activation="dice", att_weight_normalization=True,
         l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the Deep Interest Evolution Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
    :param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param gru_type: str,can be GRU AIGRU AUGRU AGRU
    :param use_negsampling: bool, whether or not use negtive sampling
    :param alpha: float ,weight of auxiliary_loss
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
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

    sparse_input, dense_input, user_behavior_input, user_behavior_length = get_input(
        feature_dim_dict, seq_feature_list, hist_len_max)
    sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
                                                  embeddings_initializer=RandomNormal(
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' + str(i) + '-' + feat.name) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}

    query_emb_list = get_embedding_vec_list(sparse_embedding_dict,sparse_input,feature_dim_dict["sparse"],return_feat_list=seq_feature_list)
    keys_emb_list = get_embedding_vec_list(sparse_embedding_dict,user_behavior_input,feature_dim_dict['sparse'],return_feat_list=seq_feature_list)
    deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict['sparse'])

    query_emb = concat_fun(query_emb_list)
    keys_emb = concat_fun(keys_emb_list)
    deep_input_emb = concat_fun(deep_input_emb_list)


    if use_negsampling:
        neg_user_behavior_input = OrderedDict()
        for i, feat in enumerate(seq_feature_list):
            neg_user_behavior_input[feat] = Input(shape=(hist_len_max,), name='neg_seq_' + str(i) + '-' + feat)

        neg_uiseq_embed_list = get_embedding_vec_list(sparse_embedding_dict,neg_user_behavior_input,feature_dim_dict["sparse"],seq_feature_list,)
           # [sparse_embedding_dict[feat](
           # neg_user_behavior_input[feat]) for feat in seq_feature_list]
        neg_concat_behavior = concat_fun(neg_uiseq_embed_list)

    else:
        neg_concat_behavior = None

    hist, aux_loss_1 = interest_evolution(keys_emb, query_emb, user_behavior_length, gru_type=gru_type,
                                          use_neg=use_negsampling, neg_concat_behavior=neg_concat_behavior,
                                          embedding_size=embedding_size, att_hidden_size=att_hidden_units,
                                          att_activation=att_activation,
                                          att_weight_normalization=att_weight_normalization, )

    deep_input_emb = Concatenate()([deep_input_emb, hist])

    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
    if len(dense_input) > 0:
        deep_input_emb = Concatenate()(
            [deep_input_emb] + list(dense_input.values()))

    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, use_bn, seed)(deep_input_emb)
    final_logit = Dense(1, use_bias=False)(output)
    output = PredictionLayer(task)(final_logit)

    model_input_list = get_inputs_list(
        [sparse_input, dense_input, user_behavior_input])

    if use_negsampling:
        model_input_list += list(neg_user_behavior_input.values())

    model_input_list += [user_behavior_length]

    model = tf.keras.models.Model(inputs=model_input_list, outputs=output)

    if use_negsampling:
        model.add_loss(alpha * aux_loss_1)
    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    return model
