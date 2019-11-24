# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
"""


import tensorflow as tf
from tensorflow.python.keras.layers import (Concatenate, Dense, Input, Permute, multiply)

from ..inputs import build_input_features, get_varlen_pooling_list,create_embedding_matrix,embedding_lookup,varlen_embedding_lookup,SparseFeat,DenseFeat,VarLenSparseFeat,get_dense_input,combined_dnn_input
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer, DynamicGRU
from ..layers.utils import concat_func,reduce_mean


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

    try:
        click_loss_ = - tf.reshape(tf.log(click_prop_),
                                   [-1, tf.shape(click_seq)[1]]) * mask
    except:
        click_loss_ = - tf.reshape(tf.compat.v1.log(click_prop_),
                                   [-1, tf.shape(click_seq)[1]]) * mask
    try:
        noclick_loss_ = - \
                            tf.reshape(tf.log(1.0 - noclick_prop_),
                                       [-1, tf.shape(noclick_seq)[1]]) * mask
    except:
        noclick_loss_ = - \
                            tf.reshape(tf.compat.v1.log(1.0 - noclick_prop_),
                                       [-1, tf.shape(noclick_seq)[1]]) * mask


    loss_ = reduce_mean(click_loss_ + noclick_loss_)

    return loss_


def auxiliary_net(in_, stag='auxiliary_net'):
    try:
        bn1 = tf.layers.batch_normalization(
            inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
    except:
        bn1 = tf.compat.v1.layers.batch_normalization(
            inputs=in_, name='bn1' + stag, reuse=tf.compat.v1.AUTO_REUSE)

    try:#todo
        dnn1 = tf.layers.dense(bn1, 100, activation=None,
                               name='f1' + stag, reuse=tf.AUTO_REUSE)
    except:
        dnn1 = tf.compat.v1.layers.dense(bn1, 100, activation=None,
                               name='f1' + stag, reuse=tf.compat.v1.AUTO_REUSE)


    dnn1 = tf.nn.sigmoid(dnn1)
    try:
        dnn2 = tf.layers.dense(dnn1, 50, activation=None,
                           name='f2' + stag, reuse=tf.AUTO_REUSE)
    except:
        dnn2 = tf.compat.v1.layers.dense(dnn1, 50, activation=None,
                           name='f2' + stag, reuse=tf.compat.v1.AUTO_REUSE)

    dnn2 = tf.nn.sigmoid(dnn2)
    try:
        dnn3 = tf.layers.dense(dnn2, 1, activation=None,
                               name='f3' + stag, reuse=tf.AUTO_REUSE)
    except:
        dnn3 = tf.compat.v1.layers.dense(dnn2, 1, activation=None,
                               name='f3' + stag, reuse=tf.compat.v1.AUTO_REUSE)

    y_hat = tf.nn.sigmoid(dnn3)

    return y_hat


def interest_evolution(concat_behavior, deep_input_item, user_behavior_length, gru_type="GRU", use_neg=False,
                       neg_concat_behavior=None,att_hidden_size=(64, 16), att_activation='sigmoid',
                       att_weight_normalization=False, ):
    if gru_type not in ["GRU", "AIGRU", "AGRU", "AUGRU"]:
        raise ValueError("gru_type error ")
    aux_loss_1 = None
    embedding_size = None
    rnn_outputs = DynamicGRU(embedding_size, return_sequence=True,
                             name="gru1")([concat_behavior, user_behavior_length])

    if gru_type == "AUGRU" and use_neg:
        aux_loss_1 = auxiliary_loss(rnn_outputs[:, :-1, :], concat_behavior[:, 1:, :],

                                    neg_concat_behavior[:, 1:, :],

                                    tf.subtract(user_behavior_length, 1), stag="gru")  # [:, 1:]

    if gru_type == "GRU":
        rnn_outputs2 = DynamicGRU(embedding_size, return_sequence=True,
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
            final_state2 = DynamicGRU(embedding_size, gru_type="GRU", return_sequence=False, name='gru2')(
                [hist, user_behavior_length])
        else:  # AGRU AUGRU
            final_state2 = DynamicGRU(embedding_size, gru_type=gru_type, return_sequence=False,
                                      name='gru2')([rnn_outputs, user_behavior_length, Permute([2, 1])(scores)])
        hist = final_state2
    return hist, aux_loss_1


def DIEN(dnn_feature_columns, history_feature_list,
         gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(200, 80),
         dnn_activation='relu',
         att_hidden_units=(64, 16), att_activation="dice", att_weight_normalization=True,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the Deep Interest Evolution Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
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
    # check_feature_config_dict(feature_columns)
    #
    # sparse_input, dense_input, user_behavior_input, user_behavior_length = get_input(
    #     feature_columns, seq_feature_list, hist_len_max)
    # sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
    #                                               embeddings_initializer=RandomNormal(
    #                                                   mean=0.0, stddev=init_std, seed=seed),
    #                                               embeddings_regularizer=l2(
    #                                                   l2_reg_embedding),
    #                                               name='sparse_emb_' + str(i) + '-' + feat.name) for i, feat in
    #                          enumerate(feature_columns["sparse"])}
    #
    # query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_columns["sparse"], return_feat_list=seq_feature_list)
    # keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input, feature_columns['sparse'], return_feat_list=seq_feature_list)
    # deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_columns['sparse'])
    #
    # query_emb = concat_fun(query_emb_list)
    # keys_emb = concat_fun(keys_emb_list)
    # deep_input_emb = concat_fun(deep_input_emb_list)

    features = build_input_features(dnn_feature_columns)

    user_behavior_length = Input(shape=(1,), name='seq_length')

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    neg_history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    neg_history_fc_names = list(map(lambda x: "neg_" + x, history_fc_names))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        elif feature_name in neg_history_fc_names:
            neg_history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, init_std, seed, prefix="",
                                             seq_mask_zero=False)

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                      return_feat_list=history_feature_list,to_list=True)

    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns,
                                     return_feat_list=history_fc_names,to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list,to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,to_list=True)
    dnn_input_emb_list += sequence_embed_list


    keys_emb = concat_func(keys_emb_list)
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list)



    if use_negsampling:

        neg_uiseq_embed_list = embedding_lookup(embedding_dict, features, neg_history_feature_columns,
                                                neg_history_fc_names,to_list=True)

        neg_concat_behavior = concat_func(neg_uiseq_embed_list)

    else:
        neg_concat_behavior = None
    hist, aux_loss_1 = interest_evolution(keys_emb, query_emb, user_behavior_length, gru_type=gru_type,
                                          use_neg=use_negsampling, neg_concat_behavior=neg_concat_behavior,
                                          att_hidden_size=att_hidden_units,
                                          att_activation=att_activation,
                                          att_weight_normalization=att_weight_normalization, )

    deep_input_emb = Concatenate()([deep_input_emb, hist])

    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)

    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, use_bn, seed)(dnn_input)
    final_logit = Dense(1, use_bias=False)(output)
    output = PredictionLayer(task)(final_logit)

    #model_input_list = get_inputs_list(
    #    [sparse_input, dense_input, user_behavior_input])
    model_input_list = inputs_list

    #if use_negsampling:
    #    model_input_list += list(neg_user_behavior_input.values())

    model_input_list += [user_behavior_length]

    model = tf.keras.models.Model(inputs=model_input_list, outputs=output)

    if use_negsampling:
        model.add_loss(alpha * aux_loss_1)
    try:
        tf.keras.backend.get_session().run(tf.global_variables_initializer())
    except:
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
    return model
