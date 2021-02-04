# -*- coding:utf-8 -*-
"""
Author:
    Zichao Li, 2843656167@qq.com

Reference:
    Xiao Z, Yang L, Jiang W, et al. Deep Multi-Interest Network for Click-through Rate Prediction[C]//Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2020: 2265-2268.
"""

import tensorflow as tf
from tensorflow.python.keras.layers import (Concatenate, Dense, Permute, multiply, Input, Embedding)
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import RandomNormal
from ..feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from ..inputs import get_varlen_pooling_list, create_embedding_matrix, embedding_lookup, varlen_embedding_lookup, \
    get_dense_input
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import Transformer
from ..layers.utils import concat_func, reduce_mean, combined_dnn_input
from ..layers.normalization import LayerNormalization


def AttentionUnit(query_emb, mha_output_item, position_hist_emb, mask, max_len):
    '''
    AttentionUnit is used for Multi-Interest Extractor Layer in DMIN.

    Note: total_emb_dim used below refers to the sum of embedding dim of all of user behavior features.
    For example: if behavior_feature_list=["item_id", "cate_id"] and the embedding dims of both is 8 and 4 respectively, then total_emb_dim=12.
    In original paper, behavior_feature_list=["item_id", "cate_id"] and the embedding dims of both are emb_dim, then total_emb_dim=2*emb_dim.

    query_emb:[bsz,total_emb_dim],
    mha2_output_item:[bsz,max_len,total_emb_dim]
    position:[bsz,max_len,position_emb_dim]
    mask:[bsz,1]
    '''

    mask = tf.sequence_mask(tf.squeeze(mask, axis=-1), max_len)
    mask = tf.equal(mask, tf.ones_like(mask))
    query_emb = tf.tile(query_emb, [1, tf.shape(mha_output_item)[1]])
    query_emb = tf.reshape(query_emb, tf.shape(mha_output_item))
    query_emb = concat_func([query_emb, position_hist_emb], axis=-1)
    query_emb = Dense(mha_output_item.get_shape().as_list()[-1])(query_emb)

    concat_output = concat_func([query_emb, mha_output_item, \
                               query_emb - mha_output_item, query_emb * mha_output_item], axis=-1)
    concat_output = Dense(80, activation="sigmoid")(concat_output)
    concat_output = Dense(40, activation="sigmoid")(concat_output)
    concat_output = Dense(1)(concat_output)
    concat_output = tf.reshape(concat_output, [-1, 1, tf.shape(mha_output_item)[1]])
    scores = concat_output

    key_masks = tf.expand_dims(mask, 1)
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)
    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    scores = tf.nn.softmax(scores)
    output = tf.matmul(scores, mha_output_item)
    return output, scores, scores_no_softmax


def auxiliary_loss(h_states, click_seq, noclick_seq, mask, stag=None):
    '''
    Different from DIEN, layer normalization is used for auxiliary loss in DMIN.

    :param h_states:[bsz,max_len-1,total_emb_dim]
    :param click_seq:[bsz,max_len-1,total_emb_dim]
    :param noclick_seq: #[bsz,max_len-1,total_emb_dim]
    :param mask:[bsz,1]
    :param stag:
    :return: scalar, auxiliary loss.
    '''
    hist_len, _ = click_seq.get_shape().as_list()[1:]
    mask = tf.sequence_mask(mask, hist_len)
    mask = mask[:, 0, :]

    mask = tf.cast(mask, tf.float32)

    click_input_ = tf.concat([h_states, click_seq], -1)

    noclick_input_ = tf.concat([h_states, noclick_seq], -1)

    layer_norm = LayerNormalization()
    auxiliary_nn = DNN([100, 50, 2], activation='softmax')

    click_input_ = layer_norm(click_input_)
    click_prop_ = auxiliary_nn(click_input_, stag=stag)[:, :, 0]

    noclick_input_ = layer_norm(noclick_input_)
    noclick_prop_ = auxiliary_nn(noclick_input_, stag=stag)[:, :, 0]  # [B,T-1]

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


def DMIN(dnn_feature_columns, history_feature_list, position_embedding_dim=2, att_head_num=4, use_negsampling=True,
         alpha=1.0, use_bn=False, dnn_hidden_units=(200, 80), dnn_activation='relu', l2_reg_dnn=0,
         l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, task='binary'):
    """Instantiates the Deep Multi-Interest Network architecture.

     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
     :param history_feature_list: list, to indicate sequence sparse field
     :param position_embedding_dim: int, the dimension of position encoding.
     :param att_head_num: int, the number of heads in multi-head self attention.
     :param use_negsampling: bool, whether or not use negtive sampling
     :param alpha: float ,weight of auxiliary_loss
     :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
     :param dnn_activation: Activation function to use in DNN
     :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
     :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
     :param seed: integer ,to use as random seed.
     :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
     :return: A Keras model instance.

     """

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    user_behavior_length = features["seq_length"]

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

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="",
                                             seq_mask_zero=False)

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                      return_feat_list=history_feature_list, to_list=True)
    hist_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns,
                                     return_feat_list=history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)

    if use_negsampling:
        neg_uiseq_embed_list = embedding_lookup(embedding_dict, features, neg_history_feature_columns,
                                                neg_history_fc_names, to_list=True)
        neg_concat_behavior = concat_func(neg_uiseq_embed_list)
    else:
        neg_concat_behavior = None

    dnn_input_emb_list += sequence_embed_list
    query_emb = tf.squeeze(concat_func(query_emb_list), axis=1)
    deep_input_emb = tf.squeeze(concat_func(dnn_input_emb_list), axis=1)
    hist_emb = concat_func(hist_emb_list)

    max_len = varlen_sparse_feature_columns[0].maxlen
    position_his_input = Input(shape=(max_len,), name='position_hist')
    position_hist_emb = Embedding(max_len, position_embedding_dim,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                  embeddings_regularizer=l2(l2_reg_embedding))(position_his_input)

    # Behavior Refiner Layer
    embedding_dim = varlen_sparse_feature_columns[0].embedding_dim
    total_embedding_dim = hist_emb.get_shape().as_list()[-1]
    mha1_att_embedding_size = hist_emb.get_shape().as_list()[-1] // att_head_num
    mha1 = Transformer(att_embedding_size=mha1_att_embedding_size, head_num=att_head_num, dropout_rate=dnn_dropout,
                       use_positional_encoding=False, use_res=True, use_feed_forward=True, use_layer_norm=True,
                       blinding=False, seed=seed, supports_masking=True)

    mask = tf.sequence_mask(tf.squeeze(user_behavior_length, axis=1), max_len)
    mha1_output = mha1([hist_emb, hist_emb], [mask, mask])

    if mha1_output.get_shape().as_list()[1] != hist_emb.get_shape().as_list()[1]:
        mha1_output = tf.tile(mha1_output, [1, hist_emb.get_shape().as_list()[1], 1])

    mha1_output_trans = Dense(embedding_dim * 4, activation='relu')(mha1_output)
    mha1_output_trans = Dense(total_embedding_dim)(mha1_output_trans)

    mha1_output = mha1_output_trans + mha1_output

    if use_negsampling:
        aux_loss = auxiliary_loss(mha1_output[:, :-1, :], hist_emb[:, 1:, :], neg_concat_behavior[:, 1:, :],
                                  user_behavior_length)
    else:
        aux_loss = 0.0

    # Multi-Interest Extractor Layer
    if mha1_output.get_shape().as_list()[1] != hist_emb.get_shape().as_list()[1]:
        mha1_output = tf.tile(mha1_output, [1, hist_emb.get_shape().as_list()[1], 1])

    mha2_input = Dense(total_embedding_dim * att_head_num)(mha1_output)
    mha2_att_embedding_size = mha2_input.get_shape().as_list()[-1] // att_head_num
    mha2 = Transformer(att_embedding_size=mha2_att_embedding_size, head_num=att_head_num, dropout_rate=dnn_dropout,
                       use_positional_encoding=False, use_res=True, use_feed_forward=True, use_layer_norm=True,
                       blinding=False, seed=seed, supports_masking=True)
    mha2_output = mha2([mha2_input, mha2_input], [mask, mask])

    if mha2_output.get_shape().as_list()[1] != hist_emb.get_shape().as_list()[1]:
        mha2_output = tf.tile(mha2_output, [1, hist_emb.get_shape().as_list()[1], 1])

    mha2_output_list = tf.split(mha2_output, att_head_num, axis=-1)

    for idx, mha2_output_item in enumerate(mha2_output_list):
        mha2_output_item_trans = Dense(embedding_dim * 4, activation='relu')(mha2_output_item)
        mha2_output_item_trans = Dense(total_embedding_dim, activation='relu')(mha2_output_item_trans)
        mha2_output_item = mha2_output_item_trans + mha2_output_item
        attn_output, attn_score, attn_scores_no_softmax = AttentionUnit(query_emb, mha2_output_item, position_hist_emb,
                                                                        user_behavior_length, max_len)
        att_fea = tf.reduce_sum(attn_output, 1)
        deep_input_emb = Concatenate(axis=1)([deep_input_emb, att_fea])

    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)

    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(dnn_input)
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list + [position_his_input], outputs=output)

    if use_negsampling:
        model.add_loss(alpha * aux_loss)
    try:
        tf.keras.backend.get_session().run(tf.global_variables_initializer())
    except:
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
    return model
