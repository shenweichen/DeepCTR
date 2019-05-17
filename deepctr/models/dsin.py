# coding: utf-8
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Feng Y, Lv F, Shen W, et al. Deep Session Interest Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.06482, 2019.(https://arxiv.org/abs/1905.06482)

"""

from collections import OrderedDict

from ..input_embedding import get_inputs_list, create_singlefeat_inputdict, get_embedding_vec_list
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer, BiLSTM, Transformer, BiasEncoding
from ..layers.utils import concat_fun, NoMask
from ..utils import check_feature_config_dict
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


def DSIN(feature_dim_dict, sess_feature_list, embedding_size=8, sess_max_count=5, sess_len_max=10,
         att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='sigmoid',
         l2_reg_dnn=0, l2_reg_embedding=1e-6, task='binary', dnn_dropout=0, init_std=0.0001, seed=1024,
         bias_encoding=False):
    check_feature_config_dict(feature_dim_dict)

    print('sess_count', sess_max_count, 'use_bias_encoding', bias_encoding, )

    sparse_input, dense_input, user_behavior_input_dict, _, user_sess_length = get_input(
        feature_dim_dict, sess_feature_list, sess_max_count, sess_len_max)

    sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
                                                  embeddings_initializer=RandomNormal(
                                                      mean=0.0, stddev=init_std, seed=seed),
                                                  embeddings_regularizer=l2(
                                                      l2_reg_embedding),
                                                  name='sparse_emb_' + str(i) + '-' + feat.name,
                                                  mask_zero=(feat.name in sess_feature_list)) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}

    query_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict["sparse"],
                                            sess_feature_list, sess_feature_list)

    query_emb = concat_fun(query_emb_list)

    deep_input_emb_list = get_embedding_vec_list(sparse_embedding_dict, sparse_input, feature_dim_dict["sparse"],
                                                 mask_feat_list=sess_feature_list)
    deep_input_emb = concat_fun(deep_input_emb_list)
    deep_input_emb = Flatten()(NoMask()(deep_input_emb))

    tr_input = sess_interest_division(sparse_embedding_dict, user_behavior_input_dict, feature_dim_dict['sparse'],
                                      sess_feature_list, sess_max_count, bias_encoding=bias_encoding)

    Self_Attention = Transformer(att_embedding_size, att_head_num, dropout_rate=0, use_layer_norm=False,
                                 use_positional_encoding=(not bias_encoding), seed=seed, supports_masking=True,
                                 blinding=True)
    sess_fea = sess_interest_extractor(tr_input, sess_max_count, Self_Attention)

    interest_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                             supports_masking=False)(
        [query_emb, sess_fea, user_sess_length])

    lstm_outputs = BiLSTM(len(sess_feature_list) * embedding_size, layers=2, res_layers=0, dropout_rate=0.2, )(sess_fea)
    lstm_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True)(
        [query_emb, lstm_outputs, user_sess_length])

    deep_input_emb = Concatenate()(
        [deep_input_emb, Flatten()(interest_attention_layer), Flatten()(lstm_attention_layer)])
    if len(dense_input) > 0:
        deep_input_emb = Concatenate()([deep_input_emb] + list(dense_input.values()))

    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(deep_input_emb)
    output = Dense(1, use_bias=False, activation=None)(output)
    output = PredictionLayer(task)(output)

    sess_input_list = []
    # sess_input_length_list = []
    for i in range(sess_max_count):
        sess_name = "sess_" + str(i)
        sess_input_list.extend(get_inputs_list([user_behavior_input_dict[sess_name]]))
        # sess_input_length_list.append(user_behavior_length_dict[sess_name])

    model_input_list = get_inputs_list([sparse_input, dense_input]) + sess_input_list + [
        user_sess_length]

    model = Model(inputs=model_input_list, outputs=output)

    return model


def get_input(feature_dim_dict, seq_feature_list, sess_max_count, seq_max_len):
    sparse_input, dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = {}
    for idx in range(sess_max_count):
        sess_input = OrderedDict()
        for i, feat in enumerate(seq_feature_list):
            sess_input[feat] = Input(shape=(seq_max_len,), name='seq_' + str(idx) + str(i) + '-' + feat)

        user_behavior_input["sess_" + str(idx)] = sess_input

    user_behavior_length = {"sess_" + str(idx): Input(shape=(1,), name='seq_length' + str(idx)) for idx in
                            range(sess_max_count)}
    user_sess_length = Input(shape=(1,), name='sess_length')

    return sparse_input, dense_input, user_behavior_input, user_behavior_length, user_sess_length


def sess_interest_division(sparse_embedding_dict, user_behavior_input_dict, sparse_fg_list, sess_feture_list,
                           sess_max_count,
                           bias_encoding=True):
    tr_input = []
    for i in range(sess_max_count):
        sess_name = "sess_" + str(i)
        keys_emb_list = get_embedding_vec_list(sparse_embedding_dict, user_behavior_input_dict[sess_name],
                                               sparse_fg_list, sess_feture_list, sess_feture_list)
        # [sparse_embedding_dict[feat](user_behavior_input_dict[sess_name][feat]) for feat in
        #             sess_feture_list]
        keys_emb = concat_fun(keys_emb_list)
        tr_input.append(keys_emb)
    if bias_encoding:
        tr_input = BiasEncoding(sess_max_count)(tr_input)
    return tr_input


def sess_interest_extractor(tr_input, sess_max_count, TR):
    tr_out = []
    for i in range(sess_max_count):
        tr_out.append(TR(
            [tr_input[i], tr_input[i]]))
    sess_fea = concat_fun(tr_out, axis=1)
    return sess_fea
