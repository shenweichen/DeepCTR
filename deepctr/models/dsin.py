# coding: utf-8
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Feng Y, Lv F, Shen W, et al. Deep Session Interest Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.06482, 2019.(https://arxiv.org/abs/1905.06482)

"""

from collections import OrderedDict

from tensorflow.python.keras.layers import (Concatenate, Dense, Embedding,
                                            Flatten, Input)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from ..feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from ..inputs import (get_embedding_vec_list, get_inputs_list, embedding_lookup, get_dense_input)
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import (AttentionSequencePoolingLayer, BiasEncoding,
                               BiLSTM, Transformer)
from ..layers.utils import concat_func, combined_dnn_input


def DSIN(dnn_feature_columns, sess_feature_list, sess_max_count=5, bias_encoding=False,
         att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='sigmoid', dnn_dropout=0,
         dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, seed=1024, task='binary',
         ):
    """Instantiates the Deep Session Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param sess_feature_list: list,to indicate  sequence sparse field
    :param sess_max_count: positive int, to indicate the max number of sessions
    :param sess_len_max: positive int, to indicate the max length of each session
    :param bias_encoding: bool. Whether use bias encoding or postional encoding
    :param att_embedding_size: positive int, the embedding size of each attention head
    :param att_head_num: positive int, the number of attention head
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """

    hist_emb_size = sum(
        map(lambda fc: fc.embedding_dim, filter(lambda fc: fc.name in sess_feature_list, dnn_feature_columns)))

    if (att_embedding_size * att_head_num != hist_emb_size):
        raise ValueError(
            "hist_emb_size must equal to att_embedding_size * att_head_num ,got %d != %d *%d" % (
                hist_emb_size, att_embedding_size, att_head_num))

    features = build_input_features(dnn_feature_columns)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "sess" + x, sess_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            continue
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    user_behavior_input_dict = {}
    for idx in range(sess_max_count):
        sess_input = OrderedDict()
        for i, feat in enumerate(sess_feature_list):
            sess_input[feat] = features["sess_" + str(idx) + "_" + feat]

        user_behavior_input_dict["sess_" + str(idx)] = sess_input

    user_sess_length = Input(shape=(1,), name='sess_length')

    embedding_dict = {feat.embedding_name: Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                     embeddings_initializer=feat.embeddings_initializer,
                                                     embeddings_regularizer=l2(
                                                         l2_reg_embedding),
                                                     name='sparse_emb_' +
                                                          str(i) + '-' + feat.name,
                                                     mask_zero=(feat.name in sess_feature_list)) for i, feat in
                      enumerate(sparse_feature_columns)}

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, sess_feature_list,
                                      sess_feature_list, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=sess_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    query_emb = concat_func(query_emb_list, mask=True)

    dnn_input_emb = Flatten()(concat_func(dnn_input_emb_list))

    tr_input = sess_interest_division(embedding_dict, user_behavior_input_dict, sparse_feature_columns,
                                      sess_feature_list, sess_max_count, bias_encoding=bias_encoding)

    Self_Attention = Transformer(att_embedding_size, att_head_num, dropout_rate=0, use_layer_norm=False,
                                 use_positional_encoding=(not bias_encoding), seed=seed, supports_masking=True,
                                 blinding=True)
    sess_fea = sess_interest_extractor(
        tr_input, sess_max_count, Self_Attention)

    interest_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                             supports_masking=False)(
        [query_emb, sess_fea, user_sess_length])

    lstm_outputs = BiLSTM(hist_emb_size,
                          layers=2, res_layers=0, dropout_rate=0.2, )(sess_fea)
    lstm_attention_layer = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True)(
        [query_emb, lstm_outputs, user_sess_length])

    dnn_input_emb = Concatenate()(
        [dnn_input_emb, Flatten()(interest_attention_layer), Flatten()(lstm_attention_layer)])

    dnn_input_emb = combined_dnn_input([dnn_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(dnn_input_emb)
    output = Dense(1, use_bias=False, activation=None)(output)
    output = PredictionLayer(task)(output)

    sess_input_list = []
    # sess_input_length_list = []
    for i in range(sess_max_count):
        sess_name = "sess_" + str(i)
        sess_input_list.extend(get_inputs_list(
            [user_behavior_input_dict[sess_name]]))
        # sess_input_length_list.append(user_behavior_length_dict[sess_name])

    model = Model(inputs=inputs_list + [user_sess_length], outputs=output)

    return model


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
        keys_emb = concat_func(keys_emb_list, mask=True)
        tr_input.append(keys_emb)
    if bias_encoding:
        tr_input = BiasEncoding(sess_max_count)(tr_input)
    return tr_input


def sess_interest_extractor(tr_input, sess_max_count, TR):
    tr_out = []
    for i in range(sess_max_count):
        tr_out.append(TR(
            [tr_input[i], tr_input[i]]))
    sess_fea = concat_func(tr_out, axis=1)
    return sess_fea
