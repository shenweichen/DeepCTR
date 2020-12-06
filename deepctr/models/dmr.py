# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.keras.layers import Lambda,Dot,Multiply
from tensorflow.python.keras import backend

from ..feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from ..inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list
from ..layers.core import DNN, PredictionLayer,SampledSoftmaxLayer,EmbeddingIndex,PoolingLayer
from ..layers.sequence import AttentionSequencePoolingLayer,PositionalEncoding,SequencePoolingLayer
from ..layers.utils import concat_func, NoMask, combined_dnn_input


def User2ItemNetwork(query, keys, user_behavior_length, deep_match_id, features, sparse_feature_columns, att_hidden_size,
                     att_activation, padding_first, att_weight_normalization, l2_reg_embedding, seed):
    dm_item_id = list(filter(lambda x: x.name == deep_match_id, sparse_feature_columns))
    dm_item_id_input = features[deep_match_id]
    dm_hist_item_id_input = features['hist_'+deep_match_id]
    dm_iid_embedding_table = create_embedding_matrix(dm_item_id, l2_reg_embedding, seed, prefix="deep_match_")[
        deep_match_id]
    dm_iid_embedding = dm_iid_embedding_table(dm_item_id_input)

    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation, padding_first=padding_first, causality=True,
                                         weight_normalization=att_weight_normalization, self_attention=True,
                                         supports_masking=False)([query, keys, user_behavior_length])

    hist = tf.keras.layers.Dense(dm_iid_embedding.get_shape().as_list()[-1], name='dm_align_2')(hist)

    user_embedding = tf.keras.layers.PReLU()(Lambda(lambda x: x[:, -1, :])(hist))
    dm_embedding = tf.keras.layers.PReLU()(Lambda(lambda x: x[:, -2, :])(hist))

    rel_u2i = Dot(axes=-1)([dm_iid_embedding, user_embedding])

    item_index = EmbeddingIndex(list(range(dm_item_id[0].vocabulary_size)))(dm_item_id[0])
    item_embedding_matrix = dm_iid_embedding_table
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))
    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    aux_loss = SampledSoftmaxLayer()([pooling_item_embedding_weight, dm_embedding,
                                      tf.cast(tf.reshape(dm_hist_item_id_input[:, -1], [-1, 1]), tf.int64), ])
    return rel_u2i,aux_loss

def Item2ItemNetwork(query, keys, user_behavior_length, att_hidden_size, att_activation, padding_first):
    scores = AttentionSequencePoolingLayer(att_hidden_size, att_activation, padding_first=padding_first,
                                           weight_normalization=False, supports_masking=False, return_score=True)([
        query, keys, user_behavior_length])
    scores_norm = tf.keras.layers.Activation('softmax')(scores)

    att_sum = Lambda(lambda x: backend.batch_dot(x[0], x[1]))([scores_norm, keys])
    rel_i2i = Lambda(lambda z: backend.sum(z, axis=-1, keepdims=False))(scores)
    return att_sum,rel_i2i

def DMR(dnn_feature_columns, history_feature_list, deep_match_id, dnn_use_bn=False,padding_first=True,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="sigmoid",
        att_weight_normalization=True, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
        task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param deep_match_id: str. An id that appears in the history_feature_list will be use to deep match.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param padding_first: bool. Is padding at the beginning of the sequence
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    if not padding_first:
        raise ValueError("Now DMR only support padding first, "
                         "input history sequence should be like [0,0,1,2,3](0 is the padding).")
    if deep_match_id not in history_feature_list:
        raise ValueError("deep_match_id must appear in the history_feature_list.")

    features = build_input_features(dnn_feature_columns)

    user_behavior_length = features["seq_length"]

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="")

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                      history_feature_list, to_list=True)
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)

    dnn_input_emb_list += sequence_embed_list
    deep_input_emb = concat_func(dnn_input_emb_list)

    keys_emb = concat_func(keys_emb_list, mask=True)
    query_emb = concat_func(query_emb_list, mask=True)

    keys_emb_pos = PositionalEncoding(use_sinusoidal=False, zero_pad=True, scale=False, use_concat=True)(keys_emb)
    keys_emb_pos = tf.keras.layers.Dense(keys_emb.get_shape().as_list()[-1], name='dm_align_1')(keys_emb_pos)
    keys_emb_pos = tf.keras.layers.PReLU()(keys_emb_pos)

    rel_u2i, aux_loss = User2ItemNetwork(keys_emb_pos,keys_emb,user_behavior_length,deep_match_id,features,sparse_feature_columns,
                                         att_hidden_size,att_activation,padding_first,att_weight_normalization,
                                         l2_reg_embedding,seed)

    att_sum, rel_i2i = Item2ItemNetwork(query_emb,keys_emb,user_behavior_length,att_hidden_size,att_activation,padding_first)

    hist_embedding = SequencePoolingLayer(mode='sum', padding_first=True)([keys_emb, user_behavior_length])

    sim_u2i = Multiply()([query_emb, hist_embedding])

    deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb),NoMask()(sim_u2i),hist_embedding,att_sum])
    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
    dnn_input = combined_dnn_input([deep_input_emb,rel_u2i,rel_i2i], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    final_logit = tf.keras.layers.Dense(1, use_bias=False,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    model.add_loss(aux_loss)
    return model
