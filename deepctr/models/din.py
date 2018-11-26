# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Deep Interest Network for Click-Through Rate Prediction (https://arxiv.org/pdf/1706.06978.pdf)
"""

from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal, TruncatedNormal
from tensorflow.python.keras.regularizers import l2

from ..layers import MLP
from ..sequence import SequencePoolingLayer, AttentionSequencePoolingLayer
from ..activations import Dice


def get_input(feature_dim_dict, seq_feature_list, seq_max_len):
    sparse_input = {feat: Input(shape=(1,), name='sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])}

    user_behavior_input = {feat: Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat) for i, feat in
                           enumerate(seq_feature_list)}

    user_behavior_length = Input(shape=(1,), name='seq_length')

    return sparse_input, user_behavior_input, user_behavior_length


def DIN(feature_dim_dict, seq_feature_list, embedding_size=4, hist_len_max=16,
        use_din=True, use_bn=False, hidden_size=[200, 80], activation=Dice, att_hidden_size=[80, 40], att_activation='sigmoid', att_weight_normalization=True,
        l2_reg_deep=5e-5, l2_reg_embedding=0, final_activation='sigmoid', keep_prob=1, init_std=0.0001, seed=1024, ):
    """Instantiates the Deep Interest Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
    :param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param use_din: bool, whether use din pooling or not.If set to ``False``,use **sum pooling**
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """
    for feature_dim_dict in [feature_dim_dict]:
        if not isinstance(feature_dim_dict,
                          dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
            raise ValueError(
                "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")
    if len(feature_dim_dict['dense']) > 0:
        raise ValueError('Now DIN only support sparse input')
    sparse_input, user_behavior_input, user_behavior_length = get_input(
        feature_dim_dict, seq_feature_list, hist_len_max)
    sparse_embedding_dict = {feat: Embedding(feature_dim_dict["sparse"][feat], embedding_size,
                                             embeddings_initializer=RandomNormal(
                                                 mean=0.0, stddev=init_std, seed=seed),
                                             embeddings_regularizer=l2(
                                                 l2_reg_embedding),
                                             name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}
    query_emb_list = [sparse_embedding_dict[feat](
        sparse_input[feat]) for feat in seq_feature_list]
    keys_emb_list = [sparse_embedding_dict[feat](
        user_behavior_input[feat]) for feat in seq_feature_list]
    deep_input_emb_list = [sparse_embedding_dict[feat](
        sparse_input[feat]) for feat in feature_dim_dict["sparse"]]

    query_emb = Concatenate()(query_emb_list) if len(
        query_emb_list) > 1 else query_emb_list[0]
    keys_emb = Concatenate()(keys_emb_list) if len(
        keys_emb_list) > 1 else keys_emb_list[0]
    deep_input_emb = Concatenate()(deep_input_emb_list) if len(
        deep_input_emb_list) > 1 else deep_input_emb_list[0]

    if use_din:
        hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation, weight_normalization=att_weight_normalization)([
            query_emb, keys_emb, user_behavior_length])
    else:
        hist = SequencePoolingLayer(hist_len_max, 'sum')(
            [keys_emb, user_behavior_length])

    deep_input_emb = Concatenate()([deep_input_emb, hist])
    output = MLP(hidden_size, activation, l2_reg_deep,
                 keep_prob, use_bn, seed)(deep_input_emb)
    output = Dense(1, final_activation)(output)
    output = Reshape([1])(output)
    model_input_list = list(sparse_input.values(
    ))+list(user_behavior_input.values()) + [user_behavior_length]

    model = Model(inputs=model_input_list, outputs=output)
    return model
