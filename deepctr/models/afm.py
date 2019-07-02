# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
    (https://arxiv.org/abs/1708.04617)

"""
import tensorflow as tf

from ..inputs import input_from_feature_columns, get_linear_logit,build_input_features
from ..layers.core import PredictionLayer
from ..layers.interaction import AFMLayer, FM
from ..layers.utils import concat_fun

def AFM(linear_feature_columns, dnn_feature_columns, embedding_size=8, use_attention=True, attention_factor=8,
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0, init_std=0.0001, seed=1024,
        task='binary'):
    """Instantiates the Attentional Factorization Machine architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_attention: bool,whether use attention or not,if set to ``False``.it is the same as **standard Factorization Machine**
    :param attention_factor: positive integer,units in attention net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_att: float. L2 regularizer strength applied to attention net
    :param afm_dropout: float in [0,1), Fraction of the attention net output units to dropout.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """


    features = build_input_features(linear_feature_columns+dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features,dnn_feature_columns,embedding_size,
                                                                                               l2_reg_embedding, init_std,
                                                                                               seed)
    # todo not support dense

    linear_logit = get_linear_logit(features, linear_feature_columns, l2_reg=l2_reg_linear, init_std=init_std,
                                    seed=seed, prefix='linear')

    fm_input = concat_fun(sparse_embedding_list, axis=1)
    if use_attention:
        fm_logit = AFMLayer(attention_factor, l2_reg_att, afm_dropout,
        seed)(sparse_embedding_list,)
    else:
        fm_logit = FM()(fm_input)

    final_logit = tf.keras.layers.add([linear_logit, fm_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
