# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com
Reference:
    [1] Lu W, Yu Y, Chang Y, et al. A Dual Input-aware Factorization Machine for CTR Prediction[C]
    //IJCAI. 2020: 3139-3145.(https://www.ijcai.org/Proceedings/2020/0434.pdf)
"""

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns, SparseFeat, \
    VarLenSparseFeat
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM, InteractingLayer
from ..layers.utils import concat_func, add_func, combined_dnn_input


def DIFM(linear_feature_columns, dnn_feature_columns,
         att_embedding_size=8, att_head_num=8, att_res=True, dnn_hidden_units=(128, 128),
         l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
         dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DIFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_embedding_size: integer, the embedding size in multi-head self-attention network.
    :param att_head_num: int. The head number in multi-head  self-attention network.
    :param att_res: bool. Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if not len(dnn_hidden_units) > 0:
        raise ValueError("dnn_hidden_units is null!")

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat),
                                      dnn_feature_columns)))
    inputs_list = list(features.values())

    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns,
                                                          l2_reg_embedding, seed)

    if not len(sparse_embedding_list) > 0:
        raise ValueError("there are no sparse features")

    att_input = concat_func(sparse_embedding_list, axis=1)
    att_out = InteractingLayer(att_embedding_size, att_head_num, att_res, scaling=True)(att_input)
    att_out = tf.keras.layers.Flatten()(att_out)
    m_vec = tf.keras.layers.Dense(
        sparse_feat_num, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(att_out)

    dnn_input = combined_dnn_input(sparse_embedding_list, [])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    m_bit = tf.keras.layers.Dense(
        sparse_feat_num, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    input_aware_factor = add_func([m_vec, m_bit])  # the complete input-aware factor m_x

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear, sparse_feat_refine_weight=input_aware_factor)

    fm_input = concat_func(sparse_embedding_list, axis=1)
    refined_fm_input = tf.keras.layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))(
        [fm_input, input_aware_factor])
    fm_logit = FM()(refined_fm_input)

    final_logit = add_func([linear_logit, fm_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
