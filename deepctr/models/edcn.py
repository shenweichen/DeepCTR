# -*- coding:utf-8 -*-
"""
Author:
    Yi He, heyi_jack@163.com

Reference:
    [1] Chen, B., Wang, Y., Liu, et al. Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models. CIKM, 2021, October (https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)
"""
import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import CrossNet, ConcatenationBridge, AttentionPoolingLayer
from ..layers.utils import add_func, concat_func
from ..layers.activation import RegulationLayer

def EDCN(linear_feature_columns,
         dnn_feature_columns,
         bridge_type='attention_pooling',
         tau=0.1, 
         use_dense_features=True,
         cross_num=2,
         cross_parameterization='vector',
         l2_reg_linear=1e-5,
         l2_reg_embedding=1e-5,
         l2_reg_cross=1e-5,
         l2_reg_dnn=0,
         seed=10000,
         dnn_dropout=0,
         dnn_use_bn=False,
         dnn_activation='relu',
         task='binary'):
    """Instantiates the Enhanced Deep&Cross Network architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bridge_type: The type of bridge interaction, one of 'pointwise_addition', 'hadamard_product', 'concatenation', 'attention_pooling'
    :param tau: Positive float, the temperature coefficient to control distribution of field-wise gating unit
    :param use_dense_features: Whether to use dense features, if True, dense feature will be projected to sparse embedding space
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """
    if cross_num == 0:
        raise ValueError("Cross layer num must > 0")

    if bridge_type == 'pointwise_addition':
        BridgeLayer = tf.keras.layers.Add
    elif bridge_type == 'hadamard_product':
        BridgeLayer = tf.keras.layers.Multiply
    elif bridge_type == 'concatenation':
        BridgeLayer = ConcatenationBridge
    elif bridge_type == 'attention_pooling':
        BridgeLayer = AttentionPoolingLayer
    else:
        raise NotImplementedError

    print('EDCN brige type: ', bridge_type)

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features,
                                    linear_feature_columns,
                                    seed=seed,
                                    prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding, seed)

    # project dense value to sparse embedding space, generate a new field feature
    if use_dense_features:
        sparse_embedding_dim = sparse_embedding_list[0].shape[-1]
        dense_value_feild = concat_func(dense_value_list)
        dense_value_feild = DNN([sparse_embedding_dim], dnn_activation,
                                l2_reg_dnn, dnn_dropout,
                                dnn_use_bn)(dense_value_feild)
        dense_value_feild = tf.expand_dims(dense_value_feild, axis=1)
        sparse_embedding_list.append(dense_value_feild)

    deep_in = sparse_embedding_list
    cross_in = sparse_embedding_list
    field_size = len(sparse_embedding_list)
    cross_dim = field_size * cross_in[0].shape[-1]

    for i in range(cross_num):

        deep_in = RegulationLayer(tau)(deep_in)
        cross_in = RegulationLayer(tau)(cross_in)
        cross_out = CrossNet(1, parameterization=cross_parameterization,
                             l2_reg=l2_reg_cross)(deep_in)
        deep_out = DNN([cross_dim], dnn_activation, l2_reg_dnn,
                       dnn_dropout, dnn_use_bn, seed=seed)(cross_in)

        bridge_out = BridgeLayer()([cross_out, deep_out])
        bridge_out_list = tf.split(tf.expand_dims(bridge_out, axis=1), field_size, axis=-1)

        deep_in = bridge_out_list
        cross_in = bridge_out_list

    stack_out = tf.keras.layers.Concatenate()(
        [cross_out, deep_out, bridge_out])
    final_logit = tf.keras.layers.Dense(1, use_bias=False,
        kernel_initializer=tf.keras.initializers.glorot_normal(seed))(stack_out)

    final_logit = add_func([final_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
