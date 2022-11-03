# -*- coding:utf-8 -*-
"""
Author:
    Yi He, heyi_jack@163.com

Reference:
    [1] Chen, B., Wang, Y., Liu, et al. Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models. CIKM, 2021, October (https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)
"""
from tensorflow.python.keras.layers import Dense, Reshape, Concatenate
from tensorflow.python.keras.models import Model

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN, RegulationModule
from ..layers.interaction import CrossNet, BridgeModule
from ..layers.utils import add_func, concat_func


def EDCN(linear_feature_columns,
         dnn_feature_columns,
         cross_num=2,
         cross_parameterization='vector',
         bridge_type='concatenation',
         tau=1.0,
         l2_reg_linear=1e-5,
         l2_reg_embedding=1e-5,
         l2_reg_cross=1e-5,
         l2_reg_dnn=0,
         seed=1024,
         dnn_dropout=0,
         dnn_use_bn=False,
         dnn_activation='relu',
         task='binary'):
    """Instantiates the Enhanced Deep&Cross Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param bridge_type: The type of bridge interaction, one of ``"pointwise_addition"``, ``"hadamard_product"``, ``"concatenation"`` , ``"attention_pooling"``
    :param tau: Positive float, the temperature coefficient to control distribution of field-wise gating unit
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

    print('EDCN brige type: ', bridge_type)

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear', l2_reg=l2_reg_linear)

    sparse_embedding_list, _ = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding, seed, support_dense=False)

    emb_input = concat_func(sparse_embedding_list, axis=1)
    deep_in = RegulationModule(tau)(emb_input)
    cross_in = RegulationModule(tau)(emb_input)

    field_size = len(sparse_embedding_list)
    embedding_size = int(sparse_embedding_list[0].shape[-1])
    cross_dim = field_size * embedding_size

    for i in range(cross_num):
        cross_out = CrossNet(1, parameterization=cross_parameterization,
                             l2_reg=l2_reg_cross)(cross_in)
        deep_out = DNN([cross_dim], dnn_activation, l2_reg_dnn,
                       dnn_dropout, dnn_use_bn, seed=seed)(deep_in)
        print(cross_out, deep_out)
        bridge_out = BridgeModule(bridge_type)([cross_out, deep_out])
        if i + 1 < cross_num:
            bridge_out_list = Reshape([field_size, embedding_size])(bridge_out)
            deep_in = RegulationModule(tau)(bridge_out_list)
            cross_in = RegulationModule(tau)(bridge_out_list)

    stack_out = Concatenate()([cross_out, deep_out, bridge_out])
    final_logit = Dense(1, use_bias=False)(stack_out)

    final_logit = add_func([final_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)

    return model
