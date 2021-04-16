# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com
Reference:
    [1] Yu Y, Wang Z, Yuan B. An Input-aware Factorization Machine for Sparse Prediction[C]//IJCAI. 2019: 1466-1472.(https://www.ijcai.org/Proceedings/2019/0203.pdf)
"""

from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from tensorflow.python.keras.layers import Multiply


def IFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the IFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
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
    print('sparse_feat_num',sparse_feat_num)
    inputs_list = list(features.values())


    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    # m'_{x}
    dnn_output = tf.keras.layers.Dense(
        sparse_feat_num, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
    print('dnn_output',dnn_output)

    # input_aware_factor m_{x,i}
    input_aware_factor = sparse_feat_num * tf.nn.softmax(dnn_output, axis=1)
    # input_aware_factor = tf.constant(sparse_feat_num, dtype=tf.float32) * tf.nn.softmax(dnn_output, axis=1)
    input_aware_factor = tf.keras.layers.Lambda(lambda x: sparse_feat_num * tf.nn.softmax(x, axis=1))(dnn_output)  # sparse_feat_num要写在lambda表达式里面
    print('input_aware_factor',input_aware_factor)

    # sparse_feat_num = tf.keras.layers.Lambda(lambda x: tf.constant(sparse_feat_num, dtype=tf.float32))(dnn_output)
    # sparse_feat_num= tf.repeat(sparse_feat_num,input_aware_factor.shape(0),axis=0)
    # input_aware_factor = tf.keras.layers.Multiply()([sparse_feat_num, input_aware_factor])

    # print('tf.convert_to_tensor(sparse_feat_num)',tf.convert_to_tensor(sparse_feat_num))
    # input_aware_factor = Multiply()([tf.convert_to_tensor(sparse_feat_num), tf.nn.softmax(dnn_output, axis=1)])

    # print('input_aware_factor',input_aware_factor.shape)

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear, sparse_feat_refine_weight=input_aware_factor)

    print('group_embedding_dict',group_embedding_dict)
    fm_group_result = []
    for k, v in group_embedding_dict.items():
        if k in fm_group:
            fm_input = concat_func(v, axis=1)
            print('fm_input',fm_input.shape)
            # print('tf.expand_dims(input_aware_factor, axis=-1)',tf.expand_dims(input_aware_factor, axis=-1).shape)

            # 全都得是层的操作  不能直接用*  不能直接用tf.
            refined_fm_input = Multiply()([fm_input, tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_aware_factor)])
            # refined_fm_input = multiply([fm_input, tf.expand_dims(input_aware_factor, axis=-1)])
            # refined_fm_input = fm_input #* tf.expand_dims(input_aware_factor, axis=-1)
            # refined_fm_input = fm_input * tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_aware_factor)
            # refined_fm_input = fm_input * tf.expand_dims(input_aware_factor, axis=-1)  # \textbf{v}_{x,i}=m_{x,i}\textbf{v}_i

            # refined_fm_input = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([fm_input, tf.expand_dims(input_aware_factor, axis=-1)])
            # refined_fm_input = tf.math.multiply(fm_input, tf.expand_dims(input_aware_factor, axis=-1))
            print('refined_fm_input',refined_fm_input.shape)
            fm_group_result.append(FM()(refined_fm_input))
    fm_logit = add_func(fm_group_result)
    print('fm_logit0',fm_logit)


    # fm_logit = add_func([FM()(concat_func(v, axis=1) * tf.expand_dims(input_aware_factor, axis=-1))  #为啥乘上iaf就不能作为output了 形状也一样
    #                      for k, v in group_embedding_dict.items() if k in fm_group])
    # print('fm_logit',fm_logit)
    # input('c')


    print('linear_logit',linear_logit)
    final_logit = add_func([linear_logit, fm_logit])

    print('final_logit',final_logit)
    output = PredictionLayer(task)(final_logit)
    print('output',output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    # input('c')
    return model
