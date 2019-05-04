# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Liu Q, Yu F, Wu S, et al. A convolutional click prediction model[C]//Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015: 1743-1746.
    (http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)

"""
import tensorflow as tf

from ..input_embedding import preprocess_input_embedding, get_linear_logit
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import KMaxPooling
from ..layers.utils import concat_fun
from ..utils import check_feature_config_dict


def CCPM(feature_dim_dict, embedding_size=8, conv_kernel_width=(6, 5), conv_filters=(4, 4), dnn_hidden_units=(256,),
         l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0, init_std=0.0001, seed=1024,
         task='binary', ):
    """Instantiates the Convolutional Click Prediction Model architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    check_feature_config_dict(feature_dim_dict)
    if len(conv_kernel_width) != len(conv_filters):
        raise ValueError(
            "conv_kernel_width must have same element with conv_filters")

    deep_emb_list, linear_emb_list, dense_input_dict, inputs_list = preprocess_input_embedding(feature_dim_dict,
                                                                                               embedding_size,
                                                                                               l2_reg_embedding,
                                                                                               l2_reg_linear, init_std,
                                                                                               seed,
                                                                                               create_linear_weight=True)

    linear_logit = get_linear_logit(
        linear_emb_list, dense_input_dict, l2_reg_linear)
    n = len(deep_emb_list)
    l = len(conv_filters)

    conv_input = concat_fun(deep_emb_list, axis=1)
    pooling_result = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, axis=3))(conv_input)

    for i in range(1, l + 1):
        filters = conv_filters[i - 1]
        width = conv_kernel_width[i - 1]
        k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3

        conv_result = tf.keras.layers.Conv2D(filters=filters, kernel_size=(width, 1), strides=(1, 1), padding='same',
                                             activation='tanh', use_bias=True, )(pooling_result)
        pooling_result = KMaxPooling(
            k=min(k, conv_result.shape[1].value), axis=1)(conv_result)

    flatten_result = tf.keras.layers.Flatten()(pooling_result)
    final_logit = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn,
                      dropout_rate=dnn_dropout)(flatten_result)
    final_logit = tf.keras.layers.Dense(1, use_bias=False)(final_logit)

    final_logit = tf.keras.layers.add([final_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
