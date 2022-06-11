# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Liu B, Tang R, Chen Y, et al. Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1904.04447, 2019.
    (https://arxiv.org/pdf/1904.04447)

"""
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda, Flatten, Concatenate

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import InnerProductLayer, FGCNNLayer
from ..layers.utils import concat_func, add_func


def unstack(input_tensor):
    input_ = tf.expand_dims(input_tensor, axis=2)
    return tf.unstack(input_, input_.shape[1], 1)


def FGCNN(linear_feature_columns, dnn_feature_columns, conv_kernel_width=(7, 7, 7, 7), conv_filters=(14, 16, 18, 20),
          new_maps=(3, 3, 3, 3),
          pooling_width=(2, 2, 2, 2), dnn_hidden_units=(256, 128, 64), l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
          l2_reg_dnn=0,
          dnn_dropout=0,
          seed=1024,
          task='binary', ):
    """Instantiates the Feature Generation by Convolutional Neural Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param new_maps: list, list of positive integer or empty list, the feature maps of generated features.
    :param pooling_width: list, list of positive integer or empty list,the width of pooling layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if not (len(conv_kernel_width) == len(conv_filters) == len(new_maps) == len(pooling_width)):
        raise ValueError(
            "conv_kernel_width,conv_filters,new_maps  and pooling_width must have same length")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    deep_emb_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)
    fg_deep_emb_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed,
                                                     prefix='fg')

    fg_input = concat_func(fg_deep_emb_list, axis=1)
    origin_input = concat_func(deep_emb_list, axis=1)

    if len(conv_filters) > 0:
        new_features = FGCNNLayer(
            conv_filters, conv_kernel_width, new_maps, pooling_width)(fg_input)
        combined_input = concat_func([origin_input, new_features], axis=1)
    else:
        combined_input = origin_input
    inner_product = Flatten()(
        InnerProductLayer()(Lambda(unstack, mask=[None] * int(combined_input.shape[1]))(combined_input)))
    linear_signal = Flatten()(combined_input)
    dnn_input = Concatenate()([linear_signal, inner_product])
    dnn_input = Flatten()(dnn_input)

    final_logit = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout)(dnn_input)
    final_logit = Dense(1, use_bias=False)(final_logit)

    final_logit = add_func([final_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)

    model = Model(inputs=inputs_list, outputs=output)
    return model
