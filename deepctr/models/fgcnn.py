# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Liu B, Tang R, Chen Y, et al. Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1904.04447, 2019.
    (https://arxiv.org/pdf/1904.04447)

"""
import tensorflow as tf

from ..input_embedding import create_singlefeat_inputdict, create_varlenfeat_inputdict, get_linear_logit
from ..input_embedding import get_inputs_embedding
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import InnerProductLayer, FGCNNLayer
from ..layers.utils import concat_fun
from ..utils import check_feature_config_dict


def preprocess_input_embedding(feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, init_std, seed,
                               return_linear_logit=True, ):
    sparse_input_dict, dense_input_dict = create_singlefeat_inputdict(
        feature_dim_dict)
    sequence_input_dict, sequence_input_len_dict, sequence_max_len_dict = create_varlenfeat_inputdict(
        feature_dim_dict)
    inputs_list, deep_emb_list, linear_emb_list = get_inputs_embedding(feature_dim_dict, embedding_size,
                                                                       l2_reg_embedding, l2_reg_linear, init_std, seed,
                                                                       sparse_input_dict, dense_input_dict,
                                                                       sequence_input_dict, sequence_input_len_dict,
                                                                       sequence_max_len_dict,
                                                                       return_linear_logit, prefix='')
    _, fg_deep_emb_list, _ = get_inputs_embedding(feature_dim_dict, embedding_size,
                                                  l2_reg_embedding, l2_reg_linear, init_std, seed,
                                                  sparse_input_dict, dense_input_dict,
                                                  sequence_input_dict, sequence_input_len_dict,
                                                  sequence_max_len_dict, False, prefix='fg')
    if return_linear_logit:
        linear_logit = get_linear_logit(
            linear_emb_list, dense_input_dict, l2_reg_linear)
    else:
        linear_logit = None
    return deep_emb_list, fg_deep_emb_list, linear_logit, inputs_list


def unstack(input_tensor):
    input_ = tf.expand_dims(input_tensor, axis=2)
    return tf.unstack(input_, input_.shape[1], 1)


def FGCNN(feature_dim_dict, embedding_size=8, conv_kernel_width=(7, 7, 7, 7), conv_filters=(14, 16, 18, 20),
          new_maps=(3, 3, 3, 3),
          pooling_width=(2, 2, 2, 2), dnn_hidden_units=(128,), l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0, init_std=0.0001, seed=1024,
          task='binary', ):
    """Instantiates the Feature Generation by Convolutional Neural Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param new_maps: list, list of positive integer or empty list, the feature maps of generated features.
    :param pooling_width: list, list of positive integer or empty list,the width of pooling layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    check_feature_config_dict(feature_dim_dict)
    if not (len(conv_kernel_width) == len(conv_filters) == len(new_maps) == len(pooling_width)):
        raise ValueError(
            "conv_kernel_width,conv_filters,new_maps  and pooling_width must have same length")

    deep_emb_list, fg_deep_emb_list, _, inputs_list = preprocess_input_embedding(feature_dim_dict,
                                                                                 embedding_size,
                                                                                 l2_reg_embedding,
                                                                                 0, init_std,
                                                                                 seed, False)
    fg_input = concat_fun(fg_deep_emb_list, axis=1)
    origin_input = concat_fun(deep_emb_list, axis=1)

    if len(conv_filters) > 0:
        new_features = FGCNNLayer(
            conv_filters, conv_kernel_width, new_maps, pooling_width)(fg_input)
        combined_input = concat_fun([origin_input, new_features], axis=1)
    else:
        combined_input = origin_input

    inner_product = tf.keras.layers.Flatten()(InnerProductLayer()(
        tf.keras.layers.Lambda(unstack, mask=[None] * combined_input.shape[1].value)(combined_input)))
    linear_signal = tf.keras.layers.Flatten()(combined_input)
    dnn_input = tf.keras.layers.Concatenate()([linear_signal, inner_product])
    dnn_input = tf.keras.layers.Flatten()(dnn_input)

    final_logit = DNN(dnn_hidden_units, dropout_rate=dnn_dropout,
                      l2_reg=l2_reg_dnn)(dnn_input)
    final_logit = tf.keras.layers.Dense(1, use_bias=False)(final_logit)
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
