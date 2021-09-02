"""
Author:
    Mincai Lai, laimc@shanghaitech.edu.cn

Reference:
    [1] Ma X, Zhao L, Huang G, et al. Entire space multi-task model: An effective approach for estimating post-click conversion rate[C]//The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.(https://arxiv.org/abs/1804.07931)
"""

import tensorflow as tf

from ...feature_column import build_input_features, input_from_feature_columns
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import combined_dnn_input


def ESMM(dnn_feature_columns, tower_dnn_hidden_units=(256, 128, 64), l2_reg_embedding=0.00001, l2_reg_dnn=0,
         seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_type=('binary', 'binary'),
         task_names=('ctr', 'ctcvr')):
    """Instantiates the Entire Space Multi-Task Model architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tower_dnn_hidden_units:  layer number and units in each layer of task-specific DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task_type:  str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss.
    :param task_names: list of str, indicating the predict target of each tasks. default value is ['ctr', 'ctcvr']

    :return: A Keras model instance.
    """
    if len(task_names) != 2:
        raise ValueError("the length of task_names must be equal to 2")

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    ctr_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(
        dnn_input)
    cvr_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(
        dnn_input)

    ctr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(ctr_output)
    cvr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cvr_output)

    ctr_pred = PredictionLayer(task_type, name=task_names[0])(ctr_logit)
    cvr_pred = PredictionLayer(task_type)(cvr_logit)

    ctcvr_pred = tf.keras.layers.Multiply(name=task_names[1])([ctr_pred, cvr_pred])  # CTCVR = CTR * CVR

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[ctr_pred, ctcvr_pred])
    return model
