# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)
"""
import tensorflow as tf

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import deepctr_model_fn, DNN_SCOPE_NAME, variable_scope
from ...layers.core import DNN
from ...layers.interaction import CrossNet
from ...layers.utils import combined_dnn_input


def DCNEstimator(linear_feature_columns, dnn_feature_columns, cross_num=2, dnn_hidden_units=(128, 128,),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5,
                 l2_reg_cross=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_use_bn=False,
                 dnn_activation='relu', task='binary', model_dir=None, config=None, linear_optimizer='Ftrl',
                 dnn_optimizer='Adagrad', training_chief_hooks=None):
    """Instantiates the Deep&Cross Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
    :param config: tf.RunConfig object to configure the runtime settings.
    :param linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the linear part of the model. Defaults to FTRL optimizer.
    :param dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
        the deep part of the model. Defaults to Adagrad optimizer.
    :param training_chief_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run on the chief worker during training.
    :return: A Tensorflow Estimator  instance.

    """
    if len(dnn_hidden_units) == 0 and cross_num == 0:
        raise ValueError("Either hidden_layer or cross layer must > 0")

    def _model_fn(features, labels, mode, config):
        train_flag = (mode == tf.estimator.ModeKeys.TRAIN)

        linear_logits = get_linear_logit(features, linear_feature_columns, l2_reg_linear=l2_reg_linear)

        with variable_scope(DNN_SCOPE_NAME):
            sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                 l2_reg_embedding=l2_reg_embedding)

            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

            if len(dnn_hidden_units) > 0 and cross_num > 0:  # Deep & Cross
                deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input, training=train_flag)
                cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross)(dnn_input)
                stack_out = tf.keras.layers.Concatenate()([cross_out, deep_out])
                final_logit = tf.keras.layers.Dense(
                    1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(stack_out)
            elif len(dnn_hidden_units) > 0:  # Only Deep
                deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input, training=train_flag)
                final_logit = tf.keras.layers.Dense(
                    1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(deep_out)
            elif cross_num > 0:  # Only Cross
                cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross)(dnn_input)
                final_logit = tf.keras.layers.Dense(
                    1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(cross_out)
            else:  # Error
                raise NotImplementedError

        logits = linear_logits + final_logit

        return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks=training_chief_hooks)

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=config)
