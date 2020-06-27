# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.(https://arxiv.org/pdf/1803.05170.pdf)
"""
import tensorflow as tf

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import deepctr_model_fn, DNN_SCOPE_NAME, variable_scope
from ...layers.core import DNN
from ...layers.interaction import CIN
from ...layers.utils import concat_func, add_func, combined_dnn_input


def xDeepFMEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
                     cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                     l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
                     dnn_activation='relu', dnn_use_bn=False, task='binary', model_dir=None, config=None,
                     linear_optimizer='Ftrl',
                     dnn_optimizer='Adagrad', training_chief_hooks=None):
    """Instantiates the xDeepFM architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
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

    def _model_fn(features, labels, mode, config):
        train_flag = (mode == tf.estimator.ModeKeys.TRAIN)

        linear_logits = get_linear_logit(features, linear_feature_columns, l2_reg_linear=l2_reg_linear)
        logits_list = [linear_logits]

        with variable_scope(DNN_SCOPE_NAME):
            sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                 l2_reg_embedding=l2_reg_embedding)
            fm_input = concat_func(sparse_embedding_list, axis=1)

            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                             dnn_use_bn, seed)(dnn_input, training=train_flag)
            dnn_logit = tf.keras.layers.Dense(
                1, use_bias=False, activation=None)(dnn_output)

            logits_list.append(dnn_logit)

            if len(cin_layer_size) > 0:
                exFM_out = CIN(cin_layer_size, cin_activation,
                               cin_split_half, l2_reg_cin, seed)(fm_input, training=train_flag)
                exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
                logits_list.append(exFM_logit)

        logits = add_func(logits_list)

        return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks=training_chief_hooks)

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=config)
