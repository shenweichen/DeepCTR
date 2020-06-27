# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com
    Harshit Pande

Reference:
    [1] Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising
    (https://arxiv.org/pdf/1806.03514.pdf)

"""

import tensorflow as tf

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import DNN_SCOPE_NAME, deepctr_model_fn, variable_scope
from ...layers.core import DNN
from ...layers.interaction import FwFMLayer
from ...layers.utils import concat_func, add_func, combined_dnn_input


def FwFMEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
                  l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_field_strength=0.00001, l2_reg_dnn=0,
                  seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary', model_dir=None,
                  config=None, linear_optimizer='Ftrl',
                  dnn_optimizer='Adagrad', training_chief_hooks=None):
    """Instantiates the DeepFwFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list if do not want DNN, the layer number and units
    in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_field_strength: float. L2 regularizer strength applied to the field pair strength parameters
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
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
        final_logit_components = [linear_logits]
        with variable_scope(DNN_SCOPE_NAME):
            sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                 l2_reg_embedding=l2_reg_embedding)

            fwfm_logit = FwFMLayer(num_fields=len(sparse_embedding_list), regularizer=l2_reg_field_strength)(
                concat_func(sparse_embedding_list, axis=1))

            final_logit_components.append(fwfm_logit)

            if dnn_hidden_units:
                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

                dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                                 dnn_use_bn, seed)(dnn_input, training=train_flag)
                dnn_logit = tf.keras.layers.Dense(
                    1, use_bias=False, activation=None)(dnn_output)
                final_logit_components.append(dnn_logit)

        logits = add_func(final_logit_components)

        return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks=training_chief_hooks)

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=config)
