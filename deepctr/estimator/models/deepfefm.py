# -*- coding:utf-8 -*-
"""
Author:
    Harshit Pande

Reference:
    [1] Field-Embedded Factorization Machines for Click-through Rate Prediction]
    (https://arxiv.org/abs/2009.09931)

"""

import tensorflow as tf

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import DNN_SCOPE_NAME, deepctr_model_fn, variable_scope
from ...layers.core import DNN
from ...layers.interaction import FEFMLayer
from ...layers.utils import concat_func, add_func, combined_dnn_input, reduce_sum


def DeepFEFMEstimator(linear_feature_columns, dnn_feature_columns,
                      dnn_hidden_units=(128, 128), l2_reg_linear=0.00001, l2_reg_embedding_feat=0.00001,
                      l2_reg_embedding_field=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0.0,
                      dnn_activation='relu', dnn_use_bn=False, task='binary', model_dir=None,
                      config=None, linear_optimizer='Ftrl', dnn_optimizer='Adagrad', training_chief_hooks=None):
    """Instantiates the DeepFEFM Network architecture or the shallow FEFM architecture (Ablation support not provided
    as estimator is meant for production, Ablation support provided in DeepFEFM implementation in models

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding_feat: float. L2 regularizer strength applied to embedding vector of features
    :param l2_reg_embedding_field: float, L2 regularizer to field embeddings
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
                                                                                 l2_reg_embedding=l2_reg_embedding_feat)

            fefm_interaction_embedding = FEFMLayer(
                regularizer=l2_reg_embedding_field)(concat_func(sparse_embedding_list, axis=1))

            fefm_logit = tf.keras.layers.Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=True))(
                fefm_interaction_embedding)

            final_logit_components.append(fefm_logit)

            if dnn_hidden_units:
                dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
                dnn_input = concat_func([dnn_input, fefm_interaction_embedding], axis=1)

                dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(
                    dnn_input, training=train_flag)

                dnn_logit = tf.keras.layers.Dense(
                    1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

                final_logit_components.append(dnn_logit)

        logits = add_func(final_logit_components)

        return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks=training_chief_hooks)

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=config)
