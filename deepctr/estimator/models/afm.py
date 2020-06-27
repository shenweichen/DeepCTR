# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
    (https://arxiv.org/abs/1708.04617)

"""
import tensorflow as tf

from ..feature_column import get_linear_logit, input_from_feature_columns
from ..utils import deepctr_model_fn, DNN_SCOPE_NAME, variable_scope
from ...layers.interaction import AFMLayer, FM
from ...layers.utils import concat_func


def AFMEstimator(linear_feature_columns, dnn_feature_columns, use_attention=True, attention_factor=8,
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0, seed=1024,
                 task='binary', model_dir=None, config=None, linear_optimizer='Ftrl',
                 dnn_optimizer='Adagrad', training_chief_hooks=None):
    """Instantiates the Attentional Factorization Machine architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_attention: bool,whether use attention or not,if set to ``False``.it is the same as **standard Factorization Machine**
    :param attention_factor: positive integer,units in attention net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_att: float. L2 regularizer strength applied to attention net
    :param afm_dropout: float in [0,1), Fraction of the attention net output units to dropout.
    :param seed: integer ,to use as random seed.
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

        with variable_scope(DNN_SCOPE_NAME):
            sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                                 l2_reg_embedding=l2_reg_embedding)
            if use_attention:

                fm_logit = AFMLayer(attention_factor, l2_reg_att, afm_dropout,
                                    seed)(sparse_embedding_list, training=train_flag)
            else:
                fm_logit = FM()(concat_func(sparse_embedding_list, axis=1))

        logits = linear_logits + fm_logit

        return deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks=training_chief_hooks)

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=config)
