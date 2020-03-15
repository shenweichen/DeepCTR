# -*- coding:utf-8 -*-
"""
Author:
    Yiyuan Liu, lyy930905@gmail.com

Reference:
    [1] [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import tensorflow as tf

from ..inputs import input_from_feature_columns, build_input_features, combined_dnn_input
from ..layers.core import PredictionLayer, DNN, MMOELayer, MultiLossLayer
from ..layers.utils import concat_func
from tensorflow.python.keras.layers import Input

def MMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, use_uncertainty=True,
         dnn_hidden_units=(128, 128), l2_reg_embedding=1e-5, l2_reg_dnn=0,
         task_dnn_units=None, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu'):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param use_uncertainty: whether to use uncertainty to weigh losses for each tasks.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: If use_uncertainty is False, return a Keras model instance, otherwise, return a tuple (prediction_model, train_model).
    train_model should be compiled and fit data first, and then prediction_model is used for prediction.
    """
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, init_std, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(mmoe_out) for mmoe_out in mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    if use_uncertainty:
        prediction_model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
        ys_true = [Input(shape=(1,)) for _ in tasks]
        loss_layer_inputs = ys_true + task_outputs
        model_out = MultiLossLayer(num_tasks, tasks)(loss_layer_inputs)
        model_inputs = inputs_list + ys_true
        train_model = tf.keras.models.Model(model_inputs, model_out)
        return prediction_model, train_model
    else:
        model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)
        return model
