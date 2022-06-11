"""
Author:
    Mincai Lai, laimc@shanghaitech.edu.cn

    Weichen Shen, weichenswc@163.com

Reference:
    [1] Ma J, Zhao Z, Yi X, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.(https://dl.acm.org/doi/abs/10.1145/3219819.3220007)
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda

from ...feature_column import build_input_features, input_from_feature_columns
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import combined_dnn_input, reduce_sum


def MMOE(dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
         gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
         dnn_activation='relu',
         dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    """Instantiates the Multi-gate Mixture-of-Experts multi-task learning architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_experts: integer, number of experts.
    :param expert_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param gate_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks

    :return: a Keras model instance
    """
    num_tasks = len(task_names)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if num_experts <= 1:
        raise ValueError("num_experts must be greater than 1")

    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # build expert layer
    expert_outs = []
    for i in range(num_experts):
        expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                             name='expert_' + str(i))(dnn_input)
        expert_outs.append(expert_network)

    expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outs)  # None,num_experts,dim

    mmoe_outs = []
    for i in range(num_tasks):  # one mmoe layer: nums_tasks = num_gates
        # build gate layers
        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                         name='gate_' + task_names[i])(dnn_input)
        gate_out = Dense(num_experts, use_bias=False, activation='softmax',
                         name='gate_softmax_' + task_names[i])(gate_input)
        gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

        # gate multiply the expert
        gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                 name='gate_mul_expert_' + task_names[i])([expert_concat, gate_out])
        mmoe_outs.append(gate_mul_expert)

    task_outs = []
    for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outs):
        # build tower layer
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(mmoe_out)

        logit = Dense(1, use_bias=False)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outs.append(output)

    model = Model(inputs=inputs_list, outputs=task_outs)
    return model
