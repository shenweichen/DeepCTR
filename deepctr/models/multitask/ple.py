"""
Author:
    Mincai Lai, laimc@shanghaitech.edu.cn

    Weichen Shen, weichenswc@163.com

Reference:
    [1] Tang H, Liu J, Zhao M, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations[C]//Fourteenth ACM Conference on Recommender Systems. 2020.(https://dl.acm.org/doi/10.1145/3383313.3412236)
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda

from ...feature_column import build_input_features, input_from_feature_columns
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import combined_dnn_input, reduce_sum


def PLE(dnn_feature_columns, shared_expert_num=1, specific_expert_num=1, num_levels=2,
        expert_dnn_hidden_units=(256,), tower_dnn_hidden_units=(64,), gate_dnn_hidden_units=(),
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
        task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    """Instantiates the multi level of Customized Gate Control of Progressive Layered Extraction architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param shared_expert_num: integer, number of task-shared experts.
    :param specific_expert_num: integer, number of task-specific experts.
    :param num_levels: integer, number of CGC levels.
    :param expert_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param gate_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks

    :return: a Keras model instance.
    """
    num_tasks = len(task_names)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")

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

    # single Extraction Layer
    def cgc_net(inputs, level_name, is_last=False):
        # inputs: [task1, task2, ... taskn, shared task]
        specific_expert_outputs = []
        # build task-specific expert layer
        for i in range(num_tasks):
            for j in range(specific_expert_num):
                expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                     seed=seed,
                                     name=level_name + 'task_' + task_names[i] + '_expert_specific_' + str(j))(
                    inputs[i])
                specific_expert_outputs.append(expert_network)

        # build task-shared expert layer
        shared_expert_outputs = []
        for k in range(shared_expert_num):
            expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                 seed=seed,
                                 name=level_name + 'expert_shared_' + str(k))(inputs[-1])
            shared_expert_outputs.append(expert_network)

        # task_specific gate (count = num_tasks)
        cgc_outs = []
        for i in range(num_tasks):
            # concat task-specific expert and task-shared expert
            cur_expert_num = specific_expert_num + shared_expert_num
            # task_specific + task_shared
            cur_experts = specific_expert_outputs[
                          i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs

            expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_specific_' + task_names[i])(
                inputs[i])  # gate[i] for task input[i]
            gate_out = Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_specific_' + task_names[i])(gate_input)
            gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=level_name + 'gate_mul_expert_specific_' + task_names[i])(
                [expert_concat, gate_out])
            cgc_outs.append(gate_mul_expert)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = num_tasks * specific_expert_num + shared_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

            expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_shared')(inputs[-1])  # gate for shared task input

            gate_out = Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_shared')(gate_input)
            gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=level_name + 'gate_mul_expert_shared')(
                [expert_concat, gate_out])

            cgc_outs.append(gate_mul_expert)
        return cgc_outs

    # build Progressive Layered Extraction
    ple_inputs = [dnn_input] * (num_tasks + 1)  # [task1, task2, ... taskn, shared task]
    ple_outputs = []
    for i in range(num_levels):
        if i == num_levels - 1:  # the last level
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=True)
        else:
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=False)
            ple_inputs = ple_outputs

    task_outs = []
    for task_type, task_name, ple_out in zip(task_types, task_names, ple_outputs):
        # build tower layer
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(ple_out)
        logit = Dense(1, use_bias=False)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outs.append(output)

    model = Model(inputs=inputs_list, outputs=task_outs)
    return model
