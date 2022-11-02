"""
Author:
    Mincai Lai, laimc@shanghaitech.edu.cn

    Weichen Shen, weichenswc@163.com

Reference:
    [1] Ruder S. An overview of multi-task learning in deep neural networks[J]. arXiv preprint arXiv:1706.05098, 2017.(https://arxiv.org/pdf/1706.05098.pdf)
"""

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense

from ...feature_column import build_input_features, input_from_feature_columns
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import combined_dnn_input


def SharedBottom(dnn_feature_columns, bottom_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    """Instantiates the SharedBottom multi-task learning Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bottom_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared bottom DNN.
    :param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks

    :return: A Keras model instance.
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
    shared_bottom_output = DNN(bottom_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(
        dnn_input)

    tasks_output = []
    for task_type, task_name in zip(task_types, task_names):
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(shared_bottom_output)

        logit = Dense(1, use_bias=False)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        tasks_output.append(output)

    model = Model(inputs=inputs_list, outputs=tasks_output)
    return model
