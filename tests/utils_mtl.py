# test utils for multi task learning

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model, save_model

from deepctr.feature_column import SparseFeat, DenseFeat, DEFAULT_GROUP_NAME
from deepctr.layers import custom_objects


def get_mtl_test_data(sample_size=10, embedding_size=4, sparse_feature_num=1,
                      dense_feature_num=1, task_types=('binary', 'binary'),
                      hash_flag=False, prefix='', use_group=False):
    feature_columns = []
    model_input = {}

    for i in range(sparse_feature_num):
        if use_group:
            group_name = str(i % 3)
        else:
            group_name = DEFAULT_GROUP_NAME
        dim = np.random.randint(1, 10)
        feature_columns.append(
            SparseFeat(prefix + 'sparse_feature_' + str(i), dim, embedding_size, use_hash=hash_flag, dtype=tf.int32,
                       group_name=group_name))

    for i in range(dense_feature_num):
        def transform_fn(x): return (x - 0.0) / 1.0

        feature_columns.append(
            DenseFeat(
                prefix + 'dense_feature_' + str(i),
                1,
                dtype=tf.float32,
                transform_fn=transform_fn
            )
        )

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            model_input[fc.name] = np.random.randint(0, fc.vocabulary_size, sample_size)
        elif isinstance(fc, DenseFeat):
            model_input[fc.name] = np.random.random(sample_size)
    y_list = []  # multi label
    for task in task_types:
        if task == 'binary':
            y = np.random.randint(0, 2, sample_size)
            y_list.append(y)
        else:
            y = np.random.random(sample_size)
            y_list.append(y)

    return model_input, y_list, feature_columns


def check_mtl_model(model, model_name, x, y_list, task_types, check_model_io=True):
    """
    compile model,train and evaluate it,then save/load weight and model file.
    :param model:
    :param model_name:
    :param x:
    :param y_list: mutil label of y
    :param check_model_io: test save/load model file or not
    :return:
    """
    loss_list = []
    metric_list = []
    for task_type in task_types:
        if task_type == 'binary':
            loss_list.append('binary_crossentropy')
            # metric_list.append('accuracy')
        elif task_type == 'regression':
            loss_list.append('mean_squared_error')
            # metric_list.append('mae')
    print('loss:', loss_list)
    print('metric:', metric_list)
    model.compile('adam', loss=loss_list, metrics=metric_list)
    model.fit(x, y_list, batch_size=100, epochs=1, validation_split=0.5)

    print(model_name + " test train valid pass!")
    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    os.remove(model_name + '_weights.h5')
    print(model_name + " test save load weight pass!")
    if check_model_io:
        save_model(model, model_name + '.h5')
        model = load_model(model_name + '.h5', custom_objects)
        os.remove(model_name + '.h5')
        print(model_name + " test save load model pass!")

    print(model_name + " test pass!")
