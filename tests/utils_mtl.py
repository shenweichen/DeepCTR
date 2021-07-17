#test utils for multi task learning

import os
import numpy as np

from tensorflow.python.keras.models import load_model, save_model
from deepctr.layers import custom_objects

def get_mtl_test_data():
    test_data = np.load('adult_mini.npy', allow_pickle=True)
    #(x, y_list, feature_columns)
    return test_data[0], test_data[1], test_data[2]

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
    for type in task_types:
        if type=='binary':
            loss_list.append('binary_crossentropy')
            metric_list.append('AUC')
        elif type=='regression':
            loss_list.append('mean_squared_error')
            metric_list.append('mae')
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