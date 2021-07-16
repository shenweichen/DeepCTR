#test utils for multi task learning

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tensorflow.python.keras.models import load_model, save_model
from deepctr.layers import custom_objects
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

def get_mtl_test_data():
    data = pd.read_csv('./adult_mini.csv')
    #define dense and sparse features
    columns = data.columns.values.tolist()
    dense_features = ['fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
    sparse_features = [col for col in columns if col not in dense_features and col not in ['label_income', 'label_marital']]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    fixlen_feature_columns = [SparseFeat(feat, data[feat].max()+1, embedding_dim=16)for feat in sparse_features] \
    + [DenseFeat(feat, 1,) for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)
    #define X,y1,y2,yn...
    model_input = {name: data[name] for name in feature_names}
    y1 = data['label_income'].values
    y2 = data['label_marital'].values
    return model_input, [y1, y2], dnn_feature_columns

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