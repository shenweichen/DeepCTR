import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import tensorflow as tf
from deepctr.models import ESMM
from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def mask_loss(y_true, y_pred):
    return tf.constant(0.0)

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[name] for name in fixlen_feature_names]

    test_model_input = [test[name] for name in fixlen_feature_names]

    # 4.Define Model,train,predict and evaluate
    model = ESMM(dnn_feature_columns, task='binary')
    model.compile("adam", ["binary_crossentropy",mask_loss,"binary_crossentropy"],
                  metrics=['binary_crossentropy',auc], )

    history = model.fit(train_model_input, [train[target].values,train[target].values,train[target].values],
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("ctr LogLoss", round(log_loss(test[target].values, pred_ans[0]), 4))
    print("ctr AUC", round(roc_auc_score(test[target].values, pred_ans[0]), 4))

    print("cvr LogLoss", round(log_loss(test[target].values, pred_ans[1]), 4))
    print("cvr AUC", round(roc_auc_score(test[target].values, pred_ans[1]), 4))
