import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('..')
import numpy as np
from deepctr.models import MMOE
from deepctr.inputs import SparseFeat, DenseFeat,get_feature_names

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.do simple Transformation for dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.set hashing space for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=1000,embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns, )

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}


    # 4.Define Model,train,predict and evaluate
    prediction_model, train_model = MMOE(dnn_feature_columns, num_tasks=2, tasks=['binary', 'binary'], use_uncertainty=True)
    train_model.compile("adam", loss=None,
                  metrics=['binary_crossentropy'], )
    train_x = list(train_model_input.values()) + [train[target].values, train[target].values]
    history = train_model.fit(train_x,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = prediction_model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans[0]), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans[0]), 4))
