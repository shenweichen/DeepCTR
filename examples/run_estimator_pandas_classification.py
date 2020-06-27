import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_pandas

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

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, data[feat].nunique()), 4))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, data[feat].nunique()))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)

    # Not setting default value for continuous feature. filled with mean.

    train_model_input = input_fn_pandas(train, sparse_features + dense_features, 'label', shuffle=True)
    test_model_input = input_fn_pandas(test, sparse_features + dense_features, None, shuffle=False)

    # 4.Define Model,train,predict and evaluate
    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, task='binary')

    model.train(train_model_input)
    pred_ans_iter = model.predict(test_model_input)
    pred_ans = list(map(lambda x: x['pred'], pred_ans_iter))
    #
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
