import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.estimator.inputs import input_fn_tfrecord
from deepctr.estimator.models.wdl import WDLEstimator

if __name__ == "__main__":
    data = pd.read_csv('../criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

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

    #train, test = train_test_split(data, test_size=0.2)

    # Not setting default value for continuous feature. filled with mean.
    feature_description = {k: tf.FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})

    #feature_description = tf.feature_column.make_parse_example_spec(linear_feature_columns+dnn_feature_columns)
    feature_description['label'] = tf.FixedLenFeature(dtype=tf.float32, shape=1)


    train_model_input = input_fn_tfrecord('./criteo_sample.tr.tfrecords',feature_description,'label',batch_size=256,num_epochs=1)
    test_model_input = input_fn_tfrecord('./criteo_sample.te.tfrecords', feature_description, 'label',batch_size=2**14,num_epochs=1)

    # 4.Define Model,train,predict and evaluate
    model = WDLEstimator(linear_feature_columns, dnn_feature_columns, l2_reg_dnn=0, l2_reg_embedding=0.0, l2_reg_linear=1e-5)

    model.train(train_model_input)
    eval_result = model.evaluate(test_model_input)

    print(eval_result)
    # all_input = linear_feature_columns + dnn_feature_columns
    # feature_spec = tf.feature_column.make_parse_example_spec(all_input)
    # ser_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    # export_model = model.export_savedmodel('./serv/',ser_input_fn)

