import tensorflow as tf

from deepctr.estimator.inputs import input_fn_tfrecord
from deepctr.estimator.models import DeepFMEstimator

if __name__ == "__main__":

    # 1.generate feature_column for linear part and dnn part

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, 100), 4))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 100))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 2.generate input data for model

    feature_description = {k: tf.FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})
    feature_description['label'] = tf.FixedLenFeature(dtype=tf.float32, shape=1)

    train_model_input = input_fn_tfrecord('./criteo_sample.tr.tfrecords',feature_description,'label',batch_size=256,num_epochs=1)
    test_model_input = input_fn_tfrecord('./criteo_sample.te.tfrecords', feature_description, 'label',batch_size=2**14,num_epochs=1)

    # 3.Define Model,train,predict and evaluate
    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, l2_reg_dnn=0, l2_reg_embedding=0.0, l2_reg_linear=1e-5)

    model.train(train_model_input)
    eval_result = model.evaluate(test_model_input)

    print(eval_result)
    # all_input = linear_feature_columns + dnn_feature_columns
    # feature_spec = tf.feature_column.make_parse_example_spec(all_input)
    # ser_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    # export_model = model.export_savedmodel('./serv/',ser_input_fn)

