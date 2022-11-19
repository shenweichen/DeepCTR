import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from examples.gen_tfrecords import write_tfrecord, get_feature_description, get_tfrecord_parse_func, get_tfdataset

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    # 1 Preprocess
    # 1.1 Fill NA/NaN values
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)

    # 1.2 Label Encoding for sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 1.3 Transform dense features by Min-Max scaling
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2 Specify the parameters for Embedding
    sparse_feat_max_len = {feat: data[feat].max() + 1 for feat in sparse_features}

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=sparse_feat_max_len[feat], embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    # 3 Write tfrecords
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train, valid = train_test_split(data, train_size=0.8, random_state=2020)
    valid, test = train_test_split(valid, test_size=0.5, random_state=2020)

    write_tfrecord("./criteo_sample.tr.tfrecords", train, sparse_features, dense_features, label_name=target[0])
    write_tfrecord("./criteo_sample.va.tfrecords", valid, sparse_features, dense_features, label_name=target[0])
    write_tfrecord("./criteo_sample.te.tfrecords", test, sparse_features, dense_features, label_name=target[0])

    # 4 Read tfrecords
    feature_description = get_feature_description(sparse_features, dense_features, label_name=target[0])

    tfrecord_parse_func = get_tfrecord_parse_func(feature_description, label_name=target[0])

    train_tfdataset = get_tfdataset("./criteo_sample.tr.tfrecords", tfrecord_parse_func, batch_size=256)
    valid_tfdataset = get_tfdataset("./criteo_sample.va.tfrecords", tfrecord_parse_func, batch_size=256)
    test_tfdataset = get_tfdataset("./criteo_sample.te.tfrecords", tfrecord_parse_func, batch_size=256)

    # 5 Define model, train, predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['binary_crossentropy'])

    history = model.fit(train_tfdataset,
                        batch_size=256, epochs=10, verbose=2, validation_data=valid_tfdataset)

    pred_ans = model.predict(test_tfdataset, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
