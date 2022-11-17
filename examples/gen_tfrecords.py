import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(line, sparse_feature_name, dense_feature_name, label_name):
    features = {feat: _int64_feature(value=[int(line[1][feat])]) for feat in sparse_feature_name}

    features.update({feat: _float_feature(value=[line[1][feat]]) for feat in dense_feature_name})

    features[label_name] = _float_feature(value=[line[1][label_name]])
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(filename, df, sparse_feature_names, dense_feature_names, label_name):
    writer = tf.io.TFRecordWriter(filename)
    for line in df.iterrows():
        ex = make_example(line, sparse_feature_names, dense_feature_names, label_name)
        writer.write(ex.SerializeToString())
    writer.close()

# write_tfrecord('./criteo_sample.tr.tfrecords', train, sparse_features, dense_features, 'label')
# write_tfrecord('./criteo_sample.te.tfrecords', test, sparse_features, dense_features, 'label')
