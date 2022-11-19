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
    features = {feat: _int64_feature(int(line[1][feat])) for feat in sparse_feature_name}

    features.update({feat: _float_feature(line[1][feat]) for feat in dense_feature_name})

    features[label_name] = _float_feature(line[1][label_name])
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(filename, df, sparse_feature_names, dense_feature_names, label_name, compression_type="GZIP"):
    writer = tf.io.TFRecordWriter(filename, options=compression_type)
    for line in df.iterrows():
        ex = make_example(line, sparse_feature_names, dense_feature_names, label_name)
        writer.write(ex.SerializeToString())
    writer.close()


# write_tfrecord('./criteo_sample.tr.tfrecords', train, sparse_features, dense_features, 'label')
# write_tfrecord('./criteo_sample.te.tfrecords', test, sparse_features, dense_features, 'label')

def get_feature_description(sparse_feature_name, dense_feature_name, label_name):
    feature_description = {feat: tf.io.FixedLenFeature([], tf.int64, default_value=0) for feat in sparse_feature_name}

    feature_description.update(
        {feat: tf.io.FixedLenFeature([], tf.float32, default_value=0.0) for feat in dense_feature_name})

    feature_description[label_name] = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)

    return feature_description


def get_tfrecord_parse_func(feature_description, label_name):
    def tfrecord_parse_func(serialized_example):
        parsed_tfrecord = tf.io.parse_single_example(serialized_example, feature_description)

        # features
        features = {}
        for feature_name, value in parsed_tfrecord.items():
            if feature_name != label_name:
                features[feature_name] = value

        # label
        labels = parsed_tfrecord[label_name]
        return features, labels

    return tfrecord_parse_func


def get_tfdataset(tfrecord_filepath, tfrecord_parse_func, batch_size=256, compression_type="GZIP"):
    files = [tfrecord_filepath]
    tfdataset = tf.data.TFRecordDataset(filenames=files, compression_type=compression_type)
    tfdataset = tfdataset \
        .map(tfrecord_parse_func) \
        .batch(batch_size)

    return tfdataset
