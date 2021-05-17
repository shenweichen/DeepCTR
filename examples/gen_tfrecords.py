import tensorflow as tf

def make_example(line, sparse_feature_name, dense_feature_name, label_name):
    features = {feat: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line[1][feat])])) for feat in
                sparse_feature_name}
    features.update(
        {feat: tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][feat]])) for feat in dense_feature_name})
    features[label_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][label_name]]))
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(filename, df, sparse_feature_names, dense_feature_names, label_name):
    writer = tf.python_io.TFRecordWriter(filename)
    for line in df.iterrows():
        ex = make_example(line, sparse_feature_names, dense_feature_names, label_name)
        writer.write(ex.SerializeToString())
    writer.close()

# write_tfrecord('./criteo_sample.tr.tfrecords',train,sparse_features,dense_features,'label')
# write_tfrecord('./criteo_sample.te.tfrecords',test,sparse_features,dense_features,'label')
