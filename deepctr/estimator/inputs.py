import tensorflow as tf


def input_fn_pandas(df, features, label=None, batch_size=256, num_epochs=1, shuffle=False, queue_capacity_factor=10,
                    num_threads=1):
    if label is not None:
        y = df[label]
    else:
        y = None
    if tf.__version__ >= "2.0.0":
        return tf.compat.v1.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size,
                                                             num_epochs=num_epochs,
                                                             shuffle=shuffle,
                                                             queue_capacity=batch_size * queue_capacity_factor,
                                                             num_threads=num_threads)

    return tf.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size, num_epochs=num_epochs,
                                               shuffle=shuffle, queue_capacity=batch_size * queue_capacity_factor,
                                               num_threads=num_threads)


def input_fn_tfrecord(filenames, feature_description, label=None, batch_size=256, num_epochs=1, num_parallel_calls=8,
                      shuffle_factor=10, prefetch_factor=1,
                      ):
    def _parse_examples(serial_exmp):
        try:
            features = tf.parse_single_example(serial_exmp, features=feature_description)
        except AttributeError:
            features = tf.io.parse_single_example(serial_exmp, features=feature_description)
        if label is not None:
            labels = features.pop(label)
            return features, labels
        return features

    def input_fn():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls)
        if shuffle_factor > 0:
            dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        if prefetch_factor > 0:
            dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)
        try:
            iterator = dataset.make_one_shot_iterator()
        except AttributeError:
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

        return iterator.get_next()

    return input_fn
