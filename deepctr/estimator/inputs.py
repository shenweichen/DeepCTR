import tensorflow as tf
from ..layers.utils import combined_dnn_input

def input_fn_pandas(df, features, label=None, batch_size=256, num_epochs=1, shuffle=False, queue_capacity=2560,
                    num_threads=1):
    """

    :param df:
    :param features:
    :param label:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :param queue_capacity:
    :param num_threads:
    :return:
    """
    if label is not None:
        y = df[label]
    else:
        y = None
    if tf.__version__ >= "2.0.0":
        return tf.compat.v1.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size,
                                                             num_epochs=num_epochs,
                                                             shuffle=shuffle, queue_capacity=queue_capacity,
                                                             num_threads=num_threads)

    return tf.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size, num_epochs=num_epochs,
                                               shuffle=shuffle, queue_capacity=queue_capacity, num_threads=num_threads)


def input_fn_tfrecord(filenames, feature_description, label=None, batch_size=256, num_epochs=1, shuffle=False,
                      num_parallel_calls=10):
    def _parse_examples(serial_exmp):
        features = tf.parse_single_example(serial_exmp, features=feature_description)
        if label is not None:
            labels = features.pop(label)
            return features, labels
        return features

    def input_fn():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls).prefetch(
            buffer_size=batch_size * 10)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    return input_fn


