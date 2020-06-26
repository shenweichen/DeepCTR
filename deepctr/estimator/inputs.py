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
    return tf.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size, num_epochs=num_epochs,
                                               shuffle=shuffle, queue_capacity=queue_capacity, num_threads=num_threads)


def input_fn_tfrecord(filenames,feature_description,label=None,batch_size=256, num_epochs=1, shuffle=False,num_parallel_calls=10):
    def _parse_examples(serial_exmp):
        features = tf.parse_single_example(serial_exmp, features=feature_description)
        if label is not None:
            labels = features.pop(label)
            return features, labels
        return features

    def input_fn():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls).prefetch(buffer_size=batch_size*10)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size*10)

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    return input_fn


# def my_input_fn(file_path, feature_names,batch_size=256,perform_shuffle=False, repeat_count=1):
#     def decode_csv(line):
#         #default = [[0.]] + [[0.] for _ in range(13)] + [[0] for _ in range(26)]
#         default = [[0] for _ in range(len(feature_names))] + [[0.]]
#         parsed_line = tf.decode_csv(line,default)
#         label = parsed_line[-1:] # Last element is the label
#         del parsed_line[-1] # Delete last element
#         features = parsed_line # Everything (but last element) are the features
#         print(len(feature_names),len(features))
#         d = dict(zip(feature_names, features)), label
#         print(d)
#         return d
#
#     dataset = (tf.data.TextLineDataset(file_path) # Read text file
#        .skip(1) # Skip header row
#        .map(decode_csv,num_parallel_calls=10)) # Transform each elem by applying decode_csv fn
#     dataset.prefetch(batch_size*10,)
#     if perform_shuffle:
#        # Randomizes input using a window of 256 elements (read into memory)
#        dataset = dataset.shuffle(buffer_size=batch_size)
#     dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
#     dataset = dataset.batch(batch_size)  # Batch size to use
#     iterator = dataset.make_one_shot_iterator()
#     batch_features, batch_labels = iterator.get_next()
#     return batch_features, batch_labels


