import tensorflow as tf
from deepctr.layers.utils import Hash
from deepctr.feature_column import SparseFeat


def test_hash():
    try:
        import tensorflow.python.ops.numpy_ops.np_config as np_config
        np_config.enable_numpy_behavior()
    finally:
        pass
    vocab_path = "./tests/layers/vocabulary_example.csv"
    vocab_size = 3+1
    sf = SparseFeat('user_id', vocab_size, vocabulary_path=vocab_path)
    hash = Hash(num_buckets=sf.vocabulary_size, vocabulary_path=sf.vocabulary_path, default_value=0)
    assert hash(tf.constant('lake')).numpy() == 1
    assert hash(tf.constant('johnson')).numpy() == 3
    assert hash(tf.constant('lakemerson')).numpy() == 0

    hash_val = tf.strings.to_hash_bucket_fast('lakemerson', vocab_size).numpy()
    hash = Hash(num_buckets=vocab_size)
    assert hash(tf.constant('lakemerson')).numpy() == hash_val

    hash_val = tf.strings.to_hash_bucket_fast('lakemerson', vocab_size-1).numpy() + 1
    hash = Hash(num_buckets=vocab_size, mask_zero=True)
    assert hash(tf.constant('lakemerson')).numpy() == hash_val

    hash = Hash(num_buckets=vocab_size, mask_zero=True)
    assert hash(tf.constant('0')).numpy() == 0
