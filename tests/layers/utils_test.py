import pytest
import tensorflow as tf
from deepctr.layers.utils import Hash
from deepctr.feature_column import SparseFeat
from tests.utils import layer_test
try:
    from tensorflow.python.keras.utils import CustomObjectScope
except:
    from tensorflow.keras.utils import CustomObjectScope


@pytest.mark.parametrize(
    'num_buckets,mask_zero,vocabulary_path,input_data,expected_output',
    [
        (3+1, False, None, ['lakemerson'], None),
        (3+1, True, None, ['lakemerson'], None),
        (3+1, False, "./tests/layers/vocabulary_example.csv", [['lake'], ['johnson'], ['lakemerson']], [[1], [3], [0]])
    ]
)
def test_Hash(num_buckets, mask_zero, vocabulary_path, input_data, expected_output):
    if tf.version.VERSION < '2.0.0':
        return
    with CustomObjectScope({'Hash': Hash}):
        layer_test(Hash, kwargs={'num_buckets': num_buckets, 'mask_zero': mask_zero, 'vocabulary_path': vocabulary_path},
                   input_dtype=tf.string, input_data=tf.constant(input_data, dtype=tf.string),
                   expected_output_dtype=tf.int64, expected_output=expected_output)
