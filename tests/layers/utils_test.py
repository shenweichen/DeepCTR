import pytest
import numpy as np
import tensorflow as tf
from deepctr.layers.utils import Hash
from tests.utils import layer_test

try:
    from tensorflow.python.keras.utils import CustomObjectScope
except ImportError:
    from tensorflow.keras.utils import CustomObjectScope


@pytest.mark.parametrize(
    'num_buckets,mask_zero,vocabulary_path,input_data,expected_output',
    [
        (3 + 1, False, None, ['lakemerson'], None),
        (3 + 1, True, None, ['lakemerson'], None),
        (
        3 + 1, False, "./tests/layers/vocabulary_example.csv", [['lake'], ['johnson'], ['lakemerson']], [[1], [3], [0]])
    ]
)
def test_Hash(num_buckets, mask_zero, vocabulary_path, input_data, expected_output):
    if not hasattr(tf, 'version') or tf.version.VERSION < '2.0.0':
        return

    with CustomObjectScope({'Hash': Hash}):
        layer_test(Hash,
                   kwargs={'num_buckets': num_buckets, 'mask_zero': mask_zero, 'vocabulary_path': vocabulary_path},
                   input_dtype=tf.string, input_data=np.array(input_data, dtype='str'),
                   expected_output_dtype=tf.int64, expected_output=expected_output)
