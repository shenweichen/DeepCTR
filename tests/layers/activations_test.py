from deepctr.layers import activation

try:
    from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
except ImportError:
    from tensorflow.python.keras.utils import CustomObjectScope
from tests.utils import layer_test


def test_dice():
    with CustomObjectScope({'Dice': activation.Dice}):
        layer_test(activation.Dice, kwargs={},
                   input_shape=(2, 3))
