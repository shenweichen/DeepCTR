from deepctr import activations
from tensorflow.python.keras.utils import CustomObjectScope
from .utils import layer_test


def test_dice():
    with CustomObjectScope({'Dice': activations.Dice}):
        layer_test(activations.Dice, kwargs={},
                   input_shape=(2, 3))
