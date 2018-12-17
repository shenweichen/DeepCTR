from deepctr import activations
from utils import layer_test
from tensorflow.python.keras.utils import CustomObjectScope


def test_dice():
    with CustomObjectScope({'Dice': activations.Dice}):
        layer_test(activations.Dice, kwargs={},
                   input_shape=(2, 3))
