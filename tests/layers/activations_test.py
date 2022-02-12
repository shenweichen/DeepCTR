import pytest

from deepctr.layers import activation

try:
    from tensorflow.python.keras.utils import CustomObjectScope
except ImportError:
    from tensorflow.keras.utils import CustomObjectScope
from tests.utils import layer_test

BATCH_SIZE = 5
FIELD_SIZE = 4
EMBEDDING_SIZE = 3

def test_dice():
    with CustomObjectScope({'Dice': activation.Dice}):
        layer_test(activation.Dice, kwargs={},
                   input_shape=(2, 3))


@pytest.mark.parametrize(
    'tau',
    [0.1, 1, 10]
)
def test_edcn_regulation(tau):
    with CustomObjectScope({'RegulationLayer': activation.RegulationLayer}):
        layer_test(activation.RegulationLayer, kwargs={'tau': tau},
                   input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE)] * FIELD_SIZE)