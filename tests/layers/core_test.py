import pytest
from tensorflow.python.keras.layers import PReLU
from tensorflow.python.keras.utils import CustomObjectScope

from deepctr import layers
from deepctr.layers import Dice
from tests.layers.interaction_test import BATCH_SIZE, EMBEDDING_SIZE, SEQ_LENGTH
from tests.utils import layer_test


@pytest.mark.parametrize(
    'hidden_size,activation',
    [(hidden_size, activation)
     for hidden_size in [(), (10,)]
     for activation in ['sigmoid', Dice, PReLU]
     ]
)
def test_LocalActivationUnit(hidden_size, activation):
    with CustomObjectScope({'LocalActivationUnit': layers.LocalActivationUnit}):
        layer_test(layers.LocalActivationUnit, kwargs={'hidden_size': hidden_size, 'activation': activation},
                   input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE), (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE)])


@pytest.mark.parametrize(
    'hidden_size,use_bn',
    [(hidden_size, use_bn)
     for hidden_size in [(), (10,)]
     for use_bn in [True, False]
     ]
)
def test_MLP(hidden_size, use_bn):
    with CustomObjectScope({'MLP': layers.MLP}):
        layer_test(layers.MLP, kwargs={'hidden_size': hidden_size, 'use_bn': use_bn}, input_shape=(
            BATCH_SIZE, EMBEDDING_SIZE))


@pytest.mark.parametrize(
    'activation,use_bias',
    [(activation, use_bias)
     for activation in ['sigmoid', PReLU]
     for use_bias in [True, False]
     ]
)
def test_PredictionLayer(activation, use_bias):
    with CustomObjectScope({'PredictionLayer': layers.PredictionLayer}):
        layer_test(layers.PredictionLayer, kwargs={'activation': activation, 'use_bias': use_bias
                                                   }, input_shape=(BATCH_SIZE, 1))


@pytest.mark.xfail(reason="dim size must be 1 except for the batch size dim")
def test_test_PredictionLayer_invalid():
    # with pytest.raises(ValueError):
    with CustomObjectScope({'PredictionLayer': layers.PredictionLayer}):
        layer_test(layers.PredictionLayer, kwargs={'use_bias': True,
                                                   }, input_shape=(BATCH_SIZE, 2, 1))