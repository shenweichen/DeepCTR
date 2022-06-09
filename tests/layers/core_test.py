import pytest
import tensorflow as tf
from tensorflow.python.keras.layers import PReLU

try:
    from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
except ImportError:
    from tensorflow.python.keras.utils import CustomObjectScope
from deepctr import layers
from deepctr.layers import Dice
from tests.layers.interaction_test import BATCH_SIZE, EMBEDDING_SIZE, SEQ_LENGTH
from tests.utils import layer_test


@pytest.mark.parametrize(
    'hidden_units,activation',
    [(hidden_units, activation)
     for hidden_units in [(), (10,)]
     for activation in ['sigmoid', Dice, PReLU]
     ]
)
def test_LocalActivationUnit(hidden_units, activation):
    if tf.__version__ >= '1.13.0' and activation != 'sigmoid':
        return

    with CustomObjectScope({'LocalActivationUnit': layers.LocalActivationUnit}):
        layer_test(layers.LocalActivationUnit,
                   kwargs={'hidden_units': hidden_units, 'activation': activation, 'dropout_rate': 0.5},
                   input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE), (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE)])


@pytest.mark.parametrize(
    'hidden_units,use_bn',
    [(hidden_units, use_bn)
     for hidden_units in [(), (10,)]
     for use_bn in [True, False]
     ]
)
def test_DNN(hidden_units, use_bn):
    with CustomObjectScope({'DNN': layers.DNN}):
        layer_test(layers.DNN, kwargs={'hidden_units': hidden_units, 'use_bn': use_bn, 'dropout_rate': 0.5},
                   input_shape=(
                       BATCH_SIZE, EMBEDDING_SIZE))


@pytest.mark.parametrize(
    'task,use_bias',
    [(task, use_bias)
     for task in ['binary', 'regression']
     for use_bias in [True, False]
     ]
)
def test_PredictionLayer(task, use_bias):
    with CustomObjectScope({'PredictionLayer': layers.PredictionLayer}):
        layer_test(layers.PredictionLayer, kwargs={'task': task, 'use_bias': use_bias
                                                   }, input_shape=(BATCH_SIZE, 1))


@pytest.mark.xfail(reason="dim size must be 1 except for the batch size dim")
def test_test_PredictionLayer_invalid():
    # with pytest.raises(ValueError):
    with CustomObjectScope({'PredictionLayer': layers.PredictionLayer}):
        layer_test(layers.PredictionLayer, kwargs={'use_bias': True,
                                                   }, input_shape=(BATCH_SIZE, 2, 1))
