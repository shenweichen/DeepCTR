import pytest
from deepctr import layers
from deepctr.activations import Dice
from utils import layer_test
from tensorflow.python.keras.utils import CustomObjectScope
from tensorflow.python.keras.layers import PReLU


BATCH_SIZE = 4
FIELD_SIZE = 3
EMBEDDING_SIZE = 8
SEQ_LENGTH = 10


@pytest.mark.parametrize(

    'layer_num,l2_reg',

    [(layer_num, l2_reg)

     for layer_num in [0, 1, 2, ]

     for l2_reg in [0, 1, ]
     ]

)
def test_CrossNet(layer_num, l2_reg,):
    with CustomObjectScope({'CrossNet': layers.CrossNet}):
        layer_test(layers.CrossNet, kwargs={
                   'layer_num': layer_num, 'l2_reg': l2_reg}, input_shape=(2, 3))


def test_CrossNet_invalid():
    with pytest.raises(ValueError):
        with CustomObjectScope({'CrossNet': layers.CrossNet}):
            layer_test(layers.CrossNet, kwargs={
                'layer_num': 1, 'l2_reg': 0}, input_shape=(2, 3, 4))


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
    'reduce_sum',
    [reduce_sum
     for reduce_sum in [True, False]
     ]
)
def test_InnerProductLayer(reduce_sum):
    with CustomObjectScope({'InnerProductLayer': layers.InnerProductLayer}):
        layer_test(layers.InnerProductLayer, kwargs={
            'reduce_sum': reduce_sum}, input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE)]*FIELD_SIZE)


@pytest.mark.parametrize(
    'kernel_type',
    [kernel_type
     for kernel_type in ['mat', 'vec', 'num']
     ]
)
def test_OutterProductLayer(kernel_type):
    with CustomObjectScope({'OutterProductLayer': layers.OutterProductLayer}):
        layer_test(layers.OutterProductLayer, kwargs={
            'kernel_type': kernel_type}, input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE)]*FIELD_SIZE)


def test_BiInteractionPooling():
    with CustomObjectScope({'BiInteractionPooling': layers.BiInteractionPooling}):
        layer_test(layers.BiInteractionPooling, kwargs={},
                   input_shape=(BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


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
        layer_test(layers.PredictionLayer, kwargs={'use_bias': use_bias
                                                   }, input_shape=(BATCH_SIZE, 2, 1))


def test_FM():
    with CustomObjectScope({'FM': layers.FM}):
        layer_test(layers.FM, kwargs={}, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


def test_AFMLayer():
    with CustomObjectScope({'AFMLayer': layers.AFMLayer}):
        layer_test(layers.AFMLayer, kwargs={}, input_shape=[(
            BATCH_SIZE, 1, EMBEDDING_SIZE)]*FIELD_SIZE)
