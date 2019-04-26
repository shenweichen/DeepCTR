import pytest
from tensorflow.python.keras.layers import PReLU
from tensorflow.python.keras.utils import CustomObjectScope

from deepctr import layers

from tests.utils import layer_test

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
    'reduce_sum',
    [reduce_sum
     for reduce_sum in [True, False]
     ]
)
def test_InnerProductLayer(reduce_sum):
    with CustomObjectScope({'InnerProductLayer': layers.InnerProductLayer}):
        layer_test(layers.InnerProductLayer, kwargs={
            'reduce_sum': reduce_sum}, input_shape=(BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


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


def test_FM():
    with CustomObjectScope({'FM': layers.FM}):
        layer_test(layers.FM, kwargs={}, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


def test_AFMLayer():
    with CustomObjectScope({'AFMLayer': layers.AFMLayer}):
        layer_test(layers.AFMLayer, kwargs={}, input_shape=[(
            BATCH_SIZE, 1, EMBEDDING_SIZE)]*FIELD_SIZE)


@pytest.mark.parametrize(
    'layer_size,activation,split_half',
    [(layer_size, activation, split_half)
     for activation in ['linear', PReLU]
     for split_half in [True, False]
     for layer_size in [(10,), (10, 8)]
     ]
)
def test_CIN(layer_size, activation, split_half):
    with CustomObjectScope({'CIN': layers.CIN}):
        layer_test(layers.CIN, kwargs={"layer_size": layer_size, "activation":
                                       activation, "split_half": split_half}, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


@pytest.mark.parametrize(
    'layer_size',
    [(), (3, 10)
     ]
)
def test_test_CIN_invalid(layer_size):
    with pytest.raises(ValueError):
        with CustomObjectScope({'CIN': layers.CIN}):
            layer_test(layers.CIN, kwargs={"layer_size": layer_size}, input_shape=(
                BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


@pytest.mark.parametrize(
    'head_num,use_res',
    [(head_num, use_res,)
     for head_num in [1, 2]
     for use_res in [True, False]
     ]
)
def test_InteractingLayer(head_num, use_res,):
    with CustomObjectScope({'InteractingLayer': layers.InteractingLayer}):
        layer_test(layers.InteractingLayer, kwargs={"head_num": head_num, "use_res":
                                                    use_res, }, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


