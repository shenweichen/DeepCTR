import pytest

try:
    from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
except ImportError:
    from tensorflow.python.keras.utils import CustomObjectScope
from deepctr import layers

from tests.utils import layer_test

BATCH_SIZE = 5
FIELD_SIZE = 4
EMBEDDING_SIZE = 3
SEQ_LENGTH = 10


def test_FEFMLayer():
    with CustomObjectScope({'FEFMLayer': layers.FEFMLayer}):
        layer_test(layers.FEFMLayer, kwargs={'regularizer': 0.000001},
                   input_shape=(BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


@pytest.mark.parametrize(
    'reg_strength',
    [0.000001]
)
def test_FwFM(reg_strength):
    with CustomObjectScope({'FwFMLayer': layers.FwFMLayer}):
        layer_test(layers.FwFMLayer, kwargs={'num_fields': FIELD_SIZE, 'regularizer': reg_strength},
                   input_shape=(BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


@pytest.mark.parametrize(

    'layer_num',

    [0, 1]

)
def test_CrossNet(layer_num, ):
    with CustomObjectScope({'CrossNet': layers.CrossNet}):
        layer_test(layers.CrossNet, kwargs={
            'layer_num': layer_num, }, input_shape=(2, 3))


# def test_CrossNet_invalid():
#     with pytest.raises(ValueError):
#         with CustomObjectScope({'CrossNet': layers.CrossNet}):
#             layer_test(layers.CrossNet, kwargs={
#                 'layer_num': 1, 'l2_reg': 0}, input_shape=(2, 3, 4))


@pytest.mark.parametrize(
    'reduce_sum',
    [reduce_sum
     for reduce_sum in [True, False]
     ]
)
def test_InnerProductLayer(reduce_sum):
    with CustomObjectScope({'InnerProductLayer': layers.InnerProductLayer}):
        layer_test(layers.InnerProductLayer, kwargs={
            'reduce_sum': reduce_sum}, input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE)] * FIELD_SIZE)


@pytest.mark.parametrize(
    'kernel_type',
    [kernel_type
     for kernel_type in ['mat', 'vec', 'num']
     ]
)
def test_OutterProductLayer(kernel_type):
    with CustomObjectScope({'OutterProductLayer': layers.OutterProductLayer}):
        layer_test(layers.OutterProductLayer, kwargs={
            'kernel_type': kernel_type}, input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE)] * FIELD_SIZE)


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
        layer_test(layers.AFMLayer, kwargs={'dropout_rate': 0.5}, input_shape=[(
            BATCH_SIZE, 1, EMBEDDING_SIZE)] * FIELD_SIZE)


@pytest.mark.parametrize(
    'layer_size,split_half',
    [((10,), False), ((10, 8), True)
     ]
)
def test_CIN(layer_size, split_half):
    with CustomObjectScope({'CIN': layers.CIN}):
        layer_test(layers.CIN, kwargs={"layer_size": layer_size, "split_half": split_half}, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


# @pytest.mark.parametrize(
#     'layer_size',
#     [(), (3, 10)
#      ]
# )
# def test_test_CIN_invalid(layer_size):
#     with pytest.raises(ValueError):
#         with CustomObjectScope({'CIN': layers.CIN}):
#             layer_test(layers.CIN, kwargs={"layer_size": layer_size}, input_shape=(
#                 BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


@pytest.mark.parametrize(
    'head_num,use_res',
    [(1, True), (2, False,)]
)
def test_InteractingLayer(head_num, use_res, ):
    with CustomObjectScope({'InteractingLayer': layers.InteractingLayer}):
        layer_test(layers.InteractingLayer, kwargs={"head_num": head_num, "use_res":
            use_res, }, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


def test_FGCNNLayer():
    with CustomObjectScope({'FGCNNLayer': layers.FGCNNLayer}):
        layer_test(layers.FGCNNLayer, kwargs={'filters': (4, 6,), 'kernel_width': (7, 7,)}, input_shape=(
            BATCH_SIZE, FIELD_SIZE, EMBEDDING_SIZE))


# def test_SENETLayer():
#     with CustomObjectScope({'SENETLayer': layers.SENETLayer}):
#         layer_test(layers.SENETLayer, kwargs={'reduction_ratio':2}, input_shape=[(
#             BATCH_SIZE, 1, EMBEDDING_SIZE)]*FIELD_SIZE)


@pytest.mark.parametrize(
    'bilinear_type',
    ['all', 'each', 'interaction'
     ]
)
def test_BilinearInteraction(bilinear_type):
    with CustomObjectScope({'BilinearInteraction': layers.BilinearInteraction}):
        layer_test(layers.BilinearInteraction, kwargs={'bilinear_type': bilinear_type}, input_shape=[(
            BATCH_SIZE, 1, EMBEDDING_SIZE)] * FIELD_SIZE)
