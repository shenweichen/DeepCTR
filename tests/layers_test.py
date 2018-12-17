import pytest
from deepctr import layers
from utils import layer_test
from tensorflow.python.keras.utils import CustomObjectScope


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
