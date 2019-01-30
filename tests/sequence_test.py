import pytest
from tensorflow.python.keras.utils import CustomObjectScope
import  tensorflow as tf
from deepctr import sequence

from .utils import layer_test

tf.keras.backend.set_learning_phase(True)
BATCH_SIZE = 4
EMBEDDING_SIZE = 8
SEQ_LENGTH = 10


@pytest.mark.parametrize(

    'weight_normalization',

    [True,False
     ]

)
def test_AttentionSequencePoolingLayer(weight_normalization):
    with CustomObjectScope({'AttentionSequencePoolingLayer': sequence.AttentionSequencePoolingLayer}):
        layer_test(sequence.AttentionSequencePoolingLayer, kwargs={'weight_normalization': weight_normalization},
                   input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE), (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE), (BATCH_SIZE, 1)])


@pytest.mark.parametrize(

    'mode,supports_masking,input_shape',

    [ ('sum',False,[(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE),(BATCH_SIZE,1)]) ,('mean',True,(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE)),( 'max',True,(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE))
     ]

)
def test_SequencePoolingLayer(mode,supports_masking,input_shape):
    with CustomObjectScope({'SequencePoolingLayer': sequence.SequencePoolingLayer}):
        layer_test(sequence.SequencePoolingLayer, kwargs={ 'mode': mode,'supports_masking':supports_masking},
                   input_shape=input_shape,supports_masking=supports_masking)
