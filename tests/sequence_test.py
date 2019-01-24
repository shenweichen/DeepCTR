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

    'seq_len_max,mode',

    [(SEQ_LENGTH, mode)

     for mode in ['sum', 'mean', 'max']
     ]

)
def test_SequencePoolingLayer(seq_len_max, mode):
    with CustomObjectScope({'SequencePoolingLayer': sequence.SequencePoolingLayer}):
        layer_test(sequence.SequencePoolingLayer, kwargs={'seq_len_max': seq_len_max, 'mode': mode},
                   input_shape=[(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE), (BATCH_SIZE, 1)])
