import pytest
from packaging import version
try:
    from tensorflow.python.keras.utils import CustomObjectScope
except:
    from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
from deepctr.layers import sequence

from tests.utils import layer_test

tf.keras.backend.set_learning_phase(True)
BATCH_SIZE = 4
EMBEDDING_SIZE = 8
SEQ_LENGTH = 10


@pytest.mark.parametrize(

    'weight_normalization',

    [True, False
     ]

)
def test_AttentionSequencePoolingLayer(weight_normalization):
    with CustomObjectScope({'AttentionSequencePoolingLayer': sequence.AttentionSequencePoolingLayer}):
        layer_test(sequence.AttentionSequencePoolingLayer, kwargs={'weight_normalization': weight_normalization},
                   input_shape=[(BATCH_SIZE, 1, EMBEDDING_SIZE), (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE), (BATCH_SIZE, 1)])


@pytest.mark.parametrize(

    'mode,supports_masking,input_shape',

    [('sum', False, [(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE), (BATCH_SIZE, 1)]), ('mean', True, (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE)), ('max', True, (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE))
     ]

)
def test_SequencePoolingLayer(mode, supports_masking, input_shape):
    if version.parse(tf.__version__) >= version.parse('1.14.0') and mode!='sum': #todo check further version
       return
    with CustomObjectScope({'SequencePoolingLayer': sequence.SequencePoolingLayer}):
        layer_test(sequence.SequencePoolingLayer, kwargs={'mode': mode, 'supports_masking': supports_masking},
                   input_shape=input_shape, supports_masking=supports_masking)


# @pytest.mark.parametrize(
#
#     'supports_masking,input_shape',
#
#     [( False, [(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE), (BATCH_SIZE, 1),(BATCH_SIZE, 1)]), ( True, [(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE),(BATCH_SIZE, 1)])
#      ]
#
# )
# def test_WeightedSequenceLayer(supports_masking, input_shape):
#     # if version.parse(tf.__version__) >= version.parse('1.14.0') : #todo check further version
#     #    return
#     with CustomObjectScope({'WeightedSequenceLayer': sequence.WeightedSequenceLayer}):
#         layer_test(sequence.WeightedSequenceLayer, kwargs={'supports_masking': supports_masking},
#                    input_shape=input_shape, supports_masking=supports_masking)
#


@pytest.mark.parametrize(

    'merge_mode',
    ['concat', 'ave', 'fw',
     ]

)
def test_BiLSTM(merge_mode):
    with CustomObjectScope({'BiLSTM': sequence.BiLSTM}):
        layer_test(sequence.BiLSTM, kwargs={'merge_mode': merge_mode, 'units': EMBEDDING_SIZE,'dropout_rate':0.0}, #todo 0.5
                   input_shape=(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE))


def test_Transformer():
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution() #todo
    with CustomObjectScope({'Transformer': sequence.Transformer}):
        layer_test(sequence.Transformer, kwargs={'att_embedding_size': 1, 'head_num': 8, 'use_layer_norm': True, 'supports_masking': False,'dropout_rate':0.5},
                   input_shape=[(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE), (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_SIZE), (BATCH_SIZE, 1), (BATCH_SIZE, 1)])


def test_KMaxPooling():
    with CustomObjectScope({'KMaxPooling':sequence.KMaxPooling}):
        layer_test(sequence.KMaxPooling,kwargs={'k':3,'axis':1},input_shape=(BATCH_SIZE,SEQ_LENGTH,EMBEDDING_SIZE,2))
