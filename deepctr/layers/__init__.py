import tensorflow as tf

from .activation import Dice
from .core import DNN, LocalActivationUnit, PredictionLayer
from .interaction import (CIN, FM, AFMLayer, BiInteractionPooling, CrossNet,
                          InnerProductLayer, InteractingLayer,
                          OutterProductLayer, FGCNNLayer,SENETLayer,BilinearInteraction)
from .normalization import LayerNormalization
from .sequence import (AttentionSequencePoolingLayer, BiasEncoding, BiLSTM,
                       KMaxPooling, SequencePoolingLayer,WeightedSequenceLayer,
                       Transformer, DynamicGRU)
from .utils import NoMask, Hash,Linear,Add

custom_objects = {'tf': tf,
                  'InnerProductLayer': InnerProductLayer,
                  'OutterProductLayer': OutterProductLayer,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'FM': FM,
                  'AFMLayer': AFMLayer,
                  'CrossNet': CrossNet,
                  'BiInteractionPooling': BiInteractionPooling,
                  'LocalActivationUnit': LocalActivationUnit,
                  'Dice': Dice,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer,
                  'CIN': CIN,
                  'InteractingLayer': InteractingLayer,
                  'LayerNormalization': LayerNormalization,
                  'BiLSTM': BiLSTM,
                  'Transformer': Transformer,
                  'NoMask': NoMask,
                  'BiasEncoding': BiasEncoding,
                  'KMaxPooling': KMaxPooling,
                  'FGCNNLayer': FGCNNLayer,
                  'Hash': Hash,
                  'Linear':Linear,
                  'DynamicGRU': DynamicGRU,
                  'SENETLayer':SENETLayer,
                  'BilinearInteraction':BilinearInteraction,
                  'WeightedSequenceLayer':WeightedSequenceLayer,
                  'Add':Add
                  }
