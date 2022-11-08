import tensorflow as tf

from .activation import Dice
from .core import DNN, LocalActivationUnit, PredictionLayer, RegulationModule
from .interaction import (CIN, FM, AFMLayer, BiInteractionPooling, CrossNet, CrossNetMix,
                          InnerProductLayer, InteractingLayer,
                          OutterProductLayer, FGCNNLayer, SENETLayer, BilinearInteraction,
                          FieldWiseBiInteraction, FwFMLayer, FEFMLayer, BridgeModule)
from .normalization import LayerNormalization
from .sequence import (AttentionSequencePoolingLayer, BiasEncoding, BiLSTM,
                       KMaxPooling, SequencePoolingLayer, WeightedSequenceLayer,
                       Transformer, DynamicGRU, PositionEncoding)
from .utils import NoMask, Hash, Linear, _Add, combined_dnn_input, softmax, reduce_sum, Concat

custom_objects = {'tf': tf,
                  'InnerProductLayer': InnerProductLayer,
                  'OutterProductLayer': OutterProductLayer,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'FM': FM,
                  'AFMLayer': AFMLayer,
                  'CrossNet': CrossNet,
                  'CrossNetMix': CrossNetMix,
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
                  'Linear': Linear,
                  'Concat': Concat,
                  'DynamicGRU': DynamicGRU,
                  'SENETLayer': SENETLayer,
                  'BilinearInteraction': BilinearInteraction,
                  'WeightedSequenceLayer': WeightedSequenceLayer,
                  '_Add': _Add,
                  'FieldWiseBiInteraction': FieldWiseBiInteraction,
                  'FwFMLayer': FwFMLayer,
                  'softmax': softmax,
                  'FEFMLayer': FEFMLayer,
                  'reduce_sum': reduce_sum,
                  'PositionEncoding': PositionEncoding,
                  'RegulationModule': RegulationModule,
                  'BridgeModule': BridgeModule
                  }
