from .core import LocalActivationUnit,MLP,PredictionLayer
from .interaction import AFMLayer,BiInteractionPooling,CIN,CrossNet,FM,InnerProductLayer,InteractingLayer,OutterProductLayer
from .normalization import LayerNormalization
from .activation import Dice
from .sequence import SequencePoolingLayer,AttentionSequencePoolingLayer,BiLSTM,Transformer,Position_Embedding,BiasEncoding
from .utils import NoMask

custom_objects = {'InnerProductLayer': InnerProductLayer,
                  'OutterProductLayer': OutterProductLayer,
                  'MLP': MLP,
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
                  'NoMask':NoMask,
                  'BiasEncoding':BiasEncoding}