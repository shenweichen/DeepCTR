from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Concatenate, Activation, Reshape
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import TruncatedNormal
from ..feature_column import build_input_features, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer
from ..layers.interaction import AFMLayer, CrossNetLayer
from ..layers.utils import concat_func


def AFN(linear_feature_columns, dnn_feature_columns, cross_num=2, cross_parameterization='vector',
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_cross=1e-5, afn_dropout=0, seed=1024,
        task='binary'):
    """Instantiates the Attentional Factorization Network architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: int, the number of cross layers in the CrossNet.
    :param cross_parameterization: str, one of "vector" or "matrix". Parameterization for the CrossNet.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to CrossNet
    :param afn_dropout: float in [0,1), Fraction of the CrossNet output units to dropout.
    :param seed: integer, to use as random seed.
    :param task: str, ``"binary"`` for binary logloss or ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                         seed, support_dense=True, support_group=True)

    linear_logit = Dense(1, use_bias=False, kernel_regularizer=l2(l2_reg_linear))(concat_func(dense_value_list))

    # CrossNet
    cross_input = Concatenate(axis=1)(list(group_embedding_dict.values()))
    for i in range(cross_num):
        cross_input = CrossNetLayer(cross_parameterization, l2_reg=l2_reg_cross)(cross_input)
        if afn_dropout:
            cross_input = Dropout(afn_dropout)(cross_input)

    # AFMLayer
    afm_input = Concatenate(axis=1)(list(group_embedding_dict.values()))
    afm_out = AFMLayer()(afm_input)

    # Concatenate
    dnn_input = Concatenate(axis=1)(list(group_embedding_dict.values()))
    dnn_input = Reshape((len(dnn_feature_columns), -1))(dnn_input)

    # DNN layers
    hidden_layers = [dnn_input]
    for i in range(3):
        fc = Dense(128, activation='relu', kernel_initializer=TruncatedNormal(seed=seed))(hidden_layers[-1])
        if afn_dropout:
            fc = Dropout(afn_dropout)(fc)
        hidden_layers.append(fc)

    # Output
    output = Concatenate(axis=1)([linear_logit, cross_input, afm_out, hidden_layers[-1]])
