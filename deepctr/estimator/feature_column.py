import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _EmbeddingColumn

from .utils import LINEAR_SCOPE_NAME, variable_scope, get_collection, get_GraphKeys, input_layer, get_losses


def linear_model(features, linear_feature_columns):
    if tf.__version__ >= '2.0.0':
        linear_logits = tf.compat.v1.feature_column.linear_model(features, linear_feature_columns)
    else:
        linear_logits = tf.feature_column.linear_model(features, linear_feature_columns)
    return linear_logits


def get_linear_logit(features, linear_feature_columns, l2_reg_linear=0):
    with variable_scope(LINEAR_SCOPE_NAME):
        if not linear_feature_columns:
            linear_logits = tf.Variable([[0.0]], name='bias_weights')
        else:

            linear_logits = linear_model(features, linear_feature_columns)

            if l2_reg_linear > 0:
                for var in get_collection(get_GraphKeys().TRAINABLE_VARIABLES, LINEAR_SCOPE_NAME)[:-1]:
                    get_losses().add_loss(l2_reg_linear * tf.nn.l2_loss(var, name=var.name.split(":")[0] + "_l2loss"),
                                          get_GraphKeys().REGULARIZATION_LOSSES)
    return linear_logits


def input_from_feature_columns(features, feature_columns, l2_reg_embedding=0.0):
    dense_value_list = []
    sparse_emb_list = []
    for feat in feature_columns:
        if is_embedding(feat):
            sparse_emb = tf.expand_dims(input_layer(features, [feat]), axis=1)
            sparse_emb_list.append(sparse_emb)
            if l2_reg_embedding > 0:
                get_losses().add_loss(l2_reg_embedding * tf.nn.l2_loss(sparse_emb, name=feat.name + "_l2loss"),
                                      get_GraphKeys().REGULARIZATION_LOSSES)

        else:
            dense_value_list.append(input_layer(features, [feat]))

    return sparse_emb_list, dense_value_list


def is_embedding(feature_column):
    try:
        from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn
    except:
        EmbeddingColumn = _EmbeddingColumn
    return isinstance(feature_column, (_EmbeddingColumn, EmbeddingColumn))
