import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _EmbeddingColumn

from .utils import LINEAR_SCOPE_NAME


def get_linear_logit(features, linear_feature_columns,l2_reg_linear=0):
    with tf.variable_scope(LINEAR_SCOPE_NAME):
        if not linear_feature_columns:
            linear_logits = tf.Variable([[0.0]], name='bias_weights')
        else:
            linear_logits = tf.feature_column.linear_model(features, linear_feature_columns)
            if l2_reg_linear > 0:
                for var in  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, LINEAR_SCOPE_NAME)[:-1]:
                        tf.losses.add_loss(tf.nn.l2_loss(var, name=var.name.split(":")[0] + "_l2loss"),
                                           tf.GraphKeys.REGULARIZATION_LOSSES)
    return linear_logits


def input_from_feature_columns(features, feature_columns, l2_reg_embedding=0.0, expand_dim=True):
    dense_value_list = []
    sparse_emb_list = []
    for feat in feature_columns:
        if is_embedding(feat):
            sparse_emb = tf.feature_column.input_layer(features, [feat])
            sparse_emb_list.append(sparse_emb)
            if l2_reg_embedding > 0:
                tf.losses.add_loss(tf.nn.l2_loss(sparse_emb, name=feat.name + "_l2loss"),
                                   tf.GraphKeys.REGULARIZATION_LOSSES)

        else:
            dense_value_list.append(tf.feature_column.input_layer(features, [feat]))


    if expand_dim:
        sparse_emb_list = [tf.expand_dims(x, axis=1) for x in sparse_emb_list]

    return sparse_emb_list, dense_value_list


def is_embedding(feature_column):
    return isinstance(feature_column, (tf.contrib.layers.feature_column._EmbeddingColumn, _EmbeddingColumn))