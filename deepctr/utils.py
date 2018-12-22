from tensorflow.python.keras.layers import Input, Embedding
from tensorflow.python.keras.initializers import RandomNormal
from .activations import *
from .layers import *
from .sequence import *

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
                  'CIN': CIN, }


def get_input(feature_dim_dict, bias_feature_dim_dict=None):
    sparse_input = [Input(shape=(1,), name='sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])]
    dense_input = [Input(shape=(1,), name='dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])]
    if bias_feature_dim_dict is None:
        return sparse_input, dense_input
    else:
        bias_sparse_input = [Input(shape=(1,), name='bias_sparse_' + str(i) + '-' + feat) for i, feat in
                             enumerate(bias_feature_dim_dict["sparse"])]
        bias_dense_input = [Input(shape=(1,), name='bias_dense_' + str(i) + '-' + feat) for i, feat in
                            enumerate(bias_feature_dim_dict["dense"])]
        return sparse_input, dense_input, bias_sparse_input, bias_dense_input


def get_share_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    sparse_embedding = [Embedding(feature_dim_dict["sparse"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(feature_dim_dict["sparse"])]
    linear_embedding = [Embedding(feature_dim_dict["sparse"][feat], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                      seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(feature_dim_dict["sparse"])]

    return sparse_embedding, linear_embedding


def get_sep_embeddings(deep_feature_dim_dict, wide_feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    sparse_embedding = [Embedding(deep_feature_dim_dict["sparse"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(deep_feature_dim_dict["sparse"])]
    linear_embedding = [Embedding(wide_feature_dim_dict["sparse"][feat], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                      seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(wide_feature_dim_dict["sparse"])]

    return sparse_embedding, linear_embedding
