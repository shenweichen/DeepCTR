from itertools import chain

from tensorflow.python.keras import Input
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Embedding, Dense, Reshape
from tensorflow.python.keras.regularizers import l2
from .sequence import SequencePoolingLayer
from .utils import get_linear_logit


def create_input_dict(feature_dim_dict, prefix=''):
    sparse_input = {feat: Input(shape=(1,), name=prefix+'sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])}
    dense_input = {feat: Input(shape=(1,), name=prefix+'dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])}
    return sparse_input, dense_input


def create_sequence_input_dict(feature_dim_dict):

    sequence_dim_dict = feature_dim_dict.get('sequence', [])
    sequence_input_dict = {feat.name: Input(shape=(feat.maxlen,), name='seq_' + str(
        i) + '-' + feat.name) for i, feat in enumerate(sequence_dim_dict)}
    sequence_pooling_dict = {feat.name: feat.combiner
                             for i, feat in enumerate(sequence_dim_dict)}
    sequence_len_dict = {feat.name: Input(shape=(
        1,), name='seq_length'+str(i)+'-'+feat.name) for i, feat in enumerate(sequence_dim_dict)}
    sequence_max_len_dict = {feat.name: feat.maxlen
                             for i, feat in enumerate(sequence_dim_dict)}
    return sequence_input_dict, sequence_pooling_dict, sequence_len_dict, sequence_max_len_dict


def create_embedding_dict(feature_dim_dict, embedding_size, init_std, seed, l2_reg, prefix='sparse'):
    if embedding_size == 'auto':

        sparse_embedding = {feat: Embedding(feature_dim_dict["sparse"][feat], 6 * int(pow(feature_dim_dict["sparse"][feat], 0.25)),
                                            embeddings_initializer=RandomNormal(
            mean=0.0, stddev=init_std, seed=seed),
            embeddings_regularizer=l2(l2_reg),
            name=prefix+'_emb_' + str(i) + '-' + feat) for i, feat in
            enumerate(feature_dim_dict["sparse"])}
    else:

        sparse_embedding = {feat: Embedding(feature_dim_dict["sparse"][feat], embedding_size,
                                            embeddings_initializer=RandomNormal(
            mean=0.0, stddev=init_std, seed=seed),
            embeddings_regularizer=l2(l2_reg),
            name=prefix+'_emb_' + str(i) + '-' + feat) for i, feat in
            enumerate(feature_dim_dict["sparse"])}

    if 'sequence' in feature_dim_dict:
        count = len(sparse_embedding)
        sequence_dim_list = feature_dim_dict['sequence']
        for feat in sequence_dim_list:
            if feat.name not in sparse_embedding:
                if embedding_size == "auto":
                    sparse_embedding[feat.name] = Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                            embeddings_initializer=RandomNormal(
                                                                mean=0.0, stddev=init_std, seed=seed),
                                                            embeddings_regularizer=l2(
                                                                l2_reg),
                                                            name=prefix + '_emb_' + str(count) + '-' + feat.name)

                else:
                    sparse_embedding[feat.name] = Embedding(feat.dimension, embedding_size,
                                                            embeddings_initializer=RandomNormal(
                                                                mean=0.0, stddev=init_std, seed=seed),
                                                            embeddings_regularizer=l2(
                                                                l2_reg),
                                                            name=prefix+'_emb_' + str(count) + '-' + feat.name)


                count += 1

    return sparse_embedding


def merge_dense_input(dense_input_, embed_list, embedding_size, l2_reg):
    dense_input = list(dense_input_.values())
    if len(dense_input) > 0:
        continuous_embedding_list = list(
            map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg), ),
                dense_input))
        continuous_embedding_list = list(
            map(Reshape((1, embedding_size)), continuous_embedding_list))
        embed_list += continuous_embedding_list

    return embed_list


def merge_sequence_input(embedding_dict, embed_list, sequence_input_dict, sequence_len_dict, sequence_max_len_dict, sequence_pooling_dict):
    if len(sequence_input_dict) > 0:
        sequence_embed_dict = get_varlen_embedding_vec_dict(
            embedding_dict, sequence_input_dict)
        sequence_embed_list = get_pooling_vec_list(
            sequence_embed_dict, sequence_len_dict, sequence_max_len_dict, sequence_pooling_dict)
        embed_list += sequence_embed_list

    return embed_list


def get_embedding_vec_list(embedding_dict, input_dict):

    return [embedding_dict[feat](v)
            for feat, v in input_dict.items()]


def get_varlen_embedding_vec_dict(embedding_dict, input_dict):

    return {feat: embedding_dict[feat](v)
            for feat, v in input_dict.items()}


def get_pooling_vec_list(sequence_embed_dict, sequence_len_dict, sequence_max_len_dict, sequence_pooling_dict):
    return [SequencePoolingLayer(sequence_max_len_dict[feat], sequence_pooling_dict[feat])(
        [v, sequence_len_dict[feat]]) for feat, v in sequence_embed_dict.items()]


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), inputs))))

def get_inputs_embedding(feature_dim_dict,embedding_size,l2_reg_embedding,l2_reg_linear,init_std,seed,include_linear=True):
    sparse_input_dict, dense_input_dict = create_input_dict(feature_dim_dict)
    sequence_input_dict, sequence_pooling_dict, sequence_input_len_dict, sequence_max_len_dict = create_sequence_input_dict(
        feature_dim_dict)

    deep_sparse_emb_dict = create_embedding_dict(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding)

    deep_emb_list = get_embedding_vec_list(deep_sparse_emb_dict, sparse_input_dict)

    deep_emb_list = merge_sequence_input(deep_sparse_emb_dict, deep_emb_list, sequence_input_dict,
                                         sequence_input_len_dict, sequence_max_len_dict, sequence_pooling_dict)

    deep_emb_list = merge_dense_input(
        dense_input_dict, deep_emb_list, embedding_size, l2_reg_embedding)
    if include_linear:
        linear_sparse_emb_dict = create_embedding_dict(
            feature_dim_dict, 1, init_std, seed, l2_reg_linear, 'linear')
        linear_emb_list = get_embedding_vec_list(linear_sparse_emb_dict, sparse_input_dict)
        linear_emb_list = merge_sequence_input(linear_sparse_emb_dict, linear_emb_list, sequence_input_dict,
                                               sequence_input_len_dict,
                                               sequence_max_len_dict, sequence_pooling_dict)

        linear_logit = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear)
    else:
        linear_logit = None

    inputs_list = get_inputs_list(
        [sparse_input_dict, dense_input_dict, sequence_input_dict, sequence_input_len_dict])
    return deep_emb_list,linear_logit,inputs_list