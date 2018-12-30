from itertools import chain

from tensorflow.python.keras import Input
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Embedding, Dense, Reshape
from tensorflow.python.keras.regularizers import l2
from .sequence import SequencePoolingLayer


def create_input_dict(feature_dim_dict, prefix=''):
    sparse_input = {feat: Input(shape=(1,), name=prefix+'sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])}
    dense_input = {feat: Input(shape=(1,), name=prefix+'dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])}
    return sparse_input, dense_input


def create_sequence_input_dict(feature_dim_dict):

    sequence_dim_dict = feature_dim_dict.get('sequence',[])
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

def merge_sequence_input(embedding_dict,embed_list,sequence_input_dict,sequence_len_dict,sequence_max_len_dict,sequence_pooling_dict):
    if len(sequence_input_dict) > 0:
        sequence_embed_dict = get_embedding_vec_list(embedding_dict,sequence_input_dict)
        sequence_embed_list = get_pooling_vec_list(sequence_embed_dict,sequence_len_dict,sequence_max_len_dict,sequence_pooling_dict)
        embed_list += sequence_embed_list

    return embed_list

def get_embedding_vec_list(embedding_dict, input_dict):

    return [embedding_dict[feat](v)
            for feat, v in input_dict.items()]

def get_pooling_vec_list(sequence_embed_dict,sequence_len_dict,sequence_max_len_dict,sequence_pooling_dict):
    return  [SequencePoolingLayer(sequence_max_len_dict[feat], sequence_pooling_dict[feat])(
        [v, sequence_len_dict[feat]]) for feat, v in sequence_embed_dict.items()]


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), inputs))))