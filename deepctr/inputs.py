# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

from collections import OrderedDict, namedtuple
from itertools import chain

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Concatenate, Dense, Embedding, Input, Reshape, add,Flatten
from tensorflow.python.keras.regularizers import l2

from .layers.sequence import SequencePoolingLayer
from .layers.utils import Hash,concat_fun


class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'hash_flag', 'dtype','embedding_name'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None):
        if embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):

        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)


class VarLenSparseFeat(namedtuple('VarLenFeat', ['name', 'dimension', 'maxlen', 'combiner', 'hash_flag', 'dtype','embedding_name'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype, embedding_name)


def get_fixlen_feature_names(feature_columns):
    features = build_input_features(feature_columns, include_varlen=False,include_fixlen=True)
    return features.keys()

def get_varlen_feature_names(feature_columns):
    features = build_input_features(feature_columns, include_varlen=True,include_fixlen=False)
    return features.keys()


def build_input_features(feature_columns, include_varlen=True, mask_zero=True, prefix='',include_fixlen=True):
    input_features = OrderedDict()
    if include_fixlen:
        for fc in feature_columns:
            if isinstance(fc,SparseFeat):
                input_features[fc.name] = Input(
                    shape=(1,), name=prefix+fc.name, dtype=fc.dtype)
            elif isinstance(fc,DenseFeat):
                input_features[fc.name] = Input(
                    shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
    if include_varlen:
        for fc in feature_columns:
            if isinstance(fc,VarLenSparseFeat):
                input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + 'seq_' + fc.name,
                                                      dtype=fc.dtype)
        if not mask_zero:
            for fc in feature_columns:
                input_features[fc.name+"_seq_length"] = Input(shape=(
                    1,), name=prefix + 'seq_length_' + fc.name)
                input_features[fc.name+"_seq_max_length"] = fc.maxlen


    return input_features


# def create_varlenfeat_inputdict(feature_dim_dict, mask_zero=True,prefix=''):
#     sequence_dim_dict = feature_dim_dict.get('sequence', [])
#     sequence_input_dict = OrderedDict()
#     for feat in sequence_dim_dict:
#         sequence_input_dict[feat.name] = Input(shape=(feat.maxlen,), name=prefix+'seq_' + feat.name, dtype=feat.dtype)
#
#     if mask_zero:
#         sequence_len_dict, sequence_max_len_dict = None, None
#     else:
#         sequence_len_dict = {feat.name: Input(shape=(
#             1,), name=prefix+'seq_length_' + feat.name) for  feat in sequence_dim_dict}
#         sequence_max_len_dict = {feat.name: feat.maxlen
#                                  for  feat in sequence_dim_dict}
#     return sequence_input_dict, sequence_len_dict, sequence_max_len_dict


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    if embedding_size == 'auto':
        print("Notice:Do not use auto embedding in models other than DCN")
        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(l2_reg),
                                                 name=prefix + '_emb_' + feat.name) for feat in
                            sparse_feature_columns}
    else:

        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, embedding_size,
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(
                                                     l2_reg),
                                                 name=prefix + '_emb_'  + feat.name) for feat in
                            sparse_feature_columns}

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            if embedding_size == "auto":
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_seq_emb_' + feat.name,
                                                        mask_zero=seq_mask_zero)

            else:
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_seq_emb_' + feat.name,
                                                        mask_zero=seq_mask_zero)


    return sparse_embedding


def merge_dense_input(dense_input_, embed_list, embedding_size, l2_reg):
    dense_input = list(dense_input_.values())
    if len(dense_input) > 0:
        if embedding_size == "auto":
            print("Notice:Do not use auto embedding in models other than DCN")
            if len(dense_input) == 1:
                continuous_embedding_list = dense_input[0]
            else:
                continuous_embedding_list = Concatenate()(dense_input)
            continuous_embedding_list = Reshape(
                [1, len(dense_input)])(continuous_embedding_list)
            embed_list.append(continuous_embedding_list)

        else:
            continuous_embedding_list = list(
                map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg), ),
                    dense_input))
            continuous_embedding_list = list(
                map(Reshape((1, embedding_size)), continuous_embedding_list))
            embed_list += continuous_embedding_list

    return embed_list


def merge_sequence_input(embedding_dict, embed_list, sequence_input_dict, sequence_len_dict, sequence_max_len_dict,
                         sequence_fd_list):
    if len(sequence_input_dict) > 0:
        sequence_embed_dict = get_varlen_pooling_list(
            embedding_dict, sequence_input_dict, sequence_fd_list)
        sequence_embed_list = get_pooling_vec_list(
            sequence_embed_dict, sequence_len_dict, sequence_max_len_dict, sequence_fd_list)
        embed_list += sequence_embed_list

    return embed_list


def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0  or feat_name in return_feat_list:
            if fg.hash_flag:
                lookup_idx = Hash(fg.dimension,mask_zero=(feat_name in mask_feat_list))(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list

def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns):
    pooling_vec_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = feature_name + '_seq_length'
        if feature_length_name in features:
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
            [embedding_dict[feature_name], features[feature_length_name]])
        else:
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
            embedding_dict[feature_name])
        pooling_vec_list.append(vec)
    return pooling_vec_list

def get_pooling_vec_list(sequence_embed_dict, sequence_len_dict, sequence_max_len_dict, sequence_fd_list):
    if sequence_max_len_dict is None or sequence_len_dict is None:
        return [SequencePoolingLayer(feat.combiner, supports_masking=True)(sequence_embed_dict[feat.name]) for feat in
                sequence_fd_list]
    else:
        return [SequencePoolingLayer(feat.combiner, supports_masking=False)(
            [sequence_embed_dict[feat.name], sequence_len_dict[feat.name]]) for feat in sequence_fd_list]


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))

def create_embedding_matrix(feature_columns,l2_reg,init_std,seed,embedding_size, prefix=""):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed,
                                                 l2_reg, prefix=prefix + 'sparse')
    return sparse_emb_dict
#
# def get_inputs_embedding(linear_feature_columns, dnn_feature_columns, l2_reg_embedding, l2_reg_linear, init_std, seed,
#                          sparse_input_dict, dense_input_dict, sequence_input_dict, sequence_input_len_dict,
#                          sequence_max_len_dict, include_linear, embedding_size, prefix=""):
#
#     dnn_sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),dnn_feature_columns)) if dnn_feature_columns else []
#     dnn_varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenFeat), dnn_feature_columns)) if dnn_feature_columns else []
#     linear_sparse_feature_columns = list(filter(lambda x:isinstance(x, SparseFeat), linear_feature_columns)) if linear_feature_columns else []
#     linear_varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenFeat), linear_feature_columns)) if linear_feature_columns else []
#
#
#     dnn_sparse_emb_dict = create_embedding_dict(dnn_sparse_feature_columns, dnn_varlen_sparse_feature_columns, embedding_size, init_std, seed,
#                                                  l2_reg_embedding, prefix=prefix + 'sparse')
#
#     deep_emb_list = get_embedding_vec_list(
#         dnn_sparse_emb_dict, sparse_input_dict, dnn_sparse_feature_columns)
#
#     #todo 添加序列输入支持
#     #deep_emb_list = merge_sequence_input(dnn_sparse_emb_dict, deep_emb_list, sequence_input_dict,
#     #                                     sequence_input_len_dict, sequence_max_len_dict, dnn_varlen_sparse_feature_columns)
#
#     deep_emb_list = merge_dense_input(
#         dense_input_dict, deep_emb_list, embedding_size, l2_reg_embedding)
#
#     if include_linear:
#         linear_sparse_emb_dict = create_embedding_dict(linear_sparse_feature_columns, linear_varlen_sparse_feature_columns, 1, init_std, seed, l2_reg_linear,
#                                                        prefix + 'linear')
#
#         linear_emb_list = get_embedding_vec_list(
#             linear_sparse_emb_dict, sparse_input_dict, linear_feature_columns)
#         # todo 添加序列输入支持
#         #linear_emb_list = merge_sequence_input(linear_sparse_emb_dict, linear_emb_list, sequence_input_dict,
#         #                                       sequence_input_len_dict,
#         #                                       sequence_max_len_dict, linear_varlen_sparse_feature_columns)
#     else:
#         linear_emb_list = None
#
#     inputs_list = get_inputs_list(
#         [sparse_input_dict, dense_input_dict, sequence_input_dict, sequence_input_len_dict])
#     return inputs_list, deep_emb_list, linear_emb_list


def get_linear_logit_v1(linear_emb_list, dense_input_list, l2_reg):
    if len(linear_emb_list) > 1:
        linear_term = add(linear_emb_list)
    elif len(linear_emb_list) == 1:
        linear_term = linear_emb_list[0]
    else:
        linear_term = None

    if len(dense_input_list) > 0:
        dense_input__ = dense_input_list[0] if len(
            dense_input_list) == 1 else Concatenate()(dense_input_list)
        linear_dense_logit = Dense(
            1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg))(dense_input__)
        if linear_term is not None:
            linear_term = add([linear_dense_logit, linear_term])
        else:
            linear_term = linear_dense_logit

    return linear_term

#features,dnn_feature_columns,embedding_size,l2_reg_embedding,init_std,seed

def get_linear_logit(features, feature_columns, l2_reg, init_std, seed, prefix='linear'):

    linear_emb_list, dense_input_list = input_from_feature_columns(features,feature_columns,1,l2_reg,init_std,seed,prefix=prefix)

    if len(linear_emb_list) > 1:
        linear_term = add(linear_emb_list)
    elif len(linear_emb_list) == 1:
        linear_term = linear_emb_list[0]
    else:
        linear_term = None

    if len(dense_input_list) > 0:
        dense_input__ = dense_input_list[0] if len(
            dense_input_list) == 1 else Concatenate()(dense_input_list)
        linear_dense_logit = Dense(
            1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg))(dense_input__)
        if linear_term is not None:
            linear_term = add([linear_dense_logit, linear_term])
        else:
            linear_term = linear_dense_logit

    return linear_term

def embedding_lookup(sparse_embedding_dict,sparse_input_dict,sparse_feature_columns,return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0  or feature_name in return_feat_list:
            if fc.hash_flag:
                lookup_idx = Hash(fc.dimension,mask_zero=(feature_name in mask_feat_list))(sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))

    return embedding_vec_list

def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.hash_flag:
            lookup_idx = Hash(fc.dimension, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict

def get_dense_input(features,feature_columns):
    dense_feature_columns = list(filter(lambda x:isinstance(x,DenseFeat),feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list




def get_varlen_vec_list(embedding_dict, features, varlen_sparse_feature_columns):
    vec_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        feature_length_name = feature_name + "_seq_length"
        if feature_length_name in features:
            #seq_length = features[feature_length_name]
            vector = SequencePoolingLayer(fc.combiner, supports_masking=False)(
            [embedding_dict[feature_name], features[feature_length_name]])
        else:
            vector = SequencePoolingLayer(fc.combiner, supports_masking=True)(embedding_dict[feature_name])
        vec_list.append(vector)
    return vec_list



def input_from_feature_columns(features,feature_columns, embedding_size, l2_reg, init_std, seed,prefix=''):


    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    embedding_dict = create_embedding_matrix(feature_columns,l2_reg,init_std,seed,embedding_size, prefix=prefix)
    sparse_embedding_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features,feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict,features,varlen_sparse_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, varlen_sparse_feature_columns)
    sparse_embedding_list += sequence_embed_list

    return sparse_embedding_list, dense_value_list



def combined_dnn_input(sparse_embedding_list,dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_fun(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_fun(dense_value_list))
        return concat_fun([sparse_dnn_input,dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_fun(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_fun(dense_value_list))
    else:
        raise NotImplementedError

