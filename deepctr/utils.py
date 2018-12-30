import collections
from itertools import chain
import json
import logging
from threading import Thread

import requests
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Embedding, Input, Dense, Concatenate, Reshape, add

from .activations import *
from .layers import *
from .sequence import *

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

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
                  'InteractingLayer': InteractingLayer}


VarLenFeature = collections.namedtuple(
    'VarLenFeatureConfig', ['name', 'dimension', 'maxlen', 'combiner'])


def create_input_dict(feature_dim_dict, prefix=''):
    sparse_input = {feat: Input(shape=(1,), name=prefix+'sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])}
    dense_input = {feat: Input(shape=(1,), name=prefix+'dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])}
    return sparse_input, dense_input


def create_sequence_input_dict(sequence_dim_dict):
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


def get_linear_logit(linear_term, dense_input_, l2_reg):
    if len(linear_term) > 1:
        linear_term = add(linear_term)
    elif len(linear_term) == 1:
        linear_term = linear_term[0]
    else:
        linear_term = None

    dense_input = list(dense_input_.values())
    if len(dense_input) > 0:
        dense_input__ = dense_input[0] if len(
            dense_input) == 1 else Concatenate()(dense_input)
        linear_dense_logit = Dense(
            1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg))(dense_input__)
        if linear_term is not None:
            linear_term = add([linear_dense_logit, linear_term])
        else:
            linear_term = linear_dense_logit

    return linear_term


def embed_dense_input(dense_input_, embed_list, embedding_size, l2_reg):
    dense_input = list(dense_input_.values())
    if len(dense_input) > 0:
        continuous_embedding_list = list(
            map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg), ),
                dense_input))
        continuous_embedding_list = list(
            map(Reshape((1, embedding_size)), continuous_embedding_list))
        embed_list += continuous_embedding_list

    return embed_list


def get_embedding_vec_list(embedding_dict, input_dict):

    return [embedding_dict[feat](v)
            for feat, v in input_dict.items()]


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), inputs))))


def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.python.org/pypi/deepctr/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if not ver.is_prerelease:
                        latest_version = max(latest_version, ver)
                if latest_version > version:
                    logging.warning('\nDeepCTR version {0} detected. Your version is {1}.\nUse `pip install -U deepctr` to upgrade.Changelog: https://github.com/shenweichen/DeepCTR/releases/tag/v{0}'.format(
                        latest_version, version))
        except Exception:
            return
    Thread(target=check, args=(version,)).start()
