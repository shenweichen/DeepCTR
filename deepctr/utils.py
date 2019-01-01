import collections
import json
import logging
from threading import Thread

import requests
from tensorflow.python.keras.layers import Dense, Concatenate, add

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

def concat_fun(inputs,axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)

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
