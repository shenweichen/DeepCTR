# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

import json
import logging
from collections import namedtuple
from threading import Thread

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


class SingleFeat(namedtuple('SingleFeat', ['name', 'dimension', 'hash_flag', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension, hash_flag=False, dtype="float32"):
        return super(SingleFeat, cls).__new__(cls, name, dimension, hash_flag, dtype)


class VarLenFeat(namedtuple('VarLenFeat', ['name', 'dimension', 'maxlen', 'combiner', 'hash_flag', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", hash_flag=False, dtype="float32"):
        return super(VarLenFeat, cls).__new__(cls, name, dimension, maxlen, combiner, hash_flag, dtype)


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
                    logging.warning(
                        '\nDeepCTR version {0} detected. Your version is {1}.\nUse `pip install -U deepctr` to upgrade.Changelog: https://github.com/shenweichen/DeepCTR/releases/tag/v{0}'.format(
                            latest_version, version))
        except Exception:
            return

    Thread(target=check, args=(version,)).start()


def check_feature_config_dict(feature_dim_dict):
    if not isinstance(feature_dim_dict, dict):
        raise ValueError(
            "feature_dim_dict must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}")
    if "sparse" not in feature_dim_dict:
        feature_dim_dict['sparse'] = []
    if "dense" not in feature_dim_dict:
        feature_dim_dict['dense'] = []
    if "sequence" not in feature_dim_dict:
        feature_dim_dict["sequence"] = []  # TODO:check if it's ok

    if not isinstance(feature_dim_dict["sparse"], list):
        raise ValueError("feature_dim_dict['sparse'] must be a list,cur is", type(
            feature_dim_dict['sparse']))

    if not isinstance(feature_dim_dict["dense"], list):
        raise ValueError("feature_dim_dict['dense'] must be a list,cur is", type(
            feature_dim_dict['dense']))
