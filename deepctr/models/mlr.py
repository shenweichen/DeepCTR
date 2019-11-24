# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
"""
from tensorflow.python.keras.layers import  Activation, dot
from tensorflow.python.keras.models import Model

from ..layers.core import PredictionLayer
from ..inputs import build_input_features,get_linear_logit
from ..layers.utils import concat_func

def MLR(region_feature_columns, base_feature_columns=None, region_num=4,
        l2_reg_linear=1e-5,
        init_std=0.0001, seed=1024, task='binary',
        bias_feature_columns=None):
    """Instantiates the Mixed Logistic Regression/Piece-wise Linear Model.

    :param region_feature_columns: An iterable containing all the features used by region part of the model.
    :param base_feature_columns: An iterable containing all the features used by base part of the model.
    :param region_num: integer > 1,indicate the piece number
    :param l2_reg_linear: float. L2 regularizer strength applied to weight
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param bias_feature_columns: An iterable containing all the features used by bias part of the model.
    :return: A Keras model instance.
    """

    if region_num <= 1:
        raise ValueError("region_num must > 1")


    if base_feature_columns is None or len(base_feature_columns) == 0:
        base_feature_columns = region_feature_columns

    if bias_feature_columns is None:
        bias_feature_columns = []

    features = build_input_features(region_feature_columns + base_feature_columns + bias_feature_columns)

    inputs_list = list(features.values())

    region_score = get_region_score(features,region_feature_columns,region_num,l2_reg_linear,init_std,seed)
    learner_score = get_learner_score(features,base_feature_columns,region_num,l2_reg_linear,init_std,seed,task=task)

    final_logit = dot([region_score,learner_score],axes=-1)

    if bias_feature_columns is not None and len(bias_feature_columns) > 0:
        bias_score =get_learner_score(features,bias_feature_columns,1,l2_reg_linear,init_std,seed,prefix='bias_',task='binary')

        final_logit = dot([final_logit,bias_score],axes=-1)

    model = Model(inputs=inputs_list, outputs=final_logit)
    return model


def get_region_score(features,feature_columns, region_number, l2_reg, init_std, seed,prefix='region_',seq_mask_zero=True):

    region_logit =concat_func([get_linear_logit(features, feature_columns, init_std=init_std, seed=seed + i,
                                                prefix=prefix + str(i + 1), l2_reg=l2_reg) for i in range(region_number)])
    return Activation('softmax')(region_logit)

def get_learner_score(features,feature_columns, region_number, l2_reg, init_std, seed,prefix='learner_',seq_mask_zero=True,task='binary'):
    region_score = [PredictionLayer(task=task,use_bias=False)(
        get_linear_logit(features, feature_columns, init_std=init_std, seed=seed + i, prefix=prefix + str(i + 1),
                         l2_reg=l2_reg)) for i in
                    range(region_number)]

    return concat_func(region_score)

