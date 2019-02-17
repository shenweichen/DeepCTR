# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
    (https://arxiv.org/abs/1708.04617)

"""
import tensorflow as tf
from ..input_embedding import preprocess_input_embedding
from ..layers.core import PredictionLayer
from ..layers.interaction import AFMLayer, FM
from ..utils import check_feature_config_dict
from ..layers.utils import concat_fun


def AFM(feature_dim_dict, embedding_size=8, use_attention=True, attention_factor=8,
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, keep_prob=1.0, init_std=0.0001, seed=1024,
        final_activation='sigmoid',):
    """Instantiates the Attentonal Factorization Machine architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_attention: bool,whether use attention or not,if set to ``False``.it is the same as **standard Factorization Machine**
    :param attention_factor: positive integer,units in attention net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_att: float. L2 regularizer strength applied to attention net
    :param keep_prob: float in (0,1]. keep_prob after attention net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :return: A Keras model instance.
    """

    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, linear_logit, inputs_list = preprocess_input_embedding(feature_dim_dict, embedding_size,
                                                                          l2_reg_embedding, l2_reg_linear, init_std,
                                                                          seed, True)

    fm_input = concat_fun(deep_emb_list, axis=1)
    if use_attention:
        fm_logit = AFMLayer(attention_factor, l2_reg_att,
                            keep_prob, seed)(deep_emb_list)
    else:
        fm_logit = FM()(fm_input)

    final_logit = tf.keras.layers.add([linear_logit, fm_logit])
    output = PredictionLayer(final_activation)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
