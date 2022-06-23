# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
    (https://arxiv.org/abs/1708.04617)

"""
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer
from ..layers.interaction import CoActionLayer
from ..layers.utils import concat_func, add_func

def CAN(dnn_feature_columns, co_action_config, l2_reg_embedding=1e-5, seed=1024, task='binary'):
    """Instantiates the CAN architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param co_action_config: A dict containing the bindings with all the features(target, hist_pref_seq) .
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    group_embedding_dict, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                         seed, prefix='', support_dense=False, support_group=True)

    # co-action for target type with multi hist pref seq
    can_output_list = []
    for conf in co_action_config:
        cur_can_layer = CoActionLayer(group_embedding_dict[conf['target']], conf['co_action_conf'], name=conf['name'])
        for his_pref_seq in conf['pref_seq']:
            can_output_list.append(cur_can_layer(group_embedding_dict[his_pref_seq]))

    can_output = concat_func(can_output_list)
    final_logit = Dense(1, use_bias=False)(can_output)
    output = PredictionLayer(task)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model
