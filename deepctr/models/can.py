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
from ..feature_column import build_input_features, input_from_seq_feature_columns
from ..layers.core import PredictionLayer
from ..layers.interaction import CoActionLayer
from ..layers.utils import concat_func


# template_can_config = [
#     {
#         'name': 'co_action_for_item',
#         'target': 'item_id',  # target emb need to reshape
#         'pref_seq': ['hist_item_id', ],  # seq emb need to co-action
#         'co_action_conf': {
#             'target_emb_w': [[16, 8], [8, 4]],
#             'target_emb_b': [0, 0],
#             'indep_action': False,
#             'orders': 3,  # exp non_linear trans
#         }
#     },
#     {
#         'name': 'co_action_for_cate',
#         'target': 'cate_id',
#         'pref_seq': ['hist_cate_id', ],
#         'co_action_conf': {
#             'target_emb_w': [[16, 8], [8, 4]],
#             'target_emb_b': [0, 0],
#             'indep_action': False,
#             'orders': 3,  # exp non_linear trans
#         }
#     }
# ]


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
    sequence_embed_dict, _ = input_from_seq_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                         seed, prefix='', support_dense=False)

    # co-action for target type with multi hist pref seq
    can_output_list = []
    for conf in co_action_config:
        cur_can_layer = CoActionLayer(sequence_embed_dict[conf['target']], conf['co_action_conf'], name=conf['name'])
        print(cur_can_layer._layer_name)
        for his_pref_seq in conf['pref_seq']:
            can_output_list.append(cur_can_layer(sequence_embed_dict[his_pref_seq]))

    can_output = concat_func(can_output_list)
    final_logit = Dense(1, use_bias=False)(can_output)
    output = PredictionLayer(task)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model
