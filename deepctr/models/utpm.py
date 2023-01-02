# -*- coding:utf-8 -*-
"""
Author:
    Kunverse, czkconverse@gmail.com

Reference:
    [1] Su Yan, Xin Chen, Ran Huo, Xu Zhang, and Leyu Lin. 2020. Learning to Build User-tag Profile in Recommendation System. In Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management (CIKM '20). Association for Computing Machinery, New York, NY, USA, 2877–2884. https://doi.org/10.1145/3340531.3412719

"""
from tensorflow.keras import Model
from deepctr.feature_column import build_input_features
from deepctr.inputs import create_embedding_matrix, embedding_lookup, varlen_embedding_lookup
from deepctr.layers.interaction import AttentionUTPM, LinerOpLayerUTPM, CrossOpLayerUTPM, JointLossUTPM
from typing import List, Dict
from collections import OrderedDict

from tensorflow.keras.layers import Concatenate, Flatten, Dense, Embedding

from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepctr.layers import SequencePoolingLayer


def UTPM(norm_sparse_feature_columns, norm_varlen_feature_columns, tag_column,
         feature_embedding_dim=16, cross_op_layer_vec_emb_size=16, attention_embedding_size=16, dense_units=64,
         seed=1024):
    """Instantiates the UTPM Network architecture.

    :param norm_sparse_feature_columns: list. Input SparseFeats.
    :param norm_varlen_feature_columns: list. Input VarLenSparseFeat, except column of tag.
    :param tag_column: list. Input VarLenSparseFeat of tag, only one column of tag included in this list.
    :param feature_embedding_dim: integer. Embedding size of all features, in UTPM, embedding size of all features should be same.
    :param cross_op_layer_vec_emb_size: integer. Embedding size of cross feature layer.
    :param attention_embedding_size: integer. Units of attention operation.
    :param dense_units: integer. Dimensionality of the fully connect layer after cross feature layer.
    :param seed: integer. random seed.
    :return: A Keras model instance.
    """
    # 1 Inputs definition
    feature_columns = norm_sparse_feature_columns + norm_varlen_feature_columns + tag_column

    features = build_input_features(feature_columns)

    # 2 Special model structure
    # 2.1 Embedding of features except tag.
    nrom_embedding_dict = create_embedding_matrix(norm_sparse_feature_columns + norm_varlen_feature_columns,
                                                  l2_reg=0.001,
                                                  seed=seed, seq_mask_zero=True)

    # 2.2 Embedding of tag
    tag_embedding_dict = create_embedding_matrix(tag_column, l2_reg=0.001, seed=seed, seq_mask_zero=True)

    # 2.3 First attention layer
    # head 1
    att_first_layer_head_1_dict = create_attention_first_layer_head_dict(
        head_rank=1,
        var_feat_cols=norm_varlen_feature_columns + tag_column,
        att_dense_units=attention_embedding_size)

    att_second_dense_head_1 = Dense(attention_embedding_size, activation="relu", name="att_second_dense_head_1")

    # head 2
    att_first_layer_head_2_dict = create_attention_first_layer_head_dict(
        head_rank=2,
        var_feat_cols=norm_varlen_feature_columns + tag_column,
        att_dense_units=attention_embedding_size)

    att_second_dense_head_2 = Dense(attention_embedding_size, activation="relu", name="att_second_dense_head_2")

    # 2.5 Attention operation
    att_unit_layer_head_1 = AttentionUTPM(att_layer_num="first", att_dense_unit=attention_embedding_size)
    att_unit_layer_head_2 = AttentionUTPM(att_layer_num="second", att_dense_unit=attention_embedding_size)

    # 3 Computation
    # 3.1 Prepare embedding data
    # 3.1.1 norm sparse feature
    sparse_embedding_list = embedding_lookup(nrom_embedding_dict, features, norm_sparse_feature_columns, to_list=True)

    # # 3.1.2 norm varlen feat - except tags
    norm_varlen_feat_emb_data_dict = varlen_embedding_lookup(nrom_embedding_dict, features, norm_varlen_feature_columns)
    norm_varlen_feat_pool_dict = get_varlen_weight_pooling_dict(norm_varlen_feat_emb_data_dict,
                                                                norm_varlen_feature_columns)

    # # 3.1.3 item - tag - special, the mask data have been set to zero, but the output dimension was not changed.
    tag_emb_data_dict = varlen_embedding_lookup(tag_embedding_dict, features, tag_column)
    tag_pool_dict = get_varlen_weight_pooling_dict(tag_emb_data_dict, tag_column)
    tag_column_name = tag_column[0].name
    tag_embbeding = tag_pool_dict[tag_column_name]

    # 3.2 Attention - first layer
    # 3.2.1 head 1
    first_layer_output_head_1 = get_first_layer_att_result_list(norm_varlen_feat_pool_dict,
                                                                att_first_layer_head_1_dict,
                                                                norm_varlen_feature_columns,
                                                                att_unit_layer_head_1)
    # 3.2.2 head 2
    first_layer_output_head_2 = get_first_layer_att_result_list(tag_pool_dict,
                                                                att_first_layer_head_2_dict,
                                                                tag_column,
                                                                att_unit_layer_head_2)

    # 3.3 Attention - second layer
    # 3.3.1 head_1
    second_layer_output_head_1 = get_second_layer_att_result(att_second_dense_head_1, att_unit_layer_head_1,
                                                             sparse_embedding_list, first_layer_output_head_1)

    # 3.3.2 head_2
    second_layer_output_head_2 = get_second_layer_att_result(att_second_dense_head_2, att_unit_layer_head_2,
                                                             sparse_embedding_list, first_layer_output_head_2)

    # 3.4 Cross feature layer
    concat_data = Concatenate(axis=1)([second_layer_output_head_1, second_layer_output_head_2])

    # 3.4.1 linear op
    linear_op_result = LinerOpLayerUTPM()(concat_data)

    # 3.4.2 cross op
    cross_op_result = CrossOpLayerUTPM(feature_embedding_dim, cross_op_layer_vec_emb_size)(concat_data)

    # 3.5 Fully connect layer
    fc_linear = Dense(dense_units, activation="relu")(linear_op_result)

    fc_cross = Dense(dense_units, activation="relu")(cross_op_result)

    concat_two_fc = Concatenate(axis=1)([fc_linear, fc_cross])

    # 3.6 user embeddings
    user_embeds = Dense(feature_embedding_dim, activation='relu')(concat_two_fc)

    # 4 special layers
    final_output = JointLossUTPM()([user_embeds, tag_embbeding])

    inputs_list = list(features.values())
    model = Model(inputs=inputs_list, outputs=final_output)

    model.__setattr__("inputs_dict", features)
    model.__setattr__("user_embedding", user_embeds)

    return model


def get_second_layer_att_result(att_second_dense_head: Dense, att_unit_layer: AttentionUTPM,
                                sparse_embedding_list, first_layer_head_output):
    first_layer_output = Concatenate(axis=1)(sparse_embedding_list + first_layer_head_output)
    first_layer_output_att_score = att_second_dense_head(first_layer_output)
    scend_layer_output = att_unit_layer([first_layer_output, first_layer_output_att_score])
    flat_output = Flatten()(scend_layer_output)

    return flat_output


def create_attention_first_layer_head_dict(head_rank: int,
                                           var_feat_cols: List[SparseFeat or VarLenSparseFeat],
                                           att_dense_units: int = 8):
    attenion_first_head_dict = OrderedDict()

    for feat in var_feat_cols:
        feat_name = feat.name
        attenion_first_head_dict[feat_name] = Dense(att_dense_units,
                                                    activation="relu",
                                                    name=feat_name + "_first_layer_dense_head_{}".format(head_rank))

    return attenion_first_head_dict


def get_first_layer_att_result_list(user_varlen_feat_pool_dict: dict,
                                    att_first_head_dict: dict,
                                    feat_cols: List[VarLenSparseFeat or SparseFeat],
                                    attention_unit: AttentionUTPM):
    result_list = []
    for feat in feat_cols:
        feat_name = feat.name
        emb_data = user_varlen_feat_pool_dict[feat_name]
        att_outputs = att_first_head_dict[feat_name](emb_data)
        result = attention_unit([emb_data, att_outputs])
        result_list.append(result)

    return result_list


def get_varlen_weight_pooling_dict(varlen_embedding_data_dict: Dict[str, Embedding],
                                   varlen_feature_columns: List[VarLenSparseFeat or SparseFeat]):
    results_dict = OrderedDict()
    for fc in varlen_feature_columns:
        feature_name = fc.name

        seq_embed_list = varlen_embedding_data_dict[feature_name]
        vec = SequencePoolingLayer(supports_masking=True)(seq_embed_list)
        results_dict[feature_name] = vec

    return results_dict
