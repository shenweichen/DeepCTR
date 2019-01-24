# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Gai K, Zhu X, Li H, et al. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction[J]. arXiv preprint arXiv:1704.05194, 2017.(https://arxiv.org/abs/1704.05194)
"""
from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Activation,  Reshape,  add, dot
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.regularizers import l2


def MLR(region_feature_dim_dict, base_feature_dim_dict={"sparse": [], "dense": []}, region_num=4,
        l2_reg_linear=1e-5,
        init_std=0.0001, seed=1024, final_activation='sigmoid',
        bias_feature_dim_dict={"sparse": [], "dense": []}):
    """Instantiates the Mixed Logistic Regression/Piece-wise Linear Model.

    :param region_feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param base_feature_dim_dict: dict or None,to indicate sparse field and dense field of base learner.if None, it is same as region_feature_dim_dict
    :param region_num: integer > 1,indicate the piece number
    :param l2_reg_linear: float. L2 regularizer strength applied to weight
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param bias_feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :return: A Keras model instance.
    """

    if region_num <= 1:
        raise ValueError("region_num must > 1")
    if not isinstance(region_feature_dim_dict,
                      dict) or "sparse" not in region_feature_dim_dict or "dense" not in region_feature_dim_dict:
        raise ValueError(
            "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

    same_flag = False
    if base_feature_dim_dict == {"sparse": [], "dense": []}:
        base_feature_dim_dict = region_feature_dim_dict
        same_flag = True

    region_sparse_input, region_dense_input, base_sparse_input, base_dense_input, bias_sparse_input, bias_dense_input = get_input(
        region_feature_dim_dict, base_feature_dim_dict, bias_feature_dim_dict, same_flag)
    region_embeddings, base_embeddings, bias_embedding = get_embedding(
        region_num, region_feature_dim_dict, base_feature_dim_dict, bias_feature_dim_dict, init_std, seed, l2_reg_linear)

    if same_flag:

        base_dense_input_ = region_dense_input

        base_sparse_input_ = region_sparse_input

    else:

        base_dense_input_ = base_dense_input

        base_sparse_input_ = base_sparse_input

    region_dense_feature_num = len(region_feature_dim_dict['dense'])
    region_sparse_feature_num = len(region_feature_dim_dict['sparse'])
    base_dense_feature_num = len(base_feature_dim_dict['dense'])
    base_sparse_feature_num = len(base_feature_dim_dict['sparse'])
    bias_dense_feature_num = len(bias_feature_dim_dict['dense'])
    bias_sparse_feature_num = len(bias_feature_dim_dict['sparse'])

    if region_dense_feature_num > 1:
        region_dense_logits_ = [Dense(1, )(Concatenate()(region_dense_input)) for _ in
                                range(region_num)]
    elif region_dense_feature_num == 1:
        region_dense_logits_ = [Dense(1, )(region_dense_input[0]) for _ in
                                range(region_num)]

    if base_dense_feature_num > 1:
        base_dense_logits = [Dense(1, )(Concatenate()(base_dense_input_))for _ in
                             range(region_num)]
    elif base_dense_feature_num == 1:
        base_dense_logits = [Dense(1, )(base_dense_input_[0])for _ in
                             range(region_num)]

    if region_dense_feature_num > 0 and region_sparse_feature_num == 0:
        region_logits = Concatenate()(region_dense_logits_)
    elif region_dense_feature_num == 0 and region_sparse_feature_num > 0:
        region_sparse_logits = [
            add([region_embeddings[j][i](region_sparse_input[i])
                 for i in range(region_sparse_feature_num)])
            if region_sparse_feature_num > 1 else region_embeddings[j][0](region_sparse_input[0])
            for j in range(region_num)]
        region_logits = Concatenate()(region_sparse_logits)

    else:
        region_sparse_logits = [
            add([region_embeddings[j][i](region_sparse_input[i])
                 for i in range(region_sparse_feature_num)])
            for j in range(region_num)]
        region_logits = Concatenate()(
            [add([region_sparse_logits[i], region_dense_logits_[i]]) for i in range(region_num)])

    if base_dense_feature_num > 0 and base_sparse_feature_num == 0:
        base_logits = base_dense_logits
    elif base_dense_feature_num == 0 and base_sparse_feature_num > 0:
        base_sparse_logits = [add(
            [base_embeddings[j][i](base_sparse_input_[i]) for i in range(base_sparse_feature_num)]) if base_sparse_feature_num > 1 else base_embeddings[j][0](base_sparse_input_[0])
            for j in range(region_num)]
        base_logits = base_sparse_logits
    else:
        base_sparse_logits = [add(
            [base_embeddings[j][i](base_sparse_input_[i]) for i in range(base_sparse_feature_num)]) if base_sparse_feature_num > 1 else base_embeddings[j][0](base_sparse_input_[0])
            for j in range(region_num)]
        base_logits = [add([base_sparse_logits[i], base_dense_logits[i]])
                       for i in range(region_num)]

    # Dense(self.region_num, activation='softmax')(final_logit)
    region_weights = Activation("softmax")(region_logits)
    learner_score = Concatenate()(
        [Activation(final_activation, name='learner' + str(i))(base_logits[i]) for i in range(region_num)])
    final_logit = dot([region_weights, learner_score], axes=-1)

    if bias_dense_feature_num + bias_sparse_feature_num > 0:

        if bias_dense_feature_num > 1:
            bias_dense_logits = Dense(1,)(Concatenate()(bias_dense_input))
        elif bias_dense_feature_num == 1:
            bias_dense_logits = Dense(1,)(bias_dense_input[0])
        else:
            pass

        if bias_sparse_feature_num > 1:
            bias_cate_logits = add([bias_embedding[i](bias_sparse_input[i])
                                    for i, feat in enumerate(bias_feature_dim_dict['sparse'])])
        elif bias_sparse_feature_num == 1:
            bias_cate_logits = bias_embedding[0](bias_sparse_input[0])
        else:
            pass

        if bias_dense_feature_num > 0 and bias_sparse_feature_num > 0:
            bias_logits = add([bias_dense_logits, bias_cate_logits])
        elif bias_dense_feature_num > 0:
            bias_logits = bias_dense_logits
        else:
            bias_logits = bias_cate_logits

        bias_prob = Activation('sigmoid')(bias_logits)
        final_logit = dot([final_logit, bias_prob], axes=-1)

    output = Reshape([1])(final_logit)
    model = Model(inputs=region_sparse_input + region_dense_input+base_sparse_input +
                  base_dense_input+bias_sparse_input+bias_dense_input, outputs=output)
    return model


def get_input(region_feature_dim_dict, base_feature_dim_dict, bias_feature_dim_dict, same_flag):
    region_sparse_input = [Input(shape=(1,), name='region_sparse_' + str(i)+"-"+feat.name)
                           for i, feat in enumerate(region_feature_dim_dict["sparse"])]
    region_dense_input = [Input(shape=(1,), name='region_dense_' + str(i)+"-"+feat.name)
                          for i, feat in enumerate(region_feature_dim_dict["dense"])]
    if same_flag == True:
        base_sparse_input = []
        base_dense_input = []
    else:
        base_sparse_input = [Input(shape=(1,), name='base_sparse_' + str(i) + "-" + feat.name) for i, feat in
                             enumerate(base_feature_dim_dict["sparse"])]
        base_dense_input = [Input(shape=(1,), name='base_dense_' + str(i) + "-" + feat.name) for i, feat in
                            enumerate(base_feature_dim_dict['dense'])]

    bias_sparse_input = [Input(shape=(1,), name='bias_cate_' + str(i) + "-" + feat.name) for i, feat in
                         enumerate(bias_feature_dim_dict['sparse'])]
    bias_dense_input = [Input(shape=(1,), name='bias_continuous_' + str(i) + "-" + feat.name) for i, feat in
                        enumerate(bias_feature_dim_dict['dense'])]
    return region_sparse_input, region_dense_input, base_sparse_input, base_dense_input, bias_sparse_input, bias_dense_input


def get_embedding(region_num, region_feature_dim_dict, base_feature_dim_dict, bias_feature_dim_dict, init_std, seed, l2_reg_linear):

    region_embeddings = [[Embedding(feat.dimension, 1, embeddings_initializer=TruncatedNormal(stddev=init_std, seed=seed+j), embeddings_regularizer=l2(l2_reg_linear),
                                    name='region_emb_' + str(j)+'_' + str(i)) for
                          i, feat in enumerate(region_feature_dim_dict['sparse'])] for j in range(region_num)]
    base_embeddings = [[Embedding(feat.dimension, 1,
                                  embeddings_initializer=TruncatedNormal(stddev=init_std, seed=seed + j), embeddings_regularizer=l2(l2_reg_linear),
                                  name='base_emb_' + str(j) + '_' + str(i)) for
                        i, feat in enumerate(base_feature_dim_dict['sparse'])] for j in range(region_num)]
    bias_embedding = [Embedding(feat.dimension, 1, embeddings_initializer=TruncatedNormal(stddev=init_std, seed=seed), embeddings_regularizer=l2(l2_reg_linear),
                                name='embed_bias' + '_' + str(i)) for
                      i, feat in enumerate(bias_feature_dim_dict['sparse'])]

    return region_embeddings, base_embeddings, bias_embedding
