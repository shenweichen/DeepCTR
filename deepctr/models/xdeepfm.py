
from tensorflow.python.keras.layers import Dense, Embedding, Concatenate, Flatten, add, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2
from deepctr.utils import get_input
from deepctr.layers import PredictionLayer, MLP, CIN


def xDeepFM(feature_dim_dict, embedding_size=8, hidden_size=(256, 256), cin_layer_size=(128, 128, ), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_deep=0, init_std=0.0001, seed=1024, keep_prob=1, activation='relu', final_activation='sigmoid', use_bn=False):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    if not isinstance(feature_dim_dict, dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
        raise ValueError(
            "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")
    sparse_input, dense_input = get_input(feature_dim_dict, None)
    sparse_embedding, linear_embedding, = get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding,
                                                         l2_reg_linear)

    embed_list = [sparse_embedding[i](sparse_input[i])
                  for i in range(len(sparse_input))]
    linear_term = [linear_embedding[i](sparse_input[i])
                   for i in range(len(sparse_input))]
    if len(linear_term) > 1:
        linear_term = add(linear_term)
    elif len(linear_term) > 0:
        linear_term = linear_term[0]
    else:
        linear_term = 0
    if len(dense_input) > 0:
        continuous_embedding_list = list(
            map(Dense(embedding_size, use_bias=False, kernel_regularizer=l2(l2_reg_embedding), ),
                dense_input))
        continuous_embedding_list = list(
            map(Reshape((1, embedding_size)), continuous_embedding_list))
        embed_list += continuous_embedding_list

        dense_input_ = dense_input[0] if len(
            dense_input) == 1 else Concatenate()(dense_input)
        linear_dense_logit = Dense(
            1, activation=None, use_bias=False, kernel_regularizer=l2(l2_reg_linear))(dense_input_)
        linear_term = add([linear_dense_logit, linear_term])

    linear_logit = linear_term

    fm_input = Concatenate(axis=1)(embed_list) if len(
        embed_list) > 1 else embed_list[0]

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, cin_activation,
                       cin_split_half, seed)(fm_input)
        exFM_logit = Dense(1, activation=None,)(exFM_out)

    deep_input = Flatten()(fm_input)
    deep_out = MLP(hidden_size, activation, l2_reg_deep, keep_prob,
                   use_bn, seed)(deep_input)
    deep_logit = Dense(1, use_bias=False, activation=None)(deep_out)

    if len(hidden_size) == 0 and len(cin_layer_size) == 0:  # only linear
        final_logit = linear_logit
    elif len(hidden_size) == 0 and len(cin_layer_size) > 0:  # linear + CIN
        final_logit = add([linear_logit, exFM_logit])
    elif len(hidden_size) > 0 and len(cin_layer_size) == 0:  # linear +　Deep
        final_logit = add([linear_logit, deep_logit])
    elif len(hidden_size) > 0 and len(cin_layer_size) > 0:  # linear + CIN + Deep
        final_logit = add([linear_logit, deep_logit, exFM_logit])
    else:
        raise NotImplementedError

    output = PredictionLayer(final_activation)(final_logit)
    model = Model(inputs=sparse_input + dense_input, outputs=output)
    return model


def get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    sparse_embedding = [Embedding(feature_dim_dict["sparse"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(feature_dim_dict["sparse"])]
    linear_embedding = [Embedding(feature_dim_dict["sparse"][feat], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                      seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(feature_dim_dict["sparse"])]

    return sparse_embedding, linear_embedding
