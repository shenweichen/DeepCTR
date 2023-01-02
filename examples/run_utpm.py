import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models.utpm import UTPM


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice: input value 0 is a special "padding"
            key2index[key] = len(key2index) + 1

    result = list(map(lambda _: key2index[_], key_ans))
    return result


def padding_column(x, maxlen=5):
    x = x[: maxlen]

    x = x + [0] * (maxlen - len(x))

    return x


def get_train_test_data(df, feature_names):
    model_input = {name: df[name] for name in feature_names}

    for column in ["tags", "varlen_feat"]:
        res_matrix = np.array([[0] * 5] * len(df[column]))

        tmp_matrix = df[column].values
        for i in range(len(tmp_matrix)):
            res_matrix[i] = tmp_matrix[i]

        model_input[column] = res_matrix

    return model_input


if __name__ == '__main__':
    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["user_id",
                       "gender", "age", "occupation", "zip"]
    target = ['label']

    # 1 Preprocess
    # 1.1 Label Encoding for sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_max_len = len(key2index)

    data["tags"] = genres_list
    data["tags"] = data["tags"].apply(lambda x: padding_column(x, maxlen=5))

    data["label"] = data["rating"].apply(lambda x: 1 if x > 3 else 0)

    data = data[["user_id", "gender", "age", "occupation", "zip", "tags", "label"]]

    # 2 Specify the parameters for Embedding
    feature_embedding_dim = 8

    sparse_feat_max_len = {feat: data[feat].max() + 1 for feat in sparse_features}
    norm_sparse_feature_columns = [
        SparseFeat(feat, vocabulary_size=sparse_feat_max_len[feat], embedding_dim=feature_embedding_dim)
        for feat in sparse_features]
    # We suppose there is a varlen feature except tag column in our dataset, so we duplicate "tags" as the different
    # varlen feature in order to test the model
    data["varlen_feat"] = data["tags"]

    norm_varlen_feature_columns = [
        VarLenSparseFeat(
            SparseFeat('varlen_feat', vocabulary_size=genres_max_len + 1, embedding_dim=feature_embedding_dim),
            maxlen=5)]

    tag_column = [
        VarLenSparseFeat(
            SparseFeat('tags', vocabulary_size=genres_max_len + 1, embedding_dim=feature_embedding_dim),
            maxlen=5)
    ]

    # 3 Generate input data for model
    feature_names = get_feature_names(norm_sparse_feature_columns + norm_varlen_feature_columns + tag_column)

    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    train_model_input = get_train_test_data(train, feature_names)
    test_model_input = get_train_test_data(test, feature_names)

    # 4 Define model, train, predict and evaluate
    model = UTPM(norm_sparse_feature_columns, norm_varlen_feature_columns, tag_column,
                 feature_embedding_dim=feature_embedding_dim)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['binary_crossentropy'])

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=2, verbose=2, validation_split=0.2)

    # get user embedding vectors
    user_embedding_model = tf.keras.Model(inputs=model.inputs_dict, outputs=model.user_embedding)
    all_data = get_train_test_data(data, feature_names)
    user_embeddings = user_embedding_model.predict(all_data)
    user_vectors = []
    for i in range(len(all_data["user_id"])):
        user_id = all_data["user_id"][i]
        user_embedding_vector = user_embeddings[i]
        user_vectors.append([user_id, list(user_embedding_vector)])
    print(user_vectors)

    # get tag embedding vectors
    tags_embedding_layer = model.get_layer("sparse_seq_emb_tags")
    print(tags_embedding_layer.get_weights())
