import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]
    target = ['rating']

    # 1 Preprocess
    # 1.1 Label Encoding for sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 1.2 Preprocess the sequence feature
    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)

    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post')

    # 2 Specify the parameters for Embedding
    # 2.1 Sparse features
    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_features]

    # 2.2 Sequence features
    # Notice : value 0 is for padding for sequence input feature
    use_weighted_sequence = False
    if use_weighted_sequence:
        varlen_feature_columns = [
            VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(key2index) + 1, embedding_dim=4), maxlen=max_len,
                             combiner='mean', weight_name='genres_weight')]
    else:
        varlen_feature_columns = [
            VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(key2index) + 1, embedding_dim=4), maxlen=max_len,
                             combiner='mean', weight_name=None)]

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    # 3 Generate input data for model
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    model_input = {name: data[name] for name in sparse_features}
    model_input["genres"] = genres_list
    model_input["genres_weight"] = np.random.randn(data.shape[0], max_len, 1)

    # 4 Define Model, compile and train
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(model_input, data[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
