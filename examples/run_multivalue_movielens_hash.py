import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.feature_column import SparseFeat, VarLenSparseFeat,get_feature_names
from deepctr.models import DeepFM

if __name__ == "__main__":
    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]

    data[sparse_features] = data[sparse_features].astype(str)
    target = ['rating']

    # 1.Use hashing encoding on the fly for sparse features,and process sequence features

    genres_list = list(map(lambda x: x.split('|'), data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)

    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=object, value=0).astype(str)
    # 2.set hashing space for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique() * 5, embedding_dim=4, use_hash=True, dtype='string')
                              for feat in sparse_features]
    varlen_feature_columns = [
        VarLenSparseFeat(SparseFeat('genres', vocabulary_size=100, embedding_dim=4, use_hash=True, dtype="string"),
                         maxlen=max_len, combiner='mean',
                         )]  # Notice : value 0 is for padding for sequence input feature
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in feature_names}
    model_input['genres'] = genres_list

    # 4.Define Model,compile and train
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')

    model.compile("adam", "mse", metrics=['mse'], )
    history = model.fit(model_input, data[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
