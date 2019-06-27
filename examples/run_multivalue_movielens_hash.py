import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_fixlen_feature_names

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
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str, value=0)

# 2.set hashing space for each sparse field and generate feature config for sequence feature

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique() * 5, use_hash=True, dtype='string')
                          for feat in sparse_features]
varlen_feature_columns = [VarLenSparseFeat('genres', 100, max_len, 'mean', use_hash=True,
                                           dtype="string")]  # Notice : value 0 is for padding for sequence input feature
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
fixlen_input = [data[name].values for name in feature_names]
varlen_input = [genres_list]

model_input = fixlen_input + varlen_input # make sure the order is right

# 4.Define Model,compile and train
model = DeepFM(linear_feature_columns,dnn_feature_columns, task='regression')

model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
