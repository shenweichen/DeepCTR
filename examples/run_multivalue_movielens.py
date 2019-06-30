import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, VarLenSparseFeat,get_fixlen_feature_names,get_varlen_feature_names


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


data = pd.read_csv("./movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]
target = ['rating']

# 1.Label Encoding for sparse features,and process sequence features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
# preprocess the sequence feature

key2index = {}
genres_list = list(map(split, data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

# 2.count #unique features for each sparse field and generate feature config for sequence feature

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                    for feat in sparse_features]
varlen_feature_columns = [VarLenSparseFeat('genres', len(
    key2index) + 1, max_len, 'mean')]  # Notice : value 0 is for padding for sequence input feature

linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
varlen_feature_names = get_varlen_feature_names(linear_feature_columns+dnn_feature_columns)


# 3.generate input data for model
fixlen_input = [data[name].values for name in fixlen_feature_names]
varlen_input = [genres_list]#varlen_feature_names[0]
model_input = fixlen_input + varlen_input # make sure the order is right

# 4.Define Model,compile and train
model = DeepFM(linear_feature_columns,dnn_feature_columns,task='regression')

model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
