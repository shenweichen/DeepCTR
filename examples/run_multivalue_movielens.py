import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr.utils import VarLenFeature


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            key2index[key] = len(key2index)
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
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post',)

# 2.count #unique features for each sparse field and generate feature config for sequence feature

sparse_feature_dim = {feat: data[feat].nunique() for feat in sparse_features}
sequence_feature = [VarLenFeature('genres', len(key2index), max_len, 'mean')]

# 3.generate input data for model
sparse_input = [data[feat].values for feat in sparse_feature_dim]
dense_input = []
sequence_input = [genres_list]
sequence_length_input = [genres_length]
model_input = sparse_input + dense_input + sequence_input + \
    sequence_length_input  # make sure the order is right

# 4.Define Model,compile and train
model = DeepFM({"sparse": sparse_feature_dim, "dense": [],
                "sequence": sequence_feature}, final_activation='linear')

model.compile("adam", "mse", metrics=['mse'],)
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2,)
