import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr.models import DeepFM
from deepctr.utils import VarLenFeat, SingleFeat

data = pd.read_csv("./movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip", ]

data[sparse_features] = data[sparse_features].astype('string')
target = ['rating']

# 1.Use hashing encoding on the fly for sparse features,and process sequence features

genres_list = list(map(lambda x: x.split('|'), data['genres'].values))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)

# Notice : padding=`post`
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype='string', value=0)

# 2.set hashing space for each sparse field and generate feature config for sequence feature

sparse_feat_list = [SingleFeat(feat, data[feat].nunique() * 5, hash_flag=True, dtype='string')
                    for feat in sparse_features]
sequence_feature = [VarLenFeat('genres', 100, max_len, 'mean', hash_flag=True,
                               dtype="string")]  # Notice : value 0 is for padding for sequence input feature

# 3.generate input data for model
sparse_input = [data[feat.name].values for feat in sparse_feat_list]
dense_input = []
sequence_input = [genres_list]
model_input = sparse_input + dense_input + \
              sequence_input  # make sure the order is right

# 4.Define Model,compile and train
model = DeepFM({"sparse": sparse_feat_list,
                "sequence": sequence_feature}, final_activation='linear')

model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
