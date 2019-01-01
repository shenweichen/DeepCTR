# Examples


## Classification: Criteo 

The Criteo Display Ads dataset is for the purpose of predicting ads 
click-through rate. It has 13 integer features and
26 categorical features where each category has a high cardinality.

![image](../pics/criteo_sample.png)

In this example,we simply normailize the integer feature between 0 and 1,you
can try other transformation technique like log normalization or discretization.

This example shows how to use ``DeepFM`` to solve a simple binary classification task. You can get the demo data [criteo_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/criteo_sample.txt)
and run the following codes.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from deepctr.models import DeepFM


data = pd.read_csv('./criteo_sample.txt')

sparse_features  = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I'+str(i) for i in range(1,14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)

target = ['label']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 2.count #unique features for each sparse field,and record dense feature field name

sparse_feature_dict = {feat: data[feat].nunique() for feat in sparse_features}
dense_feature_list = dense_features

# 3.generate input data for model

model_input = [data[feat].values for feat in sparse_feature_dict] + [data[feat].values for feat in dense_feature_list]

#4.Define Model,compile and


model = DeepFM({"sparse": sparse_feature_dict, "dense": dense_feature_list}, final_activation='sigmoid')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=1, verbose=2, validation_split=0.2,)
```

## Regression: Movielens

The MovieLens data has been used for personalized tag recommendation,which
contains 668, 953 tag applications of users on movies.
Here is a small fraction of data include  only sparse field.

![image](../pics/movielens_sample.png)


This example shows how to use ``DeepFM`` to solve a simple binary regression task. You can get the demo data 
[movielens_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/movielens_sample.txt) and run the following codes.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from deepctr.models import DeepFM


data = pd.read_csv("./movielens_sample.txt")
sparse_features = [ "movie_id","user_id","gender","age","occupation","zip"]
target = ['rating']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
#2.count #unique features for each sparse field
sparse_feature_dim = {feat:data[feat].nunique() for feat in sparse_features}
#3.generate input data for model
model_input = [data[feat].values for feat in sparse_feature_dim]
#4.Define Model,compile and train
model = DeepFM({"sparse":sparse_feature_dim,"dense":[]},final_activation='linear')

model.compile("adam","mse",metrics=['mse'],)
history = model.fit(model_input,data[target].values,
            batch_size=256,epochs=10,verbose=2,validation_split=0.2,)
```
## Multi-value Input : Movielens
----------------------------------

The MovieLens data has been used for personalized tag recommendation,which
contains 668, 953 tag applications of users on movies.
Here is a small fraction of data include  sparse fields and a multivalent field.

![image](../pics/movielens_sample_with_genres.png)

There are 2 additional steps to use DeepCTR with sequence feature input.

1. Generate the paded and encoded sequence feature and valid length of sequence feature.
2. Generate config of sequence feature with `deepctr.utils.VarLenFeature`

``VarLenFeature`` is a namedtuple with signature ``VarLenFeature(name, dimension, maxlen, combiner)``

- name : feature name,if it is already used in sparse_feature_dim,then a shared embedding mechanism will be used.
- dimension : number of unique features
- maxlen : maximum length of this feature for all samples
- combiner : pooling method,can be ``sum``,``mean`` or ``max``

Now multi-value input is avaliable for `AFM,AutoInt,DCN,DeepFM,FNN,NFM,PNN,xDeepFM`,for `DIN` please read the example in [run_din.py](https://github.com/shenweichen/DeepCTR/blob/master/examples/run_din.py)  
This example shows how to use ``DeepFM`` with sequence(multi-value) feature. You can get the demo data 
[movielens_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/movielens_sample.txt) and run the following codes.

```python
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
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
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post',)# Notice : padding='post'

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
```