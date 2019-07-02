# Examples


## Classification: Criteo 

The Criteo Display Ads dataset is for the purpose of predicting ads 
click-through rate. It has 13 integer features and
26 categorical features where each category has a high cardinality.

![image](../pics/criteo_sample.png)

In this example,we simply normailize the dense feature between 0 and 1,you
can try other transformation technique like log normalization or discretization.Then we use `SparseFeat` and `DenseFeat` to generate feature columns  for sparse features and dense features.

``SparseFeat`` is a namedtuple with signature ``SparseFeat(name, dimension, use_hash, dtype, embedding_name,embedding)``

- name : feature name
- dimension : number of unique feature values for sprase feature,hashing space when hash_flag=True, any value for dense feature.
- use_hash : defualt `False`.If `True` the input will be hashed to space of size `dimension`.
- dtype : default `float32`.dtype of input tensor.
- embedding_name : default `None`. If None, the embedding_name` will be same as `name`.
- embedding : default `True`.If `False`, the feature will not be embeded to a dense vector.

This example shows how to use ``DeepFM`` to solve a simple binary classification task. You can get the demo data [criteo_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/criteo_sample.txt)
and run the following codes.

```python
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[name] for name in fixlen_feature_names]

    test_model_input = [test[name] for name in fixlen_feature_names]

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
```

## Classification: Criteo with feature hashing on the fly
This example shows how to use ``DeepFM`` to solve a simple binary classification task using feature hashing. You can get the demo data [criteo_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/criteo_sample.txt)
and run the following codes.

```python
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat,get_fixlen_feature_names

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.do simple Transformation for dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.set hashing space for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, 1000, use_hash=True, dtype='string')  # since the input is string
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                          for feat in dense_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns, )

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[name] for name in fixlen_feature_names]

    test_model_input = [test[name] for name in fixlen_feature_names]


    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns,dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat,get_fixlen_feature_names

if __name__ == "__main__":

    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[name].values for name in fixlen_feature_names]
    test_model_input = [test[name].values for name in fixlen_feature_names]
    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(mean_squared_error(
        test[target].values, pred_ans), 4))


```
## Multi-value Input : Movielens
----------------------------------

The MovieLens data has been used for personalized tag recommendation,which
contains 668, 953 tag applications of users on movies.
Here is a small fraction of data include  sparse fields and a multivalent field.

![image](../pics/movielens_sample_with_genres.png)

There are 2 additional steps to use DeepCTR with sequence feature input.

1. Generate the paded and encoded sequence feature  of sequence input feature(**value 0 is for padding**).
2. Generate config of sequence feature with `deepctr.inputs.VarLenSparseFeat`

``VarLenSparseFeat`` is a namedtuple with signature ``VarLenSparseFeat(name, dimension, maxlen, combiner, use_hash, dtype, embedding_name,embedding)``

- name : feature name,if it is already used in sparse_feature_dim,then a shared embedding mechanism will be used.
- dimension : number of unique feature values
- maxlen : maximum length of this feature for all samples
- combiner : pooling method,can be ``sum``,``mean`` or ``max``
- use_hash : defualt `False`.if `True` the input will be hashed to space of size `dimension`.
- dtype : default `float32`.dtype of input tensor.
- embedding_name : default `None`. If None, the embedding_name` will be same as `name`.
- embedding : default `True`.If `False`, the feature will not be embeded to a dense vector.


Now multi-value input is avaliable for `AFM,AutoInt,DCN,DeepFM,FNN,NFM,PNN,xDeepFM,CCPM,FGCNN`,for `DIN,DIEN,DSIN` please read the example in [run_din.py](https://github.com/shenweichen/DeepCTR/blob/master/examples/run_din.py),[run_dien.py](https://github.com/shenweichen/DeepCTR/blob/master/examples/run_dien.py) and [run_dsin.py](https://github.com/shenweichen/DeepCTR/blob/master/examples/run_dsin.py) 

This example shows how to use ``DeepFM`` with sequence(multi-value) feature. You can get the demo data 
[movielens_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/movielens_sample.txt) and run the following codes.

```python
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

```

## Multi-value Input : Movielens with feature hashing on the fly
----------------------------------
```python
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
```