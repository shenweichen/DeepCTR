# Examples


## Classification: Criteo 

The Criteo Display Ads dataset is for the purpose of predicting ads 
click-through rate. It has 13 integer features and
26 categorical features where each category has a high cardinality.

![image](../pics/criteo_sample.png)

In this example,we simply normailize the integer feature between 0 and 1,you
can try other transformation technique like log normalization or discretization.Then we use `SingleFeat` to generate feature config dict for sparse features and dense features.

``SingleFeat`` is a namedtuple with signature ``SingleFeat(name, dimension)``

- name : feature name
- dimension : number of unique feature values for sprase feature, any value for dense feature.

This example shows how to use ``DeepFM`` to solve a simple binary classification task. You can get the demo data [criteo_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/criteo_sample.txt)
and run the following codes.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM
from deepctr import SingleFeat

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]

    # 4.Define Model,train,predict and evaluate
    model = DeepFM({"sparse": sparse_feature_list,
                    "dense": dense_feature_list}, final_activation='sigmoid')
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deepctr.models import DeepFM
from deepctr import VarLenFeat,SingleFeat

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
    sparse_feat_list = [SingleFeat(feat,data[feat].nunique()) for feat in sparse_features]

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[feat.name].values for feat in sparse_feat_list]
    test_model_input = [test[feat.name].values for feat in sparse_feat_list]
    # 4.Define Model,train,predict and evaluate
    model = DeepFM({"sparse": sparse_feat_list},
                   final_activation='linear')
    model.compile("adam", "mse", metrics=['mse'],)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2,)
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
2. Generate config of sequence feature with `deepctr.utils.VarLenFeat`

``VarLenFeat`` is a namedtuple with signature ``VarLenFeat(name, dimension, maxlen, combiner)``

- name : feature name,if it is already used in sparse_feature_dim,then a shared embedding mechanism will be used.
- dimension : number of unique feature values
- maxlen : maximum length of this feature for all samples
- combiner : pooling method,can be ``sum``,``mean`` or ``max``

Now multi-value input is avaliable for `AFM,AutoInt,DCN,DeepFM,FNN,NFM,PNN,xDeepFM`,for `DIN` please read the example in [run_din.py](https://github.com/shenweichen/DeepCTR/blob/master/examples/run_din.py)  
This example shows how to use ``DeepFM`` with sequence(multi-value) feature. You can get the demo data 
[movielens_sample.txt](https://github.com/shenweichen/DeepCTR/tree/master/examples/movielens_sample.txt) and run the following codes.

```python
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr import VarLenFeat, SingleFeat


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
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post',)

# 2.count #unique features for each sparse field and generate feature config for sequence feature

sparse_feat_list = [SingleFeat(feat, data[feat].nunique())
                    for feat in sparse_features]
sequence_feature = [VarLenFeat('genres', len(
    key2index) + 1, max_len, 'mean')]  # Notice : value 0 is for padding for sequence input feature

# 3.generate input data for model
sparse_input = [data[feat.name].values for feat in sparse_feat_list]
dense_input = []
sequence_input = [genres_list]
model_input = sparse_input + dense_input + \
    sequence_input  # make sure the order is right

# 4.Define Model,compile and train
model = DeepFM({"sparse": sparse_feat_list,
                "sequence": sequence_feature}, final_activation='linear')

model.compile("adam", "mse", metrics=['mse'],)
history = model.fit(model_input, data[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2,)
```