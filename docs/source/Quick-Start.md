# Quick-Start

## Installation Guide
Now `deepctr` is available for python `2.7 `and `3.5, 3.6, 3.7`.  
`deepctr` depends on tensorflow, you can specify to install the cpu version or gpu version through `pip`.

### CPU version

```bash
$ pip install deepctr[cpu]
```
### GPU version

```bash
$ pip install deepctr[gpu]
```
## Getting started: 4 steps to DeepCTR


### Step 1: Import model


```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat,get_feature_names

data = pd.read_csv('./criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I'+str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)
target = ['label']
```
    


### Step 2: Simple preprocessing


Usually we have two methods to encode the sparse categorical feature for embedding

- Label Encoding: map the features to integer value from 0 ~ len(#unique) - 1
  ```python
  for feat in sparse_features:
      lbe = LabelEncoder()
      data[feat] = lbe.fit_transform(data[feat])
  ```
- Hash Encoding: map the features to a fix range,like 0 ~ 9999.We have 2 methods to do that:
  - Do feature hashing before training
    ```python
    for feat in sparse_features:
        lbe = HashEncoder()
        data[feat] = lbe.transform(data[feat])
    ```
  - Do feature hashing on the fly in training process 

    We can do feature hashing by setting `use_hash=True` in `SparseFeat` or `VarlenSparseFeat` in Step3.


And for dense numerical features,they are usually  discretized to buckets,here we use normalization.

```python
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])
```


### Step 3: Generate feature columns

For sparse features, we transform them into dense vectors by embedding techniques.
For dense numerical features, we concatenate them to the input tensors of fully connected layer. 

And for varlen(multi-valued) sparse features,you can use [VarlenSparseFeat](./Features.html#varlensparsefeat).  Visit [examples](./Examples.html#multi-value-input-movielens) of using `VarlenSparseFeat`

- Label Encoding
```python
sparse_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)]
dense_feature_columns = [DenseFeat(feat, 1)
                      for feat in dense_features]
```
- Feature Hashing on the fly
```python
sparse_feature_columns = [SparseFeat(feat, vocabulary_size=1e6,embedding_dim=4,use_hash=True)
                           for i,feat in enumerate(sparse_features)]#The dimension can be set according to data
dense_feature_columns = [DenseFeat(feat, 1)
                      for feat in dense_features]
```
- generate feature columns
```python
dnn_feature_columns = sparse_feature_columns + dense_feature_columns
linear_feature_columns = sparse_feature_columns + dense_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

```
### Step 4: Generate the training samples and train the model

```python
train, test = train_test_split(data, test_size=0.2)

train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}


model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)

```
You can check the full code [here](./Examples.html#classification-criteo).








