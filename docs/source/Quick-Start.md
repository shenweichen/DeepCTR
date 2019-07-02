# Quick-Start

## Installation Guide
Now `deepctr` is available for python `2.7 `and `3.4,  3.5, 3.6`.  
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
from deepctr.inputs import  SparseFeat, DenseFeat,get_fixlen_feature_names

data = pd.read_csv('./criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I'+str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0,)
target = ['label']
```
    


### Step 2: Simple preprocessing


Usually there are two simple way to encode the sparse categorical feature for embedding

- Label Encoding: map the features to integer value from 0 ~ len(#unique) - 1
  ```python
  for feat in sparse_features:
      lbe = LabelEncoder()
      data[feat] = lbe.fit_transform(data[feat])
  ```
- Hash Encoding: map the features to a fix range,like 0 ~ 9999.Here is 2 way to do that:
  - Do feature hashing before training
    ```python
    for feat in sparse_features:
        lbe = HashEncoder()
        data[feat] = lbe.transform(data[feat])
    ```
  - Do feature hashing on the flay in training process 

    We can do feature hasing throug setting `use_hash=True` in `SparseFeat` or `VarlenSparseFeat` in Step3.


And for dense numerical features,they are usually  discretized to buckets,here we use normalization.

```python
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])
```


### Step 3: Generate feature columns

For sparse features, we transform them into dense vectors by embedding techniques.
For dense numerical features, we concatenate them to the input tensors of fully connected layer.

- Label Encoding
```python
sparse_feature_columns = [SparseFeat(feat, data[feat].nunique())
                        for feat in sparse_features]
dense_feature_columns = [DenseFeat(feat, 1)
                      for feat in dense_features]
```
- Feature Hashing on the fly
```python
sparse_feature_columns = [SparseFeat(feat, dimension=1e6,use_hash=True) for feat in sparse_features]#The dimension can be set according to data
dense_feature_columns = [DenseFeat(feat, 1)
                      for feat in dense_features]
```
- generate feature columns
```python
dnn_feature_columns = sparse_feature_columns + dense_feature_columns
linear_feature_columns = sparse_feature_columns + dense_feature_columns

feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

```
### Step 4: Generate the training samples and train the model

There are two rules here that we must follow

  - The `SparseFeat` and `DenseFeat`  are placed in front of the `VarlenSparseFeat`.
  - The order of the feature we fit into the model must be consistent with the order of the feature config list.


```python
train, test = train_test_split(data, test_size=0.2)
train_model_input = [train[name] for name in feature_names]

test_model_input = [test[name] for name in feature_names]


model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)

```
You can check the full code [here](./Examples.html#classification-criteo).








