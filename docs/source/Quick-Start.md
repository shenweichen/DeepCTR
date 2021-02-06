# Quick-Start
[![](https://pai-public-data.oss-cn-beijing.aliyuncs.com/EN-pai-dsw.svg)](https://dsw-dev.data.aliyun.com/#/?fileUrl=https://pai-public-data.oss-cn-beijing.aliyuncs.com/deep-ctr/Getting-started-4-steps-to-DeepCTR.ipynb&fileName=Getting-started-4-steps-to-DeepCTR.ipynb)
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
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names

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
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

```
- Feature Hashing on the fly
```python
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=1e6,embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                          for feat in dense_features]
```
- generate feature columns
```python
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

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



## Getting started: 4 steps to DeepCTR Estimator with TFRecord

### Step 1: Import model

```python
import tensorflow as tf

from tensorflow.python.ops.parsing_ops import  FixedLenFeature
from deepctr.estimator.inputs import input_fn_tfrecord
from deepctr.estimator.models import DeepFMEstimator

```

### Step 2: Generate feature columns for linear part and dnn part

```python
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

dnn_feature_columns = []
linear_feature_columns = []

for i, feat in enumerate(sparse_features):
    dnn_feature_columns.append(tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(feat, 1000), 4))
    linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 1000))
for feat in dense_features:
    dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
    linear_feature_columns.append(tf.feature_column.numeric_column(feat))

```
### Step 3: Generate the training samples with TFRecord format

```python
feature_description = {k: FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
feature_description.update(
    {k: FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})
feature_description['label'] = FixedLenFeature(dtype=tf.float32, shape=1)

train_model_input = input_fn_tfrecord('./criteo_sample.tr.tfrecords', feature_description, 'label', batch_size=256,
                                      num_epochs=1, shuffle_factor=10)
test_model_input = input_fn_tfrecord('./criteo_sample.te.tfrecords', feature_description, 'label',
                                     batch_size=2 ** 14, num_epochs=1, shuffle_factor=0)
```

### Step 4: Train and evaluate the model

```python
model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, task='binary')

model.train(train_model_input)
eval_result = model.evaluate(test_model_input)

print(eval_result)
```

You can check the full code [here](./Examples.html#estimator-with-tfrecord-classification-criteo).








