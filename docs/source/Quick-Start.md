# Quick-Start

## Installation Guide
### CPU version
Install `deepctr` package is through `pip` 
```bash
$ pip install deepctr
```
### GPU version
If you have a `tensorflow-gpu` on your local machine,make sure its version is
**`tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*`**  
Then,use the following command to install
```bash
$ pip install deepctr --no-deps
```
## Getting started: 4 steps to DeepCTR


### Step 1: Import model


```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from deepctr import SingleFeat

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
- Hash Encoding: map the features to a fix range,like 0 ~ 9999

And for dense numerical features,they are usually  discretized to buckets,here we use normalization.

```python
for feat in sparse_features:
    lbe = LabelEncoder()# or Hash
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])
```


### Step 3: Generate feature config dict

Here, for sparse features, we transform them into dense vectors by embedding techniques.
For dense numerical features, we add a dummy index like LIBFM.
That is to say, all dense features under the same field share the same embedding vector.
In some implementations, the dense feature is concatened to the input embedding vectors of the deep network, you can modify the code yourself.

```python
sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                        for feat in sparse_features]
dense_feature_list = [SingleFeat(feat, 0)
                      for feat in dense_features]
```

### Step 4: Generate the training samples and train the model

There are two rules here that we must follow

  - The sparse features are placed in front of the dense features.
  - The order of the feature we fit into the model must be consistent with the order of the feature config list.


```python
train, test = train_test_split(data, test_size=0.2)
train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
    [train[feat.name].values for feat in dense_feature_list]
test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
    [test[feat.name].values for feat in dense_feature_list]

model = DeepFM({"sparse": sparse_feature_list,
                "dense": dense_feature_list}, final_activation='sigmoid')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)

```
You can check the full code [here](./Examples.html#classification-criteo).








