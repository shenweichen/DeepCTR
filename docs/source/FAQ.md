# FAQ

## 1. Save or load weights/models
----------------------------------------
To save/load weights,you can write codes just like any other keras models.

```python
model = DeepFM()
model.save_weights('DeepFM_w.h5')
model.load_weights('DeepFM_w.h5')
```

To save/load models,just a little different.

```python
from tensorflow.python.keras.models import  save_model,load_model
model = DeepFM()
save_model(model, 'DeepFM.h5')# save_model, same as before

from deepctr.layers import custom_objects
model = load_model('DeepFM.h5',custom_objects)# load_model,just add a parameter
```
## 2. Set learning rate and use earlystopping
---------------------------------------------------
You can use any models in DeepCTR like a keras model object.
Here is a example of how to set learning rate and earlystopping:

```python
import deepctr
from tensorflow.python.keras.optimizers import Adam,Adagrad
from tensorflow.python.keras.callbacks import EarlyStopping

model = deepctr.models.DeepFM(linear_feature_columns,dnn_feature_columns)
model.compile(Adagrad(0.1024),'binary_crossentropy',metrics=['binary_crossentropy'])

es = EarlyStopping(monitor='val_binary_crossentropy')
history = model.fit(model_input, data[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2,callbacks=[es] )
```


## 3. Get the attentional weights of feature interactions in AFM
--------------------------------------------------------------------------
First,make sure that you have install the latest version of deepctr.

Then,use the following code,the `attentional_weights[:,i,0]` is the `feature_interactions[i]`'s attentional weight of all samples.

```python
import itertools
import deepctr
from deepctr.models import AFM
from deepctr.inputs import get_feature_names,get_varlen_feature_names
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda

model = AFM(linear_feature_columns,dnn_feature_columns)
model.fit(model_input,target)

afmlayer = model.layers[-3]
afm_weight_model = Model(model.input,outputs=Lambda(lambda x:afmlayer.normalized_att_score)(model.input))
attentional_weights = afm_weight_model.predict(model_input,batch_size=4096)

fixlen_names = get_feature_names( dnn_feature_columns)
varlen_names = get_varlen_feature_names(dnn_feature_columns)
feature_interactions = list(itertools.combinations(fixlen_names+varlen_names ,2))
```
## 4. How to extract the embedding vectors in deepfm?
```python
feature_columns = [SparseFeat('user_id',120,),SparseFeat('item_id',60,),SparseFeat('cate_id',60,)]

def get_embedding_weights(dnn_feature_columns,model):
    embedding_dict = {}
    for fc in dnn_feature_columns:
        if hasattr(fc,'embedding_name'):
            if fc.embedding_name is not None:
                name = fc.embedding_name
            else:
                name = fc.name
            embedding_dict[name] = model.get_layer("sparse_emb_"+name).get_weights()[0]
    return embedding_dict
    
embedding_dict = get_embedding_weights(feature_columns,model)

user_id_emb = embedding_dict['user_id']
item_id_emb = embedding_dict['item_id']
```

## 5. How to add a long dense feature vector as a input to the model?
```python
from deepctr.models import DeepFM
from deepctr.inputs import DenseFeat,SparseFeat,get_feature_names
import numpy as np

feature_columns = [SparseFeat('user_id',120,),SparseFeat('item_id',60,),DenseFeat("pic_vec",5)]
fixlen_feature_names = get_feature_names(feature_columns)

user_id = np.array([[1],[0],[1]])
item_id = np.array([[30],[20],[10]])
pic_vec = np.array([[0.1,0.5,0.4,0.3,0.2],[0.1,0.5,0.4,0.3,0.2],[0.1,0.5,0.4,0.3,0.2]])
label = np.array([1,0,1])

model_input = {'user_id':user_id,'item_id':item_id,'pic_vec':pic_vec}

model = DeepFM(feature_columns,feature_columns)
model.compile('adagrad','binary_crossentropy')
model.fit(model_input,label)
```

## 6. How to run the demo with GPU ?
just install deepctr with 
```bash
$ pip install deepctr[gpu]
```

## 7. How to run the demo with multiple GPUs
you can use multiple gpus with tensorflow version higher than ``1.4``,see [run_classification_criteo_multi_gpu.py](https://github.com/shenweichen/DeepCTR/blob/master/examples/run_classification_criteo_multi_gpu.py)
