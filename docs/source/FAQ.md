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

from deepctr.utils import custom_objects
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

model = deepctr.models.DeepFM({"sparse": sparse_feature_dict, "dense": dense_feature_list})
model.compile(Adagrad('0.0808'),'binary_crossentropy',metrics=['binary_crossentropy'])

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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Lambda

feature_dim_dict = {"sparse": sparse_feature_dict, "dense": dense_feature_list}
model = deepctr.models.AFM(feature_dim_dict)
model.fit(model_input,target)

afmlayer = model.layers[-3]
afm_weight_model = Model(model.input,outputs=Lambda(lambda x:afmlayer.normalized_att_score)(model.input))
attentional_weights = afm_weight_model.predict(model_input,batch_size=4096)
feature_interactions = list(itertools.combinations(list(feature_dim_dict['sparse'].keys()) + feature_dim_dict['dense'] ,2))
```

## 4. Does the models support multi-value input?
---------------------------------------------------
Now multi-value input is avaliable for `AFM,AutoInt,DCN,DeepFM,FNN,NFM,PNN,xDeepFM`,you can read the example [here](./Examples.html#multi-value-input-movielens).

For `DIN` please read the code example in [run_din.py](https://github.com/shenweichen/DeepCTR/blob/master/examples/run_din.py
).

You can also use layers in [sequence](./deepctr.layers.sequence.html)to build your own models !
