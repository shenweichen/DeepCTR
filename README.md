# DeepCTR
This project  implements serval models of the papers on CTR prediction with easy-to-use call interfaces.  

The goal is to make it possible for everyone to use complex models with `model.fit()`and`model.predict()`.  

 Most of the models have been finished in keras. The tensorflow version will be added soon~
 Please feel free to contact me if you have any questions!!
## Support Model List

  |Model|Paper|Available Framework|
  |:--:|--|--|
  |AFM|[IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435)|`keras`|
  |DCN|[ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/citation.cfm?id=3124754)|`keras`,`tensorflow`|
  |DeepFM|[IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)|`keras`,`tensorflow`|
  |MLR|[arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)|`keras`,|
  |NFM|[SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777)|`keras`|

## Keras Model
  ### Requirements
  - python3
  - tensorflow==1.4.0
  - keras==2.1.2
  ### Quick Start
  Source code [`keras_demo.py`](./keras_demo.py)  
  
  ![](docs/data_view.png)
  ```Python
data = pd.read_pickle("./demo/small_data.pkl")
sparse_features = [ "movie_id","user_id","gender","age","occupation","zip"]
target = ['rating']
#1.Label Encoding for sparse features,and Normalization for dense fetures
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
#2.count #unique features for each sparse field
sparse_feature_dim = {feat:data[feat].nunique() for feat in sparse_features}
#3.generate input data for model
model_input = [data[feat].values for feat in sparse_feature_dim] 
#4.Define Model,compile and train
model = NFM({"sparse":sparse_feature_dim,"dense":[]},final_activation='linear').model
  
model.compile("adam","mse",metrics=['mse'],)
history = model.fit(model_input,data[target],
          batch_size=256,epochs=5,verbose=2,validation_split=0.2,)
  ```
## TensorFlow Model
  ### Requirements
  - python3
  - tensorflow==1.4.0
  - numpy==1.13.3
  - scikit-learn==0.19.1
  ### Design Notes
  The `base` base class mimics the `keras` model to implement the following public methods, including:
  - compile  
  - save_model 
  - load_mdel 
  - train_on_batch 
  - fit 
  - test_on_batch 
  - evaluate 
  - predict_on_batch 
  - predict   

private methods:
  - _create_optimizer
  - _create_metrics  
  - _compute_sample_weight

At the same time, several abstract methods are designed:
  - _get_input_data
  - _get_input_target
  - _get_output_target
  - _get_optimizer_loss
  - _build_graph 

The subclass is required to call `self._build_graph()` at the end of the `__init__` method to build the calculation graph. 


  ### TODO
  - Add `tf.summary.FileWriter`
  - Add  custom metric function
  - Add weighted loss function
  - Encapsulate  models with `tf.estimator`
 ### Quick Start
  Source code [`tf_demo.py`](./tf_demo.py)  
## Experiment Result
see [docs/README.md](docs/README.md)