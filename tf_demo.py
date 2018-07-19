import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tf_model.deepfm import DeepFM
from tf_model.deep_cross_network import  DeepCrossNetwork
if __name__ == "__main__":

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
    model_input = np.array([data[feat].values for feat in sparse_feature_dim]).T
    model_label = data[target].values.reshape(-1)
    #4.Define Model,compile and train
    model_lists = [DeepCrossNetwork({"sparse":sparse_feature_dim,"dense":[]},),
                   DeepFM({"sparse":sparse_feature_dim,"dense":[]}),
                  ]
    for model in model_lists:
        model.compile("adam","mse",)
        history = model.fit(model_input,model_label,
                  batch_size=256,epochs=1,verbose=2,validation_split=0.2,)

