import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras_model import AFM,DCN,DeepFM,MLR,NFM

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
    model_input = [data[feat].values for feat in sparse_feature_dim] #+ [data[target[0]].values]
    #4.Define Model,compile and train
    model_lists = [AFM({"sparse":sparse_feature_dim,"dense":[]},final_activation='linear').model,
                   DCN({"sparse":sparse_feature_dim,"dense":[]},final_activation='linear').model,
                   DeepFM({"sparse":sparse_feature_dim,"dense":[]},final_activation='linear').model,
                   MLR({"sparse":sparse_feature_dim,"dense":[]},activation='linear').model,
                   NFM({"sparse":sparse_feature_dim,"dense":[]},final_activation='linear').model]
    for model in model_lists:
        #model = DeepFM({"sparse":sparse_feature_dim,"dense":[]},final_activation='linear').model
        model.compile("adam","mse",metrics=['mse'],)
        history = model.fit(model_input,data[target],
                  batch_size=256,epochs=1,verbose=2,validation_split=0.2,)

