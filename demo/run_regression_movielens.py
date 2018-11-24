import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM

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
    sparse_feature_dim = {feat: data[feat].nunique()
                          for feat in sparse_features}
    # 3.generate input data for model
    model_input = [data[feat].values for feat in sparse_feature_dim]
    # 4.Define Model,compile and train
    model = DeepFM({"sparse": sparse_feature_dim, "dense": []},
                   final_activation='linear')

    model.compile("adam", "mse", metrics=['mse'],)
    history = model.fit(model_input, data[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2,)

    print("demo done")
