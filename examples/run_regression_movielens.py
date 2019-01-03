import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[feat].values for feat in sparse_feature_dim]
    test_model_input = [test[feat].values for feat in sparse_feature_dim]
    # 4.Define Model,train,predict and evaluate
    model = DeepFM({"sparse": sparse_feature_dim, "dense": []},
                   final_activation='linear')
    model.compile("adam", "mse", metrics=['mse'],)

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=1, verbose=2, validation_split=0.2,)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(mean_squared_error(
        test[target].values, pred_ans), 4))
