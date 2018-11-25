import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import DeepFM

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    sparse_feature_dict = {feat: data[feat].nunique()
                           for feat in sparse_features}
    dense_feature_list = dense_features

    # 3.generate input data for model

    model_input = [data[feat].values for feat in sparse_feature_dict] + \
        [data[feat].values for feat in dense_feature_list]  # + [data[target[0]].values]

    # 4.Define Model,compile and train
    model = DeepFM({"sparse": sparse_feature_dict,
                    "dense": dense_feature_list}, final_activation='sigmoid')

    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(model_input, data[target].values,

                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    print("demo done")
