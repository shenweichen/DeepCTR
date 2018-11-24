
from tensorflow.python.keras.layers import Input




def get_input(feature_dim_dict, bias_feature_dim_dict=None):
    sparse_input = [Input(shape=(1,), name='sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])]
    dense_input = [Input(shape=(1,), name='dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])]
    if bias_feature_dim_dict is None:
        return sparse_input, dense_input
    else:
        bias_sparse_input = [Input(shape=(1,), name='bias_sparse_' + str(i) + '-' + feat) for i, feat in
                             enumerate(bias_feature_dim_dict["sparse"])]
        bias_dense_input = [Input(shape=(1,), name='bias_dense_' + str(i) + '-' + feat) for i, feat in
                            enumerate(bias_feature_dim_dict["dense"])]
        return sparse_input, dense_input, bias_sparse_input, bias_dense_input

