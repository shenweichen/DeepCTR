import numpy as np
import pytest
from deepctr.models import xDeepFM
from deepctr.utils import custom_objects
from tensorflow.python.keras.models import save_model, load_model


@pytest.mark.parametrize(
    'hidden_size,cin_layer_size,cin_split_half,cin_activation,sparse_feature_num,dense_feature_dim',
    [((), (), True, 'linear', 1, 2), ((16,), (), True, 'linear', 1, 1), ((), (16,), True, 'linear', 2, 2), ((16,), (16,), False, 'relu', 1, 0)
     ]
)
def test_xDeepFM(hidden_size, cin_layer_size, cin_split_half, cin_activation, sparse_feature_num, dense_feature_dim):
    model_name = "xDeepFM"

    sample_size = 64
    feature_dim_dict = {"sparse": {}, 'dense': []}
    for name, num in zip(["sparse", "dense"], [sparse_feature_num, dense_feature_dim]):
        if name == "sparse":
            for i in range(num):
                feature_dim_dict[name][name + '_' +
                                       str(i)] = np.random.randint(1, 10)
        else:
            for i in range(num):
                feature_dim_dict[name].append(name + '_' + str(i))
    sparse_input = [np.random.randint(0, dim, sample_size)
                    for dim in feature_dim_dict['sparse'].values()]
    dense_input = [np.random.random(sample_size)
                   for name in feature_dim_dict['dense']]

    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input

    model = xDeepFM(feature_dim_dict, hidden_size=hidden_size, cin_layer_size=cin_layer_size,
                    cin_split_half=cin_split_half, cin_activation=cin_activation, keep_prob=0.5, )
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'hidden_size,cin_layer_size,',
    [((8,), (3, 8)),
     ]
)
def test_xDeepFM_invalid(hidden_size, cin_layer_size):
    feature_dim_dict = {'sparse': {'sparse_1': 2, 'sparse_2': 5,
                                   'sparse_3': 10}, 'dense': ['dense_1', 'dense_2', 'dense_3']}
    with pytest.raises(ValueError):
        _ = xDeepFM(feature_dim_dict, hidden_size=hidden_size,
                    cin_layer_size=cin_layer_size,)


if __name__ == "__main__":
    test_xDeepFM((256), (128,), False, 'linear', 3, 1)
