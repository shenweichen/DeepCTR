import pytest
from deepctr.models import xDeepFM
from ..utils import check_model, get_test_data


@pytest.mark.parametrize(
    'hidden_size,cin_layer_size,cin_split_half,cin_activation,sparse_feature_num,dense_feature_dim',
    [((), (), True, 'linear', 1, 2), ((16,), (), True, 'linear', 1, 1), ((), (16,), True, 'linear', 2, 2), ((16,), (16,), False, 'relu', 1, 0)
     ]
)
def test_xDeepFM(hidden_size, cin_layer_size, cin_split_half, cin_activation, sparse_feature_num, dense_feature_dim):
    model_name = "xDeepFM"

    sample_size = 64
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num)

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
    test_xDeepFM((256,), (128,), False, 'linear', 3, 1)
