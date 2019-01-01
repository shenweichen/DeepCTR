import pytest
from deepctr.models import FNN

from ..utils import check_model, get_test_data


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(1, 1), (3, 3)
     ]
)
def test_FNN(sparse_feature_num, dense_feature_num):
    model_name = "FNN"

    sample_size = 64
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num)

    model = FNN(feature_dim_dict,  hidden_size=[32, 32], keep_prob=0.5, )
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(0, 1), (1, 0)
     ]
)
def test_FNN_without_seq(sparse_feature_num, dense_feature_num):
    model_name = "FNN"

    sample_size = 64
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num, sequence_feature=())

    model = FNN(feature_dim_dict,  hidden_size=[32, 32], keep_prob=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_FNN(2, 2)
