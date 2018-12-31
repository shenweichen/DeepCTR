import pytest
from deepctr.models import FNN

from ..utils import check_model, get_test_data


@pytest.mark.parametrize(
    'sparse_feature_num',
    [1, 3
     ]
)
def test_FNN(sparse_feature_num):
    model_name = "FNN"

    sample_size = 64
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num)

    model = FNN(feature_dim_dict,  hidden_size=[32, 32], keep_prob=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_FNN(2)
