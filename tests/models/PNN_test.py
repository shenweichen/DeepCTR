import pytest

from deepctr.models import PNN
from ..utils import check_model, get_test_data


@pytest.mark.parametrize(
    'use_inner, use_outter,sparse_feature_num',
    [(True, True, 1), (True, False, 2), (False, True, 3), (False, False, 1)
     ]
)
def test_PNN(use_inner, use_outter, sparse_feature_num):
    model_name = "PNN"
    sample_size = 64
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num)
    model = PNN(feature_dim_dict, embedding_size=8,
                hidden_size=[32, 32], keep_prob=0.5, use_inner=use_inner, use_outter=use_outter)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_PNN(use_inner=True, use_outter=False, sparse_feature_num=1)
