import pytest
from deepctr.models import DeepFM
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'use_fm,hidden_size,sparse_feature_num',
    [(True, (), 1), (False, (), 2), (True, (32,), 3), (False, (32,), 1)
     ]
)
def test_DeepFM(use_fm, hidden_size, sparse_feature_num):
    model_name = "DeepFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num)

    model = DeepFM(feature_dim_dict,  use_fm=use_fm,
                   hidden_size=hidden_size, keep_prob=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
