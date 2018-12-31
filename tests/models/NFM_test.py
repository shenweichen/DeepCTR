import numpy as np
import pytest
from deepctr.models import NFM
from ..utils import check_model


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((8,), 1), ((8, 8,), 2)]
)
def test_NFM(hidden_size, sparse_feature_num):

    model_name = "NFM"

    sample_size = 64
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num)

    model = NFM(feature_dim_dict, embedding_size=8,
                hidden_size=[32, 32], keep_prob=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_NFM((8, 8), 1)
