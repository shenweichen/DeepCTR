import pytest
from deepctr.models import NFFM
from ..utils import check_model, get_test_data


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((8,), 1), ((8, 8,), 2)]
)
def test_NFFM(hidden_size, sparse_feature_num):

    model_name = "NFFM"

    sample_size = 64
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num,sequence_feature=())

    model = NFFM(feature_dim_dict, embedding_size=8,
                hidden_size=[32, 32],)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_NFFM((8, 8), 1)
