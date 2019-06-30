import pytest
from deepctr.models import NFFM
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((8,), 2)]
)
def test_NFFM(hidden_size, sparse_feature_num):

    model_name = "ONN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num, sparse_feature_num, sequence_feature=(),hash_flag=True)

    model = NFFM(feature_columns, feature_columns, embedding_size=4,
                 dnn_hidden_units=[32, 32], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
