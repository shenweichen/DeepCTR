import pytest

from deepctr.models import DCNMix
from ..utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'cross_num,hidden_size,sparse_feature_num',
    [(0, (8,), 2), (1, (), 1), (1, (8,), 3)
     ]
)
def test_DCNMix(cross_num, hidden_size, sparse_feature_num):
    model_name = "DCNMix"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = DCNMix(feature_columns, feature_columns, cross_num=cross_num, dnn_hidden_units=hidden_size, dnn_dropout=0.5)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
