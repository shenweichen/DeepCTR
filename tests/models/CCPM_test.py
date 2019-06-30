import pytest

from deepctr.models import CCPM
from tests.utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(1, 1), (3, 3)
     ]
)
def test_CCPM(sparse_feature_num, dense_feature_num):
    model_name = "CCPM"

    sample_size = 32
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num)

    model = CCPM(feature_columns,feature_columns, conv_kernel_width=(3, 2), conv_filters=(
        2, 1), dnn_hidden_units=[32, ], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(2, 0),
     ]
)
def test_CCPM_without_seq(sparse_feature_num, dense_feature_num):
    model_name = "CCPM"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num, sequence_feature=())

    model = CCPM(feature_columns, feature_columns,conv_kernel_width=(3, 2), conv_filters=(
        2, 1), dnn_hidden_units=[32, ], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
