import pytest

from deepctr.models import CCPM
import tensorflow as tf
from tests.utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [ (3, 0)
     ]
)
def test_CCPM(sparse_feature_num, dense_feature_num):
    if tf.__version__ >= "2.0.0":#todo
        return
    model_name = "CCPM"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=dense_feature_num)

    model = CCPM(feature_columns,feature_columns, conv_kernel_width=(3, 2), conv_filters=(
        2, 1), dnn_hidden_units=[32, ], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(2, 0),
     ]
)
def test_CCPM_without_seq(sparse_feature_num, dense_feature_num):
    if tf.__version__ >= "2.0.0":
        return
    model_name = "CCPM"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=dense_feature_num, sequence_feature=())

    model = CCPM(feature_columns, feature_columns,conv_kernel_width=(3, 2), conv_filters=(
        2, 1), dnn_hidden_units=[32, ], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
