import pytest
from deepctr.models import FNN
import tensorflow as tf

from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(1, 1), (3, 3)
     ]
)
def test_FNN(sparse_feature_num, dense_feature_num):
    if tf.__version__ >= "2.0.0":
        return
    model_name = "FNN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=dense_feature_num)

    model = FNN(feature_columns,feature_columns, dnn_hidden_units=[32, 32], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


# @pytest.mark.parametrize(
#     'sparse_feature_num,dense_feature_num',
#     [(0, 1), (1, 0)
#      ]
# )
# def test_FNN_without_seq(sparse_feature_num, dense_feature_num):
#     model_name = "FNN"
#
#     sample_size = SAMPLE_SIZE
#     x, y, feature_columns = get_test_data(sample_size, sparse_feature_num, dense_feature_num, sequence_feature=())
#
#     model = FNN(feature_columns,feature_columns, dnn_hidden_units=[32, 32], dnn_dropout=0.5)
#     check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
