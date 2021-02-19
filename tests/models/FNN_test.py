import pytest
import tensorflow as tf

from deepctr.estimator import FNNEstimator
from deepctr.models import FNN
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, \
    Estimator_TEST_TF1


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

    model = FNN(feature_columns, feature_columns, dnn_hidden_units=[8, 8], dnn_dropout=0.5)
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

@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(2, 2),
     ]
)
def test_FNNEstimator(sparse_feature_num, dense_feature_num):
    if not Estimator_TEST_TF1 and tf.__version__ < "2.2.0":
        return
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=dense_feature_num)

    model = FNNEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[8, 8], dnn_dropout=0.5)

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
