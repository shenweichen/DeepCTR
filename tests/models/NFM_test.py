import pytest
import tensorflow as tf

from deepctr.estimator import NFMEstimator
from deepctr.models import NFM
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, \
    Estimator_TEST_TF1


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((8,), 1), ((8, 8,), 2)]
)
def test_NFM(hidden_size, sparse_feature_num):
    model_name = "NFM"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = NFM(feature_columns, feature_columns, dnn_hidden_units=[8, 8], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((8,), 1), ((8, 8,), 2)]
)
def test_FNNEstimator(hidden_size, sparse_feature_num):
    if not Estimator_TEST_TF1 and tf.__version__ < "2.2.0":
        return
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num)

    model = NFMEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[8, 8], dnn_dropout=0.5)

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
