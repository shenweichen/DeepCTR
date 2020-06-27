import pytest
import tensorflow as tf
from deepctr.models import FwFM
from deepctr.estimator import FwFMEstimator
from ..utils import check_model, get_test_data, SAMPLE_SIZE,get_test_data_estimator,check_estimator,Estimator_TEST_TF1


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 1),
     ((), 1),
     ]
)
def test_FwFM(hidden_size, sparse_feature_num):
    model_name = "FwFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = FwFM(feature_columns, feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_model(model, model_name, x, y)

@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 2),
     ]
)

def test_FwFMEstimator(hidden_size, sparse_feature_num):
    if not Estimator_TEST_TF1 and tf.__version__ < "2.2.0":
        return
    model_name = "FwFM"
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn= get_test_data_estimator(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = FwFMEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_estimator(model,input_fn)


if __name__ == "__main__":
    pass