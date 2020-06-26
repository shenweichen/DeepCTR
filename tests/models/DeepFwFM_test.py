import pytest
import tensorflow as tf
from deepctr.models import DeepFwFM
from deepctr.estimator import DeepFwFMEstimator
from ..utils import check_model, get_test_data, SAMPLE_SIZE,get_test_data_estimator,check_estimator


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 1),
     ((), 1),
     ]
)
def test_DeepFwFM(hidden_size, sparse_feature_num):
    model_name = "DeepFwFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = DeepFwFM(feature_columns, feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_model(model, model_name, x, y)

@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 2),
     ]
)

def test_DeepFwFMEstimator(hidden_size, sparse_feature_num):
    if tf.__version__ < "2.2.0":
        return
    model_name = "DeepFwFM"
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn= get_test_data_estimator(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = DeepFwFMEstimator(linear_feature_columns,dnn_feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_estimator(model,input_fn)


if __name__ == "__main__":
    pass
