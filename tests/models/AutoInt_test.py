import pytest
import tensorflow as tf
from packaging import version

from deepctr.models import AutoInt
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, \
    TEST_Estimator


@pytest.mark.parametrize(
    'att_layer_num,dnn_hidden_units,sparse_feature_num',
    [(1, (), 1), (1, (4,), 1)]  # (0, (4,), 2), (2, (4, 4,), 2)
)
def test_AutoInt(att_layer_num, dnn_hidden_units, sparse_feature_num):
    if version.parse(tf.__version__) >= version.parse("1.14.0") and len(dnn_hidden_units) == 0:  # todo check version
        return
    model_name = "AutoInt"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = AutoInt(feature_columns, feature_columns, att_layer_num=att_layer_num,
                    dnn_hidden_units=dnn_hidden_units, dnn_dropout=0.5, )
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'att_layer_num,dnn_hidden_units,sparse_feature_num',
    [(1, (4,), 1)]  # (0, (4,), 2), (2, (4, 4,), 2)
)
def test_AutoIntEstimator(att_layer_num, dnn_hidden_units, sparse_feature_num):
    if not TEST_Estimator:
        return
    from deepctr.estimator import AutoIntEstimator
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num)

    model = AutoIntEstimator(linear_feature_columns, dnn_feature_columns, att_layer_num=att_layer_num,
                             dnn_hidden_units=dnn_hidden_units, dnn_dropout=0.5, )
    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
