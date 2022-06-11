import pytest

from deepctr.models import FiBiNET
from ..utils import check_model, SAMPLE_SIZE, get_test_data, get_test_data_estimator, check_estimator, TEST_Estimator


@pytest.mark.parametrize(
    'bilinear_type',
    ["each",
     "all", "interaction"]
)
def test_FiBiNET(bilinear_type):
    model_name = "FiBiNET"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=2, dense_feature_num=2)

    model = FiBiNET(feature_columns, feature_columns, bilinear_type=bilinear_type, dnn_hidden_units=[4, ],
                    dnn_dropout=0.5, )
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'bilinear_type',
    ["interaction"]
)
def test_FiBiNETEstimator(bilinear_type):
    if not TEST_Estimator:
        return
    from deepctr.estimator import FiBiNETEstimator

    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size, sparse_feature_num=2,
                                                                                    dense_feature_num=2)

    model = FiBiNETEstimator(linear_feature_columns, dnn_feature_columns, bilinear_type=bilinear_type,
                             dnn_hidden_units=[4, ], dnn_dropout=0.5, )

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
