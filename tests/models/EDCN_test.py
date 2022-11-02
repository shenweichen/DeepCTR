import pytest
import tensorflow as tf

from deepctr.models import EDCN
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, \
    TEST_Estimator


@pytest.mark.parametrize(
    'bridge_type, tau, use_dense_features, cross_num, cross_parameterization, sparse_feature_num',
    [
        ('pointwise_addition', 1, True, 2, 'vector', 3),
        ('hadamard_product', 1, False, 2, 'vector', 4),
        ('concatenation', 1, True, 3, 'vector', 5),
        ('attention_pooling', 1, True, 2, 'matrix', 6),
    ]
)
def test_EDCN(bridge_type, tau, use_dense_features, cross_num, cross_parameterization, sparse_feature_num):
    model_name = "EDCN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = EDCN(feature_columns, feature_columns,
                 bridge_type, tau, use_dense_features, cross_num, cross_parameterization)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
