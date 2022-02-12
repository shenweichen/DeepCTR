import pytest
import tensorflow as tf

from deepctr.models import EDCN
from deepctr.estimator import EDCNEstimator
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, \
    Estimator_TEST_TF1


@pytest.mark.parametrize(
    'bridge_type, tau, use_dense_features, cross_num, cross_parameterization, sparse_feature_num', 
    [
        ('pointwise_addition', 0.1, True, 2, 'vector', 4),
        ('hadamard_product', 0.1, True, 2, 'vector', 4),
        ('concatenation', 0.1, True, 2, 'vector', 4),
        ('attention_pooling', 0.1, True, 2, 'vector', 4),

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


@pytest.mark.parametrize(
    'bridge_type, tau, use_dense_features, cross_num, cross_parameterization, sparse_feature_num', 
    [
        ('pointwise_addition', 0.1, True, 2, 'vector', 4),
        ('hadamard_product', 0.1, True, 2, 'vector', 4),
        ('concatenation', 0.1, True, 2, 'vector', 4),
        ('attention_pooling', 0.1, True, 2, 'vector', 4),

        ('pointwise_addition', 1, True, 2, 'vector', 3),
        ('hadamard_product', 1, False, 2, 'vector', 4),
        ('concatenation', 1, True, 3, 'vector', 5),
        ('attention_pooling', 1, True, 2, 'matrix', 6),      
     ]
)

def test_EDCNEstimator(bridge_type, tau, use_dense_features, cross_num, cross_parameterization, sparse_feature_num):
    if not Estimator_TEST_TF1 and tf.__version__ < "2.2.0":
        return
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num)

    model = EDCNEstimator(linear_feature_columns, dnn_feature_columns, 
        bridge_type, tau, use_dense_features, cross_num, cross_parameterization)
    check_estimator(model, input_fn)

if __name__ == "__main__":
    pass
