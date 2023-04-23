import pytest

from deepctr.models import AFM
from ..utils import check_model, check_estimator, get_test_data, get_test_data_estimator, SAMPLE_SIZE, \
    TEST_Estimator

@pytest.mark.parametrize(
    'afn_dnn_hidden_units, sparse_feature_num, dense_feature_num',
    [((32, 16), 3, 0),
     ((32, 16), 3, 3),
     ((32, 16), 0, 3)]
)
def test_AFN(afn_dnn_hidden_units, sparse_feature_num, dense_feature_num):
    model_name = 'AFN'
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = AFN(feature_columns, feature_columns, afn_dnn_hidden_units=afn_dnn_hidden_units, device=get_device())

    check_model(model, model_name, x, y)


if __name__ == '__main__':
    pass