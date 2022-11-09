import pytest

from deepctr.models import EDCN
from ..utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'bridge_type, cross_num, cross_parameterization, sparse_feature_num',
    [
        ('pointwise_addition', 2, 'vector', 3),
        ('hadamard_product', 2, 'vector', 4),
        ('concatenation', 1, 'vector', 5),
        ('attention_pooling', 2, 'matrix', 6),
    ]
)
def test_EDCN(bridge_type, cross_num, cross_parameterization, sparse_feature_num):
    model_name = "EDCN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=0)

    model = EDCN(feature_columns, feature_columns, cross_num, cross_parameterization, bridge_type)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
