import pytest
import tensorflow as tf
from packaging import version

from deepctr.models import ONN
from ..utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'sparse_feature_num',
    [2]
)
def test_ONN(sparse_feature_num):
    if version.parse(tf.__version__) >= version.parse('1.15.0'):
        return
    model_name = "ONN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num,
                                          sequence_feature=('sum', 'mean', 'max',), hash_flag=True)

    model = ONN(feature_columns, feature_columns,
                dnn_hidden_units=[4, 4], dnn_dropout=0.5)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
