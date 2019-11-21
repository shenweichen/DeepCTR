import pytest

from deepctr.models import PNN
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'use_inner, use_outter,sparse_feature_num',
    [(True, True, 1), (True, False, 2), (False, True, 3), (False, False, 1)
     ]
)
def test_PNN(use_inner, use_outter, sparse_feature_num):
    model_name = "PNN"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = PNN(feature_columns, embedding_size=4, dnn_hidden_units=[4, 4], dnn_dropout=0.5, use_inner=use_inner,
                use_outter=use_outter)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
