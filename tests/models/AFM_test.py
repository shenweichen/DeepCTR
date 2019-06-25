import pytest
from deepctr.models import AFM
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'use_attention,sparse_feature_num,dense_feature_num',
    [(True, 1, 1), (False, 3, 3),
     ]
)
def test_AFM(use_attention, sparse_feature_num, dense_feature_num):
    model_name = "AFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num, dense_feature_num)

    print('xxxxxx',feature_columns)

    model = AFM(feature_columns, feature_columns, use_attention=use_attention, afm_dropout=0.5)

    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
