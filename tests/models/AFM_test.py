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
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num)
    from tensorflow.python.keras import backend as K
    #K.set_learning_phase(True)
    model = AFM(feature_dim_dict, use_attention=use_attention, att_dropout=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
