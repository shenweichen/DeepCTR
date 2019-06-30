import pytest
from deepctr.models import DeepFM
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'use_fm,hidden_size,sparse_feature_num',
    [(True, (), 1), (False, (), 2), (True, (32,), 3), (False, (32,), 1)
     ]
)
def  test_DeepFM(use_fm, hidden_size, sparse_feature_num):
    model_name = "DeepFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num, sparse_feature_num)

    model = DeepFM(feature_columns,feature_columns, use_fm=use_fm, dnn_hidden_units=hidden_size, dnn_dropout=0.5)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
