import pytest
from deepctr.models import DeepFwFM
from ..utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 0),
     ((), 1),
     ]
)
def test_DeepFwFM(hidden_size, sparse_feature_num):
    model_name = "DeepFwFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = DeepFwFM(feature_columns, feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
