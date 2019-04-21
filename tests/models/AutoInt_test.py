import pytest
from deepctr.models import AutoInt
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'att_layer_num,hidden_size,sparse_feature_num',
    [(0, (4,), 2), (1, (), 1), (1, (4,), 1), (2, (4, 4,), 2)]
)
def test_AutoInt(att_layer_num, hidden_size, sparse_feature_num):
    model_name = "AutoInt"
    sample_size = SAMPLE_SIZE
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num)

    model = AutoInt(feature_dim_dict,  att_layer_num=att_layer_num,
                    hidden_size=hidden_size, keep_prob=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_AutoInt(2, (32, 32), 2)
