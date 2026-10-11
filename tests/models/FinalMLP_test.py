import pytest

from deepctr.models import FinalMLP
from ..utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'mlp_units,num_heads,sparse_feature_num',
    [((4,), 1, 1),
     ((4, 2), 2, 2)]
)
def test_FinalMLP(mlp_units, num_heads, sparse_feature_num):
    model_name = "FinalMLP"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = FinalMLP(feature_columns, feature_columns,
                     mlp1_hidden_units=mlp_units, mlp2_hidden_units=mlp_units,
                     fs1_gate_units=(4,), fs2_gate_units=(4,), num_heads=num_heads, dnn_dropout=0.5)

    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
