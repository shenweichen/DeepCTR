import pytest
from deepctr.models import AutoInt
from ..utils import check_model, get_test_data,SAMPLE_SIZE
import tensorflow as tf


@pytest.mark.parametrize(
    'att_layer_num,dnn_hidden_units,sparse_feature_num',
    [(1, (), 1), (1, (4,), 1)]#(0, (4,), 2), (2, (4, 4,), 2)
)
def test_AutoInt(att_layer_num, dnn_hidden_units, sparse_feature_num):
    if tf.__version__ >= "2.0.0" and len(dnn_hidden_units)==0:#todo check version
        return
    model_name = "AutoInt"
    sample_size = SAMPLE_SIZE
    x, y, feature_dim_dict = get_test_data(sample_size, sparse_feature_num, sparse_feature_num)

    model = AutoInt(feature_dim_dict, att_layer_num=att_layer_num,
                    dnn_hidden_units=dnn_hidden_units, dnn_dropout=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
