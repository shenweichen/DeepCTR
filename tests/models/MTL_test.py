import pytest
import tensorflow as tf

from deepctr.models.multitask import SharedBottom, ESMM, MMOE, PLE
from ..utils_mtl import get_mtl_test_data, check_mtl_model


def test_SharedBottom():
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "SharedBottom"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = SharedBottom(dnn_feature_columns, bottom_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,),
                         task_types=['binary', 'binary'], task_names=['label_income', 'label_marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_ESMM():
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "ESMM"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = ESMM(dnn_feature_columns, tower_dnn_hidden_units=(8,), task_types=['binary', 'binary'],
                 task_names=['label_marital', 'label_income'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_MMOE():
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "MMOE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = MMOE(dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(8,),
                 tower_dnn_hidden_units=(8,),
                 gate_dnn_hidden_units=(), task_types=['binary', 'binary'],
                 task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


@pytest.mark.parametrize(
    'num_levels,gate_dnn_hidden_units',
    [(2, ()),
     (1, (4,))]
)
def test_PLE(num_levels, gate_dnn_hidden_units):
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "PLE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = PLE(dnn_feature_columns, num_levels=num_levels, expert_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,),
                gate_dnn_hidden_units=gate_dnn_hidden_units,
                task_types=['binary', 'binary'], task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


if __name__ == "__main__":
    pass
