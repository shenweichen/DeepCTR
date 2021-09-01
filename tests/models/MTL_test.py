import pytest

from deepctr.models.mtl.cgc import CGC
from deepctr.models.mtl.esmm import ESMM
from deepctr.models.mtl.mmoe import MMOE
from deepctr.models.mtl.ple import PLE
from deepctr.models.mtl.sharedbottom import SharedBottom
from ..utils_mtl import get_mtl_test_data, check_mtl_model


def test_Shared_Bottom():
    model_name = "Shared_Bottom"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = SharedBottom(dnn_feature_columns, bottom_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,),
                         task_types=['binary', 'binary'], task_names=['label_income', 'label_marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_ESMM():
    model_name = "ESMM"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = ESMM(dnn_feature_columns, tower_dnn_hidden_units=(8,), task_type='binary',
                 task_names=['label_marital', 'label_income'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


@pytest.mark.parametrize(
    'gate_dnn_hidden_units',
    [None,
     (4,)]
)
def test_MMOE(gate_dnn_hidden_units):
    model_name = "MMOE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = MMOE(dnn_feature_columns, num_experts=8, expert_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,),
                 gate_dnn_hidden_units=gate_dnn_hidden_units, task_types=['binary', 'binary'],
                 task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


@pytest.mark.parametrize(
    'gate_dnn_hidden_units',
    [None,
     (4,)]
)
def test_CGC(gate_dnn_hidden_units):
    model_name = "CGC"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = CGC(dnn_feature_columns,
                specific_expert_num=1, shared_expert_num=1, expert_dnn_hidden_units=(8,),
                gate_dnn_hidden_units=gate_dnn_hidden_units,
                tower_dnn_hidden_units=(8,), task_types=['binary', 'binary'], task_names=['income', 'marital'], )
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


@pytest.mark.parametrize(
    'gate_dnn_hidden_units',
    [None,
     (4,)]
)
def test_PLE(gate_dnn_hidden_units):
    model_name = "PLE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = PLE(dnn_feature_columns, shared_expert_num=1, specific_expert_num=1, num_levels=2,
                expert_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,), gate_dnn_hidden_units=gate_dnn_hidden_units,
                task_types=['binary', 'binary'], task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


if __name__ == "__main__":
    pass
