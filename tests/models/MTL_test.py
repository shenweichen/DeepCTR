# test multi task learning models

from deepctr.models.mtl.cgc import CGC
from deepctr.models.mtl.esmm import ESMM
from deepctr.models.mtl.mmoe import MMOE
from deepctr.models.mtl.ple import PLE
from deepctr.models.mtl.sharedbottom import SharedBottom
from ..utils_mtl import get_mtl_test_data, check_mtl_model


def test_Shared_Bottom():
    model_name = "Shared_Bottom"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = SharedBottom(dnn_feature_columns, bottom_dnn_units=(8,), tower_dnn_units_lists=(8,),
                         task_types=['binary', 'binary'], task_names=['label_income', 'label_marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_ESSM():
    model_name = "ESSM"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = ESMM(dnn_feature_columns, tower_dnn_hidden_units=(8,), task_type='binary',
                 task_names=['label_marital', 'label_income'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_MMOE():
    model_name = "MMOE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = MMOE(dnn_feature_columns, num_experts=8, expert_dnn_hidden_units=[16], tower_dnn_hidden_units=[[8], [8]],
                 gate_dnn_units=None, task_types=['binary', 'binary'], task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_CGC():
    model_name = "CGC"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = CGC(dnn_feature_columns,
                specific_expert_num=2, shared_expert_num=1, expert_dnn_hidden_units=(16,), gate_dnn_units=None,
                tower_dnn_hidden_units=(8,), task_types=['binary', 'binary'], task_names=['income', 'marital'], )
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


def test_PLE():
    model_name = "PLE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = PLE(dnn_feature_columns, shared_expert_num=1, specific_expert_num=1, num_levels=2,
                expert_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,), gate_dnn_units=None,
                task_types=['binary', 'binary'], task_names=['income', 'marital'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])


if __name__ == "__main__":
    pass
