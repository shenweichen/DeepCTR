#test multi task learning models

from ..utils_mtl import get_mtl_test_data, check_mtl_model
from deepctr.models.mtl.shared_bottom import Shared_Bottom
from deepctr.models.mtl.essm import ESSM
from deepctr.models.mtl.mmoe import MMOE
from deepctr.models.mtl.cgc import CGC
from deepctr.models.mtl.ple import PLE
import tensorflow as tf

def test_Shared_Bottom():
    model_name = "Shared_Bottom"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = Shared_Bottom(dnn_feature_columns, num_tasks=2, task_types= ['binary', 'binary'],
                          task_names=['label_income','label_marital'], bottom_dnn_units=[16],
                          tower_dnn_units_lists=[[8],[8]])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])

def test_ESSM():
    model_name = "ESSM"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = ESSM(dnn_feature_columns, task_type='binary', task_names=['label_marital', 'label_income'], tower_dnn_units_lists=[[8],[8]])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary','binary'])

def test_MMOE():
    if tf.__version__ == "1.15.0" or tf.__version__ =="1.4.0":
        return
    model_name = "MMOE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()

    model = MMOE(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], task_names=['income','marital'],
                num_experts=8, expert_dnn_units=[16], gate_dnn_units=None, tower_dnn_units_lists=[[8],[8]])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])

def test_CGC():
    if tf.__version__ == "1.15.0" or tf.__version__ =="1.4.0":
        return
    model_name = "CGC"
    x, y_list, dnn_feature_columns = get_mtl_test_data()
    
    model = CGC(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], task_names=['income','marital'],
                    num_experts_specific=4, num_experts_shared=4, expert_dnn_units=[16], gate_dnn_units=None, tower_dnn_units_lists=[[8],[8]])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])

def test_PLE():
    if tf.__version__ == "1.15.0" or tf.__version__ =="1.4.0":
        return
    model_name = "PLE"
    x, y_list, dnn_feature_columns = get_mtl_test_data()
    
    model = PLE(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], task_names=['income','marital'],
                num_levels=2, num_experts_specific=4, num_experts_shared=4, expert_dnn_units=[16],
                gate_dnn_units=None,tower_dnn_units_lists=[[8],[8]])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])

if __name__ == "__main__":
    pass
    # test_Shared_Bottom()
    # test_ESSM()
    # test_MMOE()
    # test_CGC()
    # test_PLE()