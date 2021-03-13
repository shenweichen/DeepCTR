from deepctr.models import BST
from ..utils import check_model
from .DIN_test import get_xy_fd


def test_BST():
    model_name = "BST"

    x, y, feature_columns, behavior_feature_list = get_xy_fd(hash_flag=True)

    model = BST(dnn_feature_columns=feature_columns,
                history_feature_list=behavior_feature_list,
                att_head_num=4)

    check_model(model, model_name, x, y,
                check_model_io=True)


if __name__ == "__main__":
    pass
