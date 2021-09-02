import numpy as np
import pytest

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr.models.sequence.dsin import DSIN
from ..utils import check_model


def get_xy_fd(hash_flag=False):
    feature_columns = [SparseFeat('user', 3, use_hash=hash_flag),
                       SparseFeat('gender', 2, use_hash=hash_flag),
                       SparseFeat('item', 3 + 1, use_hash=hash_flag),
                       SparseFeat('item_gender', 2 + 1, use_hash=hash_flag),
                       DenseFeat('score', 1)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('sess_0_item', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item'),
                         maxlen=4), VarLenSparseFeat(
            SparseFeat('sess_0_item_gender', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item_gender'),
            maxlen=4)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('sess_1_item', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item'),
                         maxlen=4), VarLenSparseFeat(
            SparseFeat('sess_1_item_gender', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item_gender'),
            maxlen=4)]

    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    sess1_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [0, 0, 0, 0]])
    sess1_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [0, 0, 0, 0]])

    sess2_iid = np.array([[1, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    sess2_igender = np.array([[1, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    sess_number = np.array([2, 1, 0])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'sess_0_item': sess1_iid, 'sess_0_item_gender': sess1_igender, 'score': score,
                    'sess_1_item': sess2_iid, 'sess_1_item_gender': sess2_igender, }

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    x["sess_length"] = sess_number

    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list


@pytest.mark.parametrize(
    'bias_encoding',
    [True, False]
)
def test_DSIN(bias_encoding):
    model_name = "DSIN"

    x, y, feature_columns, behavior_feature_list = get_xy_fd(True)

    model = DSIN(feature_columns, behavior_feature_list, sess_max_count=2, bias_encoding=bias_encoding,
                 dnn_hidden_units=[4, 4], dnn_dropout=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
