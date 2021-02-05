import numpy as np
import pytest
import tensorflow as tf
from packaging import version

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from deepctr.models import DMIN
from tests.utils import check_model


def get_xy_fd(use_neg=False, hash_flag=False):
    feature_columns = [SparseFeat('user', 3, embedding_dim=12, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 1)]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4,
                         length_name="seq_length")]

    behavior_feature_list = ["item_id", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cate_id = np.array([1, 2, 2])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])

    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                    'pay_score': score, "seq_length": behavior_length}

    if use_neg:
        feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_cate_id'] = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])
        feature_columns += [
            VarLenSparseFeat(
                SparseFeat('neg_hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                maxlen=4, length_name="seq_length"),
            VarLenSparseFeat(SparseFeat('neg_hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'),
                             maxlen=4, length_name="seq_length")]

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    x["position_hist"] = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    return x, y, feature_columns, behavior_feature_list


# @pytest.mark.xfail(reason="There is a bug when save model use Dice")
# @pytest.mark.skip(reason="misunderstood the API")

def test_DMIN():
    if version.parse(tf.__version__) >= version.parse('2.0.0'):
        tf.compat.v1.disable_eager_execution() 
    model_name = "DMIN"

    x, y, feature_columns, behavior_feature_list = get_xy_fd(hash_flag=False)

    model = DMIN(dnn_feature_columns=feature_columns,
                 history_feature_list=behavior_feature_list,
                 position_embedding_dim=2,
                 use_negsampling=False)

    check_model(model, model_name, x, y)


def test_DMIN_neg():
    model_name = "DMIN_neg"
    if version.parse(tf.__version__) >= version.parse("1.14.0"):
        tf.compat.v1.disable_eager_execution()

    x, y, feature_columns, behavior_feature_list = get_xy_fd(use_neg=True)

    model = DMIN(dnn_feature_columns=feature_columns,
                 history_feature_list=behavior_feature_list,
                 position_embedding_dim=2,
                 use_negsampling=True)
    check_model(model, model_name, x, y,check_model_io=True)


if __name__ == "__main__":
    test_DMIN()
    test_DMIN_neg()