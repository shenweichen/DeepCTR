import numpy as np
import tensorflow as tf

from deepctr.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DIEN


def get_xy_fd(use_neg=False, hash_flag=False):
    feature_columns = [SparseFeat('user', 3, embedding_dim=10, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item', 3 + 1, embedding_dim=8, use_hash=hash_flag),
                       SparseFeat('item_gender', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('score', 1)]

    feature_columns += [
        VarLenSparseFeat(SparseFeat('hist_item', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item'),
                         maxlen=4, length_name="seq_length"),
        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=4, embedding_name='item_gender'), maxlen=4,
                         length_name="seq_length")]

    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])

    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender,
                    'score': score, "seq_length": behavior_length}

    if use_neg:
        feature_dict['neg_hist_item'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_item_gender'] = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
        feature_columns += [
            VarLenSparseFeat(SparseFeat('neg_hist_item', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item'),
                             maxlen=4, length_name="seq_length"),
            VarLenSparseFeat(SparseFeat('neg_hist_item_gender', 2 + 1, embedding_dim=4, embedding_name='item_gender'),
                             maxlen=4, length_name="seq_length")]

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = [1, 0, 1]
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    x, y, feature_columns, behavior_feature_list = get_xy_fd(use_neg=True)
    model = DIEN(feature_columns, behavior_feature_list,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.6, gru_type="AUGRU", use_negsampling=True)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
