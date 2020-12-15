import numpy as np
from deepctr.models import DMR
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat,get_feature_names

def get_xy_fd():

    feature_columns = [SparseFeat('user',3,embedding_dim=10),SparseFeat(
        'gender', 2,embedding_dim=4), SparseFeat('item_id', 3 + 1,embedding_dim=8), SparseFeat('cate_id', 2 + 1,embedding_dim=4),DenseFeat('pay_score', 1)]

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
    pay_score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [1, 2, 0, 0]])
    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                    'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id, 'pay_score': pay_score, "seq_length": behavior_length}
    x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list

if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    deep_match_id = "item_id"
    model = DMR(feature_columns, behavior_feature_list, deep_match_id)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
