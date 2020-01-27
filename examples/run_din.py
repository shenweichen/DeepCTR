import numpy as np

from deepctr.models import DIN
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_feature_names


def get_xy_fd():

    feature_columns = [SparseFeat('user',3,embedding_dim=10),SparseFeat(
        'gender', 2,embedding_dim=4), SparseFeat('item', 3 + 1,embedding_dim=8), SparseFeat('item_gender', 2 + 1,embedding_dim=4),DenseFeat('score', 1)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', vocabulary_size=3 + 1,embedding_dim=8,embedding_name='item'), maxlen=4),
                        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1,embedding_dim=4, embedding_name='item_gender'), maxlen=4)]

    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score}
    x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
    y = [1, 0, 1]
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DIN(feature_columns, behavior_feature_list)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
