import numpy as np

from deepctr.models import DSIN
from deepctr.utils import SingleFeat


def get_xy_fd(hash_flag=False):
    feature_dim_dict = {"sparse": [SingleFeat('user', 3, hash_flag), SingleFeat(
        'gender', 2, hash_flag), SingleFeat('item', 3 + 1, hash_flag), SingleFeat('item_gender', 2 + 1, hash_flag)],
                        "dense": [SingleFeat('score', 0)]}
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
                    'sess1_item': sess1_iid, 'sess1_item_gender': sess1_igender, 'score': score,
                    'sess2_item': sess2_iid, 'sess2_item_gender': sess2_igender, }

    x = [feature_dict[feat.name] for feat in feature_dim_dict["sparse"]] + [feature_dict[feat.name] for feat in
                                                                            feature_dim_dict["dense"]] + [
            feature_dict['sess1_' + feat] for feat in behavior_feature_list] + [
            feature_dict['sess2_' + feat] for feat in behavior_feature_list]

    x += [sess_number]

    y = [1, 0, 1]
    return x, y, feature_dim_dict, behavior_feature_list

if __name__ == "__main__":
    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd(True)

    model = DSIN(feature_dim_dict, behavior_feature_list, sess_max_count=2, sess_len_max=4, embedding_size=4,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.5, )

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)

