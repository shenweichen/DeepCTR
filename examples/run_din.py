import numpy as np
from deepctr.models import DIN


def get_xy_fd():

    feature_dim_dict = {"sparse": {'user_age': 4, 'user_gender': 2,
                                   'item_id': 4, 'item_gender': 2}, "dense": []}  # raw feature:single value feature

    # history behavior feature:multi-value value feature
    behavior_feature_list = ["item_id", "item_gender"]
    # single value feature input
    user_age = np.array([1, 2, 3])
    user_gender = np.array([0, 1, 0])
    item_id = np.array([0, 1, 2])
    item_gender = np.array([0, 1, 0])

    # multi-value feature input
    hist_item_id = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 0]])
    hist_item_gender = np.array([[0, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 0]])
    # valid length of behavior sequence of every sample
    hist_length = np.array([4, 4, 3])

    feature_dict = {'user_age': user_age, 'user_gender': user_gender, 'item_id': item_id, 'item_gender': item_gender,
                    'hist_item_id': hist_item_id, 'hist_item_gender': hist_item_gender, }

    x = [feature_dict[feat] for feat in feature_dim_dict["sparse"]] + \
        [feature_dict['hist_'+feat]
            for feat in behavior_feature_list] + [hist_length]
    # Notice the concatenation order: single feature + multi-value feature + length
    # Since the length of the historical sequences of different features in DIN are the same(they are all extended from item_id),only one length vector is enough.
    y = [1, 0, 1]

    return x, y, feature_dim_dict, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd()
    model = DIN(feature_dim_dict, behavior_feature_list, hist_len_max=4,)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, validation_split=0.5)
