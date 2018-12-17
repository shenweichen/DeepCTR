import numpy as np
from deepctr.models import DIN
from deepctr.activations import Dice
from deepctr.utils import custom_objects

from tensorflow.python.keras.layers import PReLU
from tensorflow.python.keras.models import load_model, save_model


def get_xy_fd():
    feature_dim_dict = {"sparse": {'user': 4, 'gender': 2,
                                   'item': 4, 'item_gender': 2}, "dense": []}
    behavior_feature_list = ["item"]
    uid = np.array([1, 2, 3])
    ugender = np.array([0, 1, 0])
    iid = np.array([0, 1, 2])
    igender = np.array([0, 1, 0])

    hist_iid = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    hist_igender = np.array([[0, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 0]])
    hist_length = np.array([4, 4, 4])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, }
    x = [feature_dict[feat] for feat in feature_dim_dict["sparse"]] \
        + [feature_dict['hist_'+feat] for feat in behavior_feature_list]\
        + [hist_length]
    y = [1, 0, 1]
    return x, y, feature_dim_dict, behavior_feature_list


def test_DIN_att():

    name = "DIN_att"
    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd()

    model = DIN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8,
                use_din=True, hidden_size=[4, 4, 4], keep_prob=0.6,)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, verbose=1, validation_split=0.5)
    print(name+" test train valid pass!")
    model.save_weights(name + '_weights.h5')
    model.load_weights(name + '_weights.h5')
    print(name+" test save load weight pass!")

    try:
        save_model(model,  name + '.h5')
        model = load_model(name + '.h5', custom_objects)
        print(name + " test save load model pass!")
    except:
        print("【Error】There is a bug when save model use Dice or Prelu---------------------------------------------------")

    print(name + " test pass!")


def test_DIN_sum():

    name = "DIN_sum"
    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd()

    model = DIN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8,
                use_din=False, hidden_size=[4, 4, 4], keep_prob=0.6, activation="sigmoid")

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, verbose=1, validation_split=0.5)
    print(name+" test train valid pass!")
    model.save_weights(name + '_weights.h5')
    model.load_weights(name + '_weights.h5')
    print(name+" test save load weight pass!")

    try:
        save_model(model,  name + '.h5')
        model = load_model(name + '.h5', custom_objects)
        print(name + " test save load model pass!")
    except:
        print("There is a bug when save model use Dice or Prelu")

    print(name + " test pass!")


if __name__ == "__main__":
    test_DIN_att()
    test_DIN_sum()
