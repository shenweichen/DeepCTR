import numpy as np
import pytest
from deepctr.models.dien import DIEN
from deepctr.layers.activation import Dice
from deepctr.utils import SingleFeat
from deepctr.layers import custom_objects
from tensorflow.python.keras.models import load_model, save_model
import tensorflow as tf


def get_xy_fd(use_neg=False):
    feature_dim_dict = {"sparse": [SingleFeat('user', 3), SingleFeat(
        'gender', 2), SingleFeat('item', 3+1), SingleFeat('item_gender', 2+1)], "dense": [SingleFeat('score', 0)]}
    behavior_feature_list = ["item","item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])#0 is mask value
    igender = np.array([1, 2, 1])# 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[ 1, 2, 3,0], [ 1, 2, 3,0], [ 1, 2, 0,0]])
    hist_igender = np.array([[1, 1, 2,0 ], [2, 1, 1, 0], [2, 1, 0, 0]])

    behavior_length = np.array([3,3,2])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender,
                    'score': score}

    x = [feature_dict[feat.name] for feat in feature_dim_dict["sparse"]] + [feature_dict[feat.name] for feat in
                                                                            feature_dim_dict["dense"]] + [
            feature_dict['hist_' + feat] for feat in behavior_feature_list]
    if use_neg:
        feature_dict['neg_hist_item'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_item_gender'] = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
        x += [feature_dict['neg_hist_'+feat] for feat in behavior_feature_list]

    x += [behavior_length]
    y = [1, 0, 1]
    return x, y, feature_dim_dict, behavior_feature_list


@pytest.mark.xfail(reason="There is a bug when save model use Dice")
# @pytest.mark.skip(reason="misunderstood the API")
def test_DIEN_model_io():

    model_name = "DIEN"
    _, _, feature_dim_dict, behavior_feature_list = get_xy_fd()

    model = DIEN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8, att_activation=Dice,

                hidden_size=[4, 4, 4], keep_prob=0.6,use_negsampling=False)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
   #model.fit(x, y, verbose=1, validation_split=0.5)
    save_model(model,  model_name + '.h5')
    model = load_model(model_name + '.h5', custom_objects)
    print(model_name + " test save load model pass!")

@pytest.mark.parametrize(
    'gru_type',
    ['GRU','AIGRU','AGRU','AUGRU',
     ]
)
def test_DIEN(gru_type):
    model_name = "DIEN"

    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd()

    model = DIEN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8,
                hidden_size=[4, 4, 4], keep_prob=0.6,gru_type=gru_type)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])

    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    model.fit(x, y, verbose=1, validation_split=0.5)

    print(model_name+" test train valid pass!")
    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    print(model_name+" test save load weight pass!")

    # try:
    #     save_model(model,  name + '.h5')
    #     model = load_model(name + '.h5', custom_objects)
    #     print(name + " test save load model pass!")
    # except:
    #     print("【Error】There is a bug when save model use Dice---------------------------------------------------")

    print(model_name + " test pass!")


def test_DIEN_neg():
    model_name = "DIEN_neg"

    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd(use_neg=True)

    model = DIEN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8,
                hidden_size=[4, 4, 4], keep_prob=0.6,gru_type="AUGRU",use_negsampling=True)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    model.fit(x, y, verbose=1, validation_split=0.5)

    print(model_name+" test train valid pass!")
    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    print(model_name+" test save load weight pass!")

    # try:
    #     save_model(model,  name + '.h5')
    #     model = load_model(name + '.h5', custom_objects)
    #     print(name + " test save load model pass!")
    # except:
    #     print("【Error】There is a bug when save model use Dice---------------------------------------------------")

    print(model_name + " test pass!")

if __name__ == "__main__":
    test_DIEN()
