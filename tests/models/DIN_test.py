import numpy as np
import pytest
from deepctr.models.din import DIN
from deepctr.layers.activation import Dice
from deepctr.utils import SingleFeat
from deepctr.layers import custom_objects
from tensorflow.python.keras.models import load_model, save_model


def get_xy_fd():
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


    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score}

    x = [feature_dict[feat.name] for feat in feature_dim_dict["sparse"]] + [feature_dict[feat.name] for feat in feature_dim_dict["dense"]] + [feature_dict['hist_'+feat] for feat in behavior_feature_list]

    y = [1, 0, 1]
    return x, y, feature_dim_dict, behavior_feature_list


@pytest.mark.xfail(reason="There is a bug when save model use Dice")
# @pytest.mark.skip(reason="misunderstood the API")
def xtest_DIN_model_io():

    model_name = "DIN_att"
    _, _, feature_dim_dict, behavior_feature_list = get_xy_fd()

    model = DIN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8, att_activation=Dice,

                hidden_size=[4, 4, 4], keep_prob=0.6,)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
   #model.fit(x, y, verbose=1, validation_split=0.5)
    save_model(model,  model_name + '.h5')
    model = load_model(model_name + '.h5', custom_objects)
    print(model_name + " test save load model pass!")


def test_DIN_att():
    model_name = "DIN_att"

    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd()

    model = DIN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8,
                hidden_size=[4, 4, 4], keep_prob=0.6,)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, verbose=1, validation_split=0.5)

    print(model_name+" test train valid pass!")
    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    print(model_name+" test save load weight pass!")

    print(model_name + " test pass!")


if __name__ == "__main__":
    test_DIN_att()
