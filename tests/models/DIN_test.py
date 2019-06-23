import numpy as np

from deepctr.models.din import DIN
from deepctr.inputs import SparseFeat
from ..utils import check_model



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

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score}

    x = [feature_dict[feat.name] for feat in feature_dim_dict["sparse"]] + [feature_dict[feat.name] for feat in
                                                                            feature_dim_dict["dense"]] + [
            feature_dict['hist_' + feat] for feat in behavior_feature_list]

    y = [1, 0, 1]
    return x, y, feature_dim_dict, behavior_feature_list


#@pytest.mark.xfail(reason="There is a bug when save model use Dice")
#@pytest.mark.skip(reason="misunderstood the API")


def test_DIN():
    model_name = "DIN"

    x, y, feature_dim_dict, behavior_feature_list = get_xy_fd(True)

    model = DIN(feature_dim_dict, behavior_feature_list, hist_len_max=4, embedding_size=8,
                dnn_hidden_units=[4, 4, 4], dnn_dropout=0.5, )

    check_model(model,model_name,x,y)


if __name__ == "__main__":
    pass
