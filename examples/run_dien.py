import numpy as np

from deepctr.models import DIEN
from deepctr.inputs import SparseFeat,DenseFeat,VarLenSparseFeat,get_fixlen_feature_names,get_varlen_feature_names


def get_xy_fd(use_neg=False, hash_flag=False):

    feature_columns = [SparseFeat('user', 3,hash_flag),
                       SparseFeat('gender', 2,hash_flag),
                       SparseFeat('item', 3+1,hash_flag),
                       SparseFeat('item_gender', 2+1,hash_flag),
                       DenseFeat('score', 1)]

    feature_columns += [VarLenSparseFeat('hist_item',3+1, maxlen=4, embedding_name='item'),
                        VarLenSparseFeat('hist_item_gender',3+1, maxlen=4, embedding_name='item_gender')]

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

    if use_neg:
        feature_dict['neg_hist_item'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_item_gender'] = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
        feature_columns += [VarLenSparseFeat('neg_hist_item',3+1, maxlen=4, embedding_name='item'),
                        VarLenSparseFeat('neg_hist_item_gender',3+1, maxlen=4, embedding_name='item_gender')]
        #x += [feature_dict['neg_hist_'+feat] for feat in behavior_feature_list]


    fix_feature_names = get_fixlen_feature_names(feature_columns)
    varlen_feature_names = get_varlen_feature_names(feature_columns)
    x = [feature_dict[name] for name in fix_feature_names] + [feature_dict[name] for name in varlen_feature_names]

    x += [behavior_length]
    y = [1, 0, 1]
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd(use_neg=True)
    model = DIEN(feature_columns, behavior_feature_list, hist_len_max=4, embedding_size=8,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.6, gru_type="AUGRU", use_negsampling=True)

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
