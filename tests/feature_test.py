from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr.inputs import create_embedding_matrix
import numpy as np
import pytest


def test_long_dense_vector():
    feature_columns = [SparseFeat('user_id', 4, ), SparseFeat('item_id', 5, ), DenseFeat("pic_vec", 5)]
    fixlen_feature_names = get_feature_names(feature_columns)

    user_id = np.array([[1], [0], [1]])
    item_id = np.array([[3], [2], [1]])
    pic_vec = np.array([[0.1, 0.5, 0.4, 0.3, 0.2], [0.1, 0.5, 0.4, 0.3, 0.2], [0.1, 0.5, 0.4, 0.3, 0.2]])
    label = np.array([1, 0, 1])

    input_dict = {'user_id': user_id, 'item_id': item_id, 'pic_vec': pic_vec}
    model_input = [input_dict[name] for name in fixlen_feature_names]

    model = DeepFM(feature_columns, feature_columns[:-1])
    model.compile('adagrad', 'binary_crossentropy')
    model.fit(model_input, label)


def test_feature_column_sparsefeat_vocabulary_path():
    vocab_path = "./dummy_test.csv"
    sf = SparseFeat('user_id', 4, vocabulary_path=vocab_path)
    if sf.vocabulary_path != vocab_path:
        raise ValueError("sf.vocabulary_path is invalid")
    vlsf = VarLenSparseFeat(sf, 6)
    if vlsf.vocabulary_path != vocab_path:
        raise ValueError("vlsf.vocabulary_path is invalid")


def test_shared_embedding_name_creates_one_sequence_embedding():
    feature_columns = [
        SparseFeat('item', 4, embedding_dim=8),
        VarLenSparseFeat(SparseFeat('hist_item', 4, embedding_dim=8, embedding_name='item'), maxlen=4),
        VarLenSparseFeat(SparseFeat('sess_item', 4, embedding_dim=8, embedding_name='item'), maxlen=4),
    ]

    embedding_dict = create_embedding_matrix(feature_columns, l2_reg=0, seed=1024)

    assert list(embedding_dict.keys()) == ['item']
    assert embedding_dict['item'].name.startswith('sparse_seq_emb_item')
    assert embedding_dict['item'].mask_zero is True


def test_shared_embedding_name_respects_seq_mask_zero():
    feature_columns = [
        SparseFeat('item', 4, embedding_dim=8),
        VarLenSparseFeat(SparseFeat('hist_item', 4, embedding_dim=8, embedding_name='item'), maxlen=4),
    ]

    embedding_dict = create_embedding_matrix(feature_columns, l2_reg=0, seed=1024, seq_mask_zero=False)

    assert embedding_dict['item'].mask_zero is False


def test_sparse_features_can_share_embedding_name():
    feature_columns = [
        SparseFeat('item', 4, embedding_dim=8),
        SparseFeat('query_item', 4, embedding_dim=8, embedding_name='item'),
    ]

    embedding_dict = create_embedding_matrix(feature_columns, l2_reg=0, seed=1024)

    assert list(embedding_dict.keys()) == ['item']
    assert embedding_dict['item'].name.startswith('sparse_emb_item')


def test_shared_embedding_name_requires_same_shape():
    feature_columns = [
        SparseFeat('item', 4, embedding_dim=8),
        VarLenSparseFeat(SparseFeat('hist_item', 5, embedding_dim=8, embedding_name='item'), maxlen=4),
    ]

    with pytest.raises(ValueError):
        create_embedding_matrix(feature_columns, l2_reg=0, seed=1024)
