import pytest

from deepctr.models import DCN
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'cross_num,hidden_size,sparse_feature_num',
    [( 0, (32,), 2), ( 1, (), 1), ( 1, (32,), 3)
     ]
)
def test_DCN( cross_num, hidden_size, sparse_feature_num):
    model_name = "DCN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = DCN(feature_columns,feature_columns, cross_num=cross_num, dnn_hidden_units=hidden_size, dnn_dropout=0.5)
    check_model(model, model_name, x, y)


# def test_DCN_invalid(embedding_size=8, cross_num=0, hidden_size=()):
#     feature_dim_dict = {'sparse': [SparseFeat('sparse_1', 2), SparseFeat('sparse_2', 5), SparseFeat('sparse_3', 10)],
#                         'dense': [SparseFeat('dense_1', 1), SparseFeat('dense_1', 1), SparseFeat('dense_1', 1)]}
#     with pytest.raises(ValueError):
#         _ = DCN(None, embedding_size=embedding_size, cross_num=cross_num, dnn_hidden_units=hidden_size, dnn_dropout=0.5)


if __name__ == "__main__":
    pass
