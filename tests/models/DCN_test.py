import pytest

from deepctr.utils import SingleFeat
from deepctr.models import DCN
from ..utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'embedding_size,cross_num,hidden_size,sparse_feature_num',
    [(8, 0, (32,), 2), ('auto', 1, (), 1), ('auto', 1, (32,), 3)
     ]
)
def test_DCN(embedding_size, cross_num, hidden_size, sparse_feature_num):
    model_name = "DCN"

    sample_size = SAMPLE_SIZE
    x, y, feature_dim_dict = get_test_data(sample_size, sparse_feature_num, sparse_feature_num)

    model = DCN(feature_dim_dict, embedding_size=embedding_size, cross_num=cross_num,
                dnn_hidden_units=hidden_size, dnn_dropout=0.5, )
    check_model(model, model_name, x, y)


def test_DCN_invalid(embedding_size=8, cross_num=0, hidden_size=()):
    feature_dim_dict = {'sparse': [SingleFeat('sparse_1', 2), SingleFeat('sparse_2', 5), SingleFeat('sparse_3', 10)],
                        'dense': [SingleFeat('dense_1', 1), SingleFeat('dense_1', 1), SingleFeat('dense_1', 1)]}
    with pytest.raises(ValueError):
        _ = DCN(feature_dim_dict, embedding_size=embedding_size, cross_num=cross_num,
                dnn_hidden_units=hidden_size, dnn_dropout=0.5, )


if __name__ == "__main__":
    pass
