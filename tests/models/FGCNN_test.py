import pytest

from deepctr.models import FGCNN
from tests.utils import check_model, get_test_data, SAMPLE_SIZE


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(1, 1), (3, 3)
     ]
)
def test_FGCNN(sparse_feature_num, dense_feature_num):
    model_name = "FGCNN"

    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, embedding_size=8, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=dense_feature_num)

    model = FGCNN(feature_columns, feature_columns, conv_kernel_width=(3, 2), conv_filters=(2, 1), new_maps=(
        2, 2), pooling_width=(2, 2), dnn_hidden_units=(32,), dnn_dropout=0.5, )
    # TODO: add model_io check
    check_model(model, model_name, x, y, check_model_io=False)


# @pytest.mark.parametrize(
#     'sparse_feature_num,dense_feature_num',
#     [(2, 1),
#      ]
# )
# def test_FGCNN_without_seq(sparse_feature_num, dense_feature_num):
#     model_name = "FGCNN_noseq"
#
#     sample_size = SAMPLE_SIZE
#     x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
#                                           dense_feature_num=dense_feature_num, sequence_feature=())
#
#     model = FGCNN(feature_columns, feature_columns, conv_kernel_width=(), conv_filters=(
#     ), new_maps=(), pooling_width=(), dnn_hidden_units=(32,), dnn_dropout=0.5, )
#     # TODO: add model_io check
#     check_model(model, model_name, x, y, check_model_io=False)


if __name__ == "__main__":
    pass
