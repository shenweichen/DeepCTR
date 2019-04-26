import pytest

from deepctr.models import FGCNN
from tests.utils import check_model, get_test_data,SAMPLE_SIZE


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(1, 1), (3, 3)
     ]
)
def test_FGCNN(sparse_feature_num, dense_feature_num):
    model_name = "FGCNN"

    sample_size = 32
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num)

    model = FGCNN(feature_dim_dict, conv_kernel_width=(3, 2), conv_filters=(2, 1), hidden_size=[32, ], keep_prob=0.5, )
    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(1, 1),
     ]
)
def test_CCPM_without_seq(sparse_feature_num, dense_feature_num):
    model_name = "FGCNN"

    sample_size = SAMPLE_SIZE
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num, sequence_feature=())

    model = FGCNN(feature_dim_dict, conv_kernel_width=(3, ), conv_filters=(2, ), hidden_size=[32, ], keep_prob=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
