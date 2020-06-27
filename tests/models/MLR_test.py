import pytest

from deepctr.models import MLR
from ..utils import check_model, SAMPLE_SIZE, get_test_data


@pytest.mark.parametrize(

    'region_sparse,region_dense,base_sparse,base_dense,bias_sparse,bias_dense',

    [(0, 2, 0, 2, 0, 1), (0, 2, 0, 1, 0, 2), (0, 2, 0, 0, 1, 0),
     #     (0, 1, 1, 2, 1, 1,), (0, 1, 1, 1, 1, 2), (0, 1, 1, 0, 2, 0),
     #     (1, 0, 2, 2, 2, 1), (2, 0, 2, 1, 2, 2), (2, 0, 2, 0, 0, 0)
     ]

)
def test_MLRs(region_sparse, region_dense, base_sparse, base_dense, bias_sparse, bias_dense):
    model_name = "MLRs"
    region_x, y, region_feature_columns = get_test_data(SAMPLE_SIZE, sparse_feature_num=region_sparse,
                                                        dense_feature_num=region_dense, prefix='region')
    base_x, y, base_feature_columns = get_test_data(SAMPLE_SIZE, sparse_feature_num=region_sparse,
                                                    dense_feature_num=region_dense, prefix='base')
    bias_x, y, bias_feature_columns = get_test_data(SAMPLE_SIZE, sparse_feature_num=region_sparse,
                                                    dense_feature_num=region_dense, prefix='bias')

    model = MLR(region_feature_columns, base_feature_columns, bias_feature_columns=bias_feature_columns)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    print(model_name + " test pass!")


def test_MLR():
    model_name = "MLR"
    region_x, y, region_feature_columns = get_test_data(SAMPLE_SIZE, sparse_feature_num=3, dense_feature_num=3,
                                                        prefix='region')
    base_x, y, base_feature_columns = get_test_data(SAMPLE_SIZE, sparse_feature_num=3, dense_feature_num=3,
                                                    prefix='base')
    bias_x, y, bias_feature_columns = get_test_data(SAMPLE_SIZE, sparse_feature_num=3, dense_feature_num=3,
                                                    prefix='bias')

    model = MLR(region_feature_columns)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])

    check_model(model, model_name, region_x, y)
    print(model_name + " test pass!")


if __name__ == "__main__":
    pass
