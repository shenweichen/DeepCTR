import pytest
import tensorflow as tf

from deepctr.models import DeepFEFM
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, TEST_Estimator


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num,use_fefm,use_linear,use_fefm_embed_in_dnn',
    [((2,), 1, True, True, True),
     ((2,), 1, True, True, False),
     ((2,), 1, True, False, True),
     ((2,), 1, False, True, True),
     ((2,), 1, True, False, False),
     ((2,), 1, False, True, False),
     ((2,), 1, False, False, True),
     ((2,), 1, False, False, False),
     ((), 1, True, True, True)
     ]
)
def test_DeepFEFM(hidden_size, sparse_feature_num, use_fefm, use_linear, use_fefm_embed_in_dnn):
    if tf.__version__ == "1.15.0" or tf.__version__ == "1.4.0":  # slow in tf 1.15
        return
    model_name = "DeepFEFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = DeepFEFM(feature_columns, feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5,
                     use_linear=use_linear, use_fefm=use_fefm, use_fefm_embed_in_dnn=use_fefm_embed_in_dnn)

    check_model(model, model_name, x, y)


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 2),
     ((), 2),
     ]
)
def test_DeepFEFMEstimator(hidden_size, sparse_feature_num):
    import tensorflow as tf
    if not TEST_Estimator or tf.__version__ == "1.4.0":
        return
    from deepctr.estimator import DeepFEFMEstimator

    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num)

    model = DeepFEFMEstimator(linear_feature_columns, dnn_feature_columns,
                              dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
