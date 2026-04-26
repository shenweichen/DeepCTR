import pytest
import numpy as np
import os
import tempfile
from tensorflow.keras.models import load_model, save_model

from deepctr.feature_column import DenseFeat
from deepctr.layers import custom_objects
from deepctr.models import DeepFM
from ..utils import check_model, get_test_data, SAMPLE_SIZE, get_test_data_estimator, check_estimator, TEST_Estimator


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [((2,), 1),  #
     ((3,), 2)
     ]  # (True, (32,), 3), (False, (32,), 1)
)
def test_DeepFM(hidden_size, sparse_feature_num):
    model_name = "DeepFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)

    model = DeepFM(feature_columns, feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5)

    check_model(model, model_name, x, y)


def test_DeepFM_dense_only_model_io():
    sample_size = SAMPLE_SIZE
    feature_columns = [DenseFeat('dense_feature_' + str(i), 1) for i in range(2)]
    x = {fc.name: np.random.random(sample_size) for fc in feature_columns}
    y = np.random.randint(0, 2, (sample_size, 1))

    model = DeepFM(feature_columns, feature_columns, dnn_hidden_units=(4,), dnn_dropout=0)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, batch_size=4, epochs=1, validation_split=0.5)

    fd = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    model_path = fd.name
    fd.close()
    try:
        save_model(model, model_path)
        load_model(model_path, custom_objects)
    finally:
        os.remove(model_path)


@pytest.mark.parametrize(
    'hidden_size,sparse_feature_num',
    [
        ((3,), 2)
    ]  # (True, (32,), 3), (False, (32,), 1)
)
def test_DeepFMEstimator(hidden_size, sparse_feature_num):
    if not TEST_Estimator:
        return
    from deepctr.estimator import DeepFMEstimator
    sample_size = SAMPLE_SIZE
    linear_feature_columns, dnn_feature_columns, input_fn = get_test_data_estimator(sample_size,
                                                                                    sparse_feature_num=sparse_feature_num,
                                                                                    dense_feature_num=sparse_feature_num,
                                                                                    classification=False)

    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=hidden_size, dnn_dropout=0.5,
                            task="regression")

    check_estimator(model, input_fn)


if __name__ == "__main__":
    pass
