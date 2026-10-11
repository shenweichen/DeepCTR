"""Release Keras graphs between tests in the full model matrix."""
import gc

import pytest
from tensorflow.keras import backend as K


@pytest.fixture(autouse=True)
def clear_keras_session():
    yield
    K.clear_session()
    gc.collect()
