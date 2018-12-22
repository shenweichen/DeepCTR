import numpy as np
import pytest

from deepctr.models import PNN
from deepctr.utils import custom_objects
from tensorflow.python.keras.models import save_model, load_model
from ..utils import check_model


@pytest.mark.parametrize(
    'use_inner, use_outter,sparse_feature_num',
    [(True, True, 1), (True, False, 2), (False, True, 3), (False, False, 1)
     ]
)
def test_PNN(use_inner, use_outter, sparse_feature_num):
    model_name = "PNN"
    sample_size = 64
    feature_dim_dict = {"sparse": {}, 'dense': []}
    for name, num in zip(["sparse", "dense"], [sparse_feature_num, sparse_feature_num]):
        if name == "sparse":
            for i in range(num):
                feature_dim_dict[name][name + '_' +
                                       str(i)] = np.random.randint(1, 10)
        else:
            for i in range(num):
                feature_dim_dict[name].append(name + '_' + str(i))
    sparse_input = [np.random.randint(0, dim, sample_size)
                    for dim in feature_dim_dict['sparse'].values()]
    dense_input = [np.random.random(sample_size)
                   for name in feature_dim_dict['dense']]
    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input

    model = PNN(feature_dim_dict, embedding_size=8,
                hidden_size=[32, 32], keep_prob=0.5, use_inner=use_inner, use_outter=use_outter)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_PNN(use_inner=True, use_outter=False, sparse_feature_num=1)
