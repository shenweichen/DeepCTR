import numpy as np
import pytest
from deepctr.models import AFM
from ..utils import check_model


@pytest.mark.parametrize(
    'use_attention,sparse_feature_num',
    [(True, 1), (False, 3)
     ]
)
def test_AFM(use_attention, sparse_feature_num):
    model_name = "AFM"

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

    model = AFM(feature_dim_dict, use_attention=use_attention, keep_prob=0.5,)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_AFM(use_attention=True, sparse_feature_num=2)
