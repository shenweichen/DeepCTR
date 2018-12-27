import numpy as np
import pytest
from deepctr.models import AutoInt
from ..utils import check_model


@pytest.mark.parametrize(
    'att_layer_num,hidden_size,sparse_feature_num',
    [(0, (4,), 2), (1, (), 1), (1, (4,), 1), (2, (4, 4,), 2)]
)
def test_AutoInt(att_layer_num, hidden_size, sparse_feature_num):
    model_name = "AutoInt"
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

    model = AutoInt(feature_dim_dict,  att_layer_num=att_layer_num,
                    hidden_size=hidden_size, keep_prob=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_AutoInt(True, (32, 32), 2)
