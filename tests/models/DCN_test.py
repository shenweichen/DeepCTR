import numpy as np
import pytest
from deepctr.models import DCN
from deepctr.utils import custom_objects
from tensorflow.python.keras.models import save_model, load_model


@pytest.mark.parametrize(
    'embedding_size,cross_num,hidden_size,sparse_feature_num',
    [(8, 0, (32,), 2), (8, 1, (), 1), ('auto', 1, (32,), 3)
     ]
)
def test_DCN(embedding_size, cross_num, hidden_size, sparse_feature_num):
    model_name = "DCN"

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

    model = DCN(feature_dim_dict, embedding_size=embedding_size, cross_num=cross_num,
                hidden_size=hidden_size, keep_prob=0.5, )
    check_model(model, model_name, x, y)


def test_DCN_invalid(embedding_size=8, cross_num=0, hidden_size=()):
    feature_dim_dict = {'sparse': {'sparse_1': 2, 'sparse_2': 5,
                                   'sparse_3': 10}, 'dense': ['dense_1', 'dense_2', 'dense_3']}
    with pytest.raises(ValueError):
        _ = DCN(feature_dim_dict, embedding_size=embedding_size, cross_num=cross_num,
                hidden_size=hidden_size, keep_prob=0.5, )


if __name__ == "__main__":
    test_DCN(8, 2, [32, 32], 2)
