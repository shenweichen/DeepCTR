import numpy as np
import pytest
from deepctr.models import MLR
from deepctr.utils import custom_objects
from tensorflow.python.keras.models import save_model, load_model


@pytest.mark.parametrize(

    'region_sparse,region_dense,base_sparse,base_dense,bias_sparse,bias_dense',

    [(0, 2, 0, 2, 0, 1), (0, 2, 0, 1, 0, 2), (0, 2, 0, 0, 1, 0),
     (0, 1, 1, 2, 1, 1,), (0, 1, 1, 1, 1, 2), (0, 1, 1, 0, 2, 0),
     (1, 0, 2, 2, 2, 1), (2, 0, 2, 1, 2, 2), (2, 0, 2, 0, 0, 0)
     ]

)
def test_MLRs(region_sparse, region_dense, base_sparse, base_dense, bias_sparse, bias_dense):
    model_name = "MLRs"
    region_fd = {"sparse": {}, 'dense': []}
    for name, num in zip(["sparse", "dense"], [region_sparse, region_dense]):
        if name == "sparse":
            for i in range(num):
                region_fd[name][name + '_' + str(i)] = np.random.randint(1, 10)
        else:
            for i in range(num):
                region_fd[name].append(name + '_' + str(i))

    base_fd = {"sparse": {}, 'dense': []}
    for name, num in zip(["sparse", "dense"], [base_sparse, base_dense]):
        if name == "sparse":
            for i in range(num):
                base_fd[name][name + '_' + str(i)] = np.random.randint(1, 10)
        else:
            for i in range(num):
                base_fd[name].append(name + '_' + str(i))
    bias_fd = {"sparse": {}, 'dense': []}
    for name, num in zip(["sparse", "dense"], [bias_sparse, bias_dense]):
        if name == "sparse":
            for i in range(num):
                bias_fd[name][name + '_' + str(i)] = np.random.randint(1, 10)
        else:
            for i in range(num):
                bias_fd[name].append(name + '_' + str(i))

    model = MLR(region_fd, base_fd, bias_feature_dim_dict=bias_fd)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    print(model_name + " test pass!")


def test_MLR():
    model_name = "MLR"
    sample_size = 64
    feature_dim_dict = {'sparse': {'sparse_1': 2, 'sparse_2': 5,
                                   'sparse_3': 10}, 'dense': ['dense_1', 'dense_2', 'dense_3']}
    sparse_input = [np.random.randint(0, dim, sample_size)
                    for dim in feature_dim_dict['sparse'].values()]
    dense_input = [np.random.random(sample_size)
                   for name in feature_dim_dict['dense']]
    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input

    model = MLR(feature_dim_dict)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    test_MLR()
