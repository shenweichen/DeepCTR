import numpy as np
import pytest
from deepctr.models import MLR
from ..utils import check_model, SAMPLE_SIZE
from deepctr.inputs import SparseFeat


@pytest.mark.parametrize(

    'region_sparse,region_dense,base_sparse,base_dense,bias_sparse,bias_dense',

    [(0, 2, 0, 2, 0, 1), (0, 2, 0, 1, 0, 2), (0, 2, 0, 0, 1, 0),
     (0, 1, 1, 2, 1, 1,), (0, 1, 1, 1, 1, 2), (0, 1, 1, 0, 2, 0),
     (1, 0, 2, 2, 2, 1), (2, 0, 2, 1, 2, 2), (2, 0, 2, 0, 0, 0)
     ]

)
def test_MLRs(region_sparse, region_dense, base_sparse, base_dense, bias_sparse, bias_dense):
    model_name = "MLRs"
    region_fd = {"sparse": [], 'dense': []}
    for name, num in zip(["sparse", "dense"], [region_sparse, region_dense]):
        if name == "sparse":
            for i in range(num):
                region_fd[name].append(SparseFeat(name + '_' + str(i),np.random.randint(1, 10)))
        else:
            for i in range(num):
                region_fd[name].append(SparseFeat(name + '_' + str(i),0))

    base_fd = {"sparse": [], 'dense': []}
    for name, num in zip(["sparse", "dense"], [base_sparse, base_dense]):
        if name == "sparse":
            for i in range(num):
                base_fd[name].append(SparseFeat(name + '_' + str(i),np.random.randint(1, 10)))
        else:
            for i in range(num):
                base_fd[name].append(SparseFeat(name + '_' + str(i),0))
    bias_fd = {"sparse": [], 'dense': []}
    for name, num in zip(["sparse", "dense"], [bias_sparse, bias_dense]):
        if name == "sparse":
            for i in range(num):
                bias_fd[name].append(SparseFeat(name + '_' + str(i),np.random.randint(1, 10)))
        else:
            for i in range(num):
                bias_fd[name].append(SparseFeat(name + '_' + str(i),0))

    model = MLR(region_fd, base_fd, bias_feature_dim_dict=bias_fd)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    print(model_name + " test pass!")


def test_MLR():
    model_name = "MLR"
    sample_size = SAMPLE_SIZE
    feature_dim_dict = {'sparse': [SparseFeat('sparse_1',2),SparseFeat('sparse_2',5),SparseFeat('sparse_3',10)] ,
                                    'dense': [SparseFeat('dense_1',0),SparseFeat('dense_2',0),SparseFeat('dense_3',0)]}
    sparse_input = [np.random.randint(0, feat.dimension, sample_size)
                    for feat in feature_dim_dict['sparse']]
    dense_input = [np.random.random(sample_size)
                   for _ in feature_dim_dict['dense']]
    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input

    model = MLR(feature_dim_dict)
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
