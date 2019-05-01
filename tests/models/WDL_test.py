import numpy as np
import pytest

from deepctr.models import WDL
from ..utils import check_model,SingleFeat,SAMPLE_SIZE


@pytest.mark.parametrize(
    'sparse_feature_num,wide_feature_num',
    [(1, 0), (1, 2), (2, 0), (2, 1)
     ]
)
def test_WDL(sparse_feature_num, wide_feature_num):
    model_name = "WDL"
    sample_size = SAMPLE_SIZE
    feature_dim_dict = {"sparse": [], 'dense': []}
    wide_feature_dim_dict = {"sparse": [], 'dense': []}
    for name, num in zip(["sparse", "dense"], [sparse_feature_num, sparse_feature_num]):
        if name == "sparse":
            for i in range(num):
                feature_dim_dict[name].append(SingleFeat(name + '_' +str(i),np.random.randint(1, 10)))
        else:
            for i in range(num):
                feature_dim_dict[name].append(SingleFeat(name + '_' + str(i),0))
    for name, num in zip(["sparse", "dense"], [wide_feature_num, wide_feature_num]):
        if name == "sparse":
            for i in range(num):
                wide_feature_dim_dict[name].append(SingleFeat(name + 'wide_' +str(i),np.random.randint(1, 10)))
        else:
            for i in range(num):
                wide_feature_dim_dict[name].append(SingleFeat(name + 'wide_' + str(i),0))

    sparse_input = [np.random.randint(0, feat.dimension, sample_size)
                    for feat in feature_dim_dict['sparse']]
    dense_input = [np.random.random(sample_size)
                   for _ in feature_dim_dict['dense']]
    wide_sparse_input = [np.random.randint(0, feat.dimension, sample_size)
                         for feat in wide_feature_dim_dict['sparse']]
    wide_dense_input = [np.random.random(sample_size)
                        for _ in wide_feature_dim_dict['dense']]
    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input
    x_wide = wide_sparse_input + wide_dense_input

    model = WDL(feature_dim_dict, wide_feature_dim_dict,
                hidden_size=[32, 32], keep_prob=0.5)
    check_model(model, model_name, x+x_wide, y)


if __name__ == "__main__":
    pass
