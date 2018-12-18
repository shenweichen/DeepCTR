import numpy as np
import pytest

from deepctr.models import PNN
from deepctr.utils import custom_objects
from tensorflow.python.keras.models import save_model, load_model



@pytest.mark.parametrize(
    'use_inner, use_outter',
    [(True, True), (True, False), (False, True), (False, False)
     ]
)
def test_PNN(use_inner, use_outter):
    model_name = "PNN"
    sample_size = 64
    feature_dim_dict = {'sparse': {'sparse_1': 2, 'sparse_2': 5,
                                   'sparse_3': 10}, 'dense': ['dense_1', 'dense_2', 'dense_3']}
    sparse_input = [np.random.randint(0, dim, sample_size)
                    for dim in feature_dim_dict['sparse'].values()]
    dense_input = [np.random.random(sample_size)
                   for name in feature_dim_dict['dense']]
    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input

    model = PNN(feature_dim_dict, embedding_size=8,
                hidden_size=[32, 32], keep_prob=0.5, use_inner=use_inner, use_outter=use_outter)
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)
    print(model_name+" test train valid pass!")
    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    print(model_name+" test save load weight pass!")
    save_model(model,  model_name + '.h5')
    model = load_model(model_name + '.h5', custom_objects)
    print(model_name + " test save load model pass!")

    print(model_name + " test pass!")


if __name__ == "__main__":
    test_PNN(use_inner=True, use_outter=False)