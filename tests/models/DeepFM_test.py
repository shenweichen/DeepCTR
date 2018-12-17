import numpy as np
import pytest
from deepctr.models import DeepFM
from deepctr.utils import custom_objects
from tensorflow.python.keras.models import save_model, load_model


@pytest.mark.parametrize(
    'use_fm,hidden_size',
    [(use_fm, hidden_size)
     for use_fm in [True, False, ]
     for hidden_size in [(), (32,)]
     ]
)
def test_DeepFM(use_fm, hidden_size):
    name = "DeepFM"
    sample_size = 64
    feature_dim_dict = {'sparse': {'sparse_1': 2, 'sparse_2': 5,
                                   'sparse_3': 10}, 'dense': ['dense_1', 'dense_2', 'dense_3']}
    sparse_input = [np.random.randint(0, dim, sample_size)
                    for dim in feature_dim_dict['sparse'].values()]
    dense_input = [np.random.random(sample_size)
                   for name in feature_dim_dict['dense']]
    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input

    model = DeepFM(feature_dim_dict,  use_fm=use_fm,
                   hidden_size=hidden_size, keep_prob=0.5, )
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)
    print(name+" test train valid pass!")
    model.save_weights(name + '_weights.h5')
    model.load_weights(name + '_weights.h5')
    print(name+" test save load weight pass!")
    save_model(model,  name + '.h5')
    model = load_model(name + '.h5', custom_objects)
    print(name + " test save load model pass!")

    print(name + " test pass!")


if __name__ == "__main__":
    test_DeepFM(True, (32, 32))
