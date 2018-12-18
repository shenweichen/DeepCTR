import numpy as np
import pytest
from deepctr.models import DCN
from deepctr.utils import custom_objects
from tensorflow.python.keras.models import save_model, load_model


@pytest.mark.parametrize(
    'embedding_size,cross_num,hidden_size',
    [(embedding_size, cross_num, hidden_size)
     for embedding_size in ['auto', 8]
     for cross_num in [0, 1, ]
     for hidden_size in [(), (32,)]
     if cross_num > 0 or len(hidden_size) > 0
     ]
)
def test_DCN(embedding_size, cross_num, hidden_size):
    name = "DCN"

    sample_size = 64
    feature_dim_dict = {'sparse': {'sparse_1': 2, 'sparse_2': 5,
                                   'sparse_3': 10}, 'dense': ['dense_1', 'dense_2', 'dense_3']}
    sparse_input = [np.random.randint(0, dim, sample_size)
                    for dim in feature_dim_dict['sparse'].values()]
    dense_input = [np.random.random(sample_size)
                   for name in feature_dim_dict['dense']]
    y = np.random.randint(0, 2, sample_size)
    x = sparse_input + dense_input

    model = DCN(feature_dim_dict, embedding_size=embedding_size, cross_num=cross_num,
                hidden_size=hidden_size, keep_prob=0.5, )
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)
    print(name+" test train valid pass!")
    model.save_weights(name + '_weights.h5')
    model.load_weights(name + '_weights.h5')
    print(name+" test save load weight pass!")
    save_model(model, name + '.h5')
    model = load_model(name + '.h5', custom_objects)
    print(name + " test save load model pass!")

    print(name + " test pass!")


def test_DCN_invalid(embedding_size=8, cross_num=0, hidden_size=()):
    feature_dim_dict = {'sparse': {'sparse_1': 2, 'sparse_2': 5,
                                   'sparse_3': 10}, 'dense': ['dense_1', 'dense_2', 'dense_3']}
    with pytest.raises(ValueError):
        _ = DCN(feature_dim_dict, embedding_size=embedding_size, cross_num=cross_num,
                    hidden_size=hidden_size, keep_prob=0.5, )


if __name__ == "__main__":
    test_DCN(8, 2, [32, 32])
