import pytest

from deepctr.models import MMOE
from deepctr.layers import custom_objects
from ..utils import check_model, get_test_data,SAMPLE_SIZE
import os
from tensorflow.python.keras.models import load_model, save_model

@pytest.mark.parametrize(
    'num_tasks, tasks',
    [(2, ['binary','regression']), (3, ['binary', 'binary', 'regression'])
     ]
)
def test_MMOE(num_tasks, tasks):
    model_name = "MMOE"
    sample_size = SAMPLE_SIZE
    sparse_feature_num = 3
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)
    model = MMOE(feature_columns, num_tasks=num_tasks, tasks=tasks,
                 expert_dim=4, dnn_hidden_units=[4, 4])
    loss_list = []
    for task in tasks:
        if task == 'binary':
            loss_list.append('binary_crossentropy')
        elif task == 'regression':
            loss_list.append('mean_squared_error')
    loss_weights = [1 / num_tasks] * num_tasks
    model.compile('adam',
                  loss=loss_list,
                  metrics=loss_list,
                  loss_weights=loss_weights)
    model.fit(x, [y] * num_tasks, batch_size=100, epochs=1, validation_split=0.5)

    print(model_name + " test train valid pass!")
    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    os.remove(model_name + '_weights.h5')
    print(model_name + " test save load weight pass!")
    save_model(model, model_name + '.h5')
    model = load_model(model_name + '.h5', custom_objects)
    os.remove(model_name + '.h5')
    print(model_name + " test save load model pass!")

    print(model_name + " test pass!")

    # check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
