import pytest

from deepctr.models import MMOE
from deepctr.layers import custom_objects, MultiLossLayer
from ..utils import check_model, get_test_data, SAMPLE_SIZE
import os
from tensorflow.python.keras.models import load_model, save_model, Model


@pytest.mark.parametrize(
    'num_tasks, tasks, use_uncertainty, loss_weights',
    [(2, ['binary','regression'], True, None),
     (3, ['binary', 'binary', 'regression'], False, [0.2, 0.3, 0.5])
     ]
)
def test_MMOE(num_tasks, tasks, use_uncertainty, loss_weights):
    model_name = "MMOE"
    sample_size = SAMPLE_SIZE
    sparse_feature_num = 3
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
                                          dense_feature_num=sparse_feature_num)


    if use_uncertainty:
        _, model = MMOE(feature_columns, num_tasks=num_tasks, tasks=tasks,
                          expert_dim=4, use_uncertainty=use_uncertainty, dnn_hidden_units=[4, 4])
        model.compile('adam', loss=None)
        model_x = list(x.values()) + ([y] * num_tasks)
        model.fit(model_x, epochs=1, validation_split=0.5, steps_per_epoch=int(SAMPLE_SIZE/2), validation_steps=int(SAMPLE_SIZE/2))
    else:
        model = MMOE(feature_columns, num_tasks=num_tasks, tasks=tasks,
                          expert_dim=4, use_uncertainty=use_uncertainty, dnn_hidden_units=[4, 4])
        loss_list = []
        for task in tasks:
            if task == 'binary':
                loss_list.append('binary_crossentropy')
            elif task == 'regression':
                loss_list.append('mean_squared_error')
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

if __name__ == "__main__":
    pass
