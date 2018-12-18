from __future__ import absolute_import

from __future__ import division

from __future__ import print_function


import numpy as np

from numpy.testing import assert_allclose


from generic_utils import has_arg

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

from tensorflow.python.keras import backend as K


def get_test_data(num_train=1000, num_test=500, input_shape=(10,),

                  output_shape=(2,),

                  classification=True, num_classes=2):
    """Generates test data to train a model on.



    classification=True overrides output_shape

    (i.e. output_shape is set to (1,)) and the output

    consists in integers in [0, num_classes-1].



    Otherwise: float output with shape output_shape.

    """

    samples = num_train + num_test

    if classification:

        y = np.random.randint(0, num_classes, size=(samples,))

        X = np.zeros((samples,) + input_shape, dtype=np.float32)

        for i in range(samples):

            X[i] = np.random.normal(loc=y[i], scale=0.7, size=input_shape)

    else:

        y_loc = np.random.random((samples,))

        X = np.zeros((samples,) + input_shape, dtype=np.float32)

        y = np.zeros((samples,) + output_shape, dtype=np.float32)

        for i in range(samples):

            X[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)

            y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (X[:num_train], y[:num_train]), (X[num_train:], y[num_train:])


def layer_test(layer_cls, kwargs={}, input_shape=None, input_dtype=None,

               input_data=None, expected_output=None,

               expected_output_dtype=None, fixed_batch_size=False):
    """Test routine for a layer with a single input tensor

    and single output tensor.

    """

    # generate input data

    if input_data is None:

        assert input_shape

        if not input_dtype:

            input_dtype = K.floatx()

        input_data_shape = list(input_shape)

        for i, e in enumerate(input_data_shape):

            if e is None:

                input_data_shape[i] = np.random.randint(1, 4)

        if all(isinstance(e, tuple) for e in input_data_shape):
            input_data = []
            for e in input_data_shape:
                input_data.append(
                    (10 * np.random.random(e)).astype(input_dtype))

        else:

            input_data = (10 * np.random.random(input_data_shape))

            input_data = input_data.astype(input_dtype)


    else:

        if input_shape is None:

            input_shape = input_data.shape

        if input_dtype is None:

            input_dtype = input_data.dtype

    if expected_output_dtype is None:

        expected_output_dtype = input_dtype

    # instantiation

    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level

    weights = layer.get_weights()

    layer.set_weights(weights)

    try:
        expected_output_shape = layer.compute_output_shape(input_shape)
    except:
        expected_output_shape = layer._compute_output_shape(input_shape)

    # test in functional API
    if isinstance(input_shape, list):
        if fixed_batch_size:

            x = [Input(batch_shape=e, dtype=input_dtype) for e in input_shape]

        else:

            x = [Input(shape=e[1:], dtype=input_dtype) for e in input_shape]
    else:
        if fixed_batch_size:

            x = Input(batch_shape=input_shape, dtype=input_dtype)

        else:

            x = Input(shape=input_shape[1:], dtype=input_dtype)

    y = layer(x)

    assert K.dtype(y) == expected_output_dtype

    # check with the functional API

    model = Model(x, y)

    actual_output = model.predict(input_data)

    actual_output_shape = actual_output.shape

    for expected_dim, actual_dim in zip(expected_output_shape,

                                        actual_output_shape):

        if expected_dim is not None:

            assert expected_dim == actual_dim

    if expected_output is not None:

        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level

    model_config = model.get_config()

    recovered_model = model.__class__.from_config(model_config)

    if model.weights:

        weights = model.get_weights()

        recovered_model.set_weights(weights)

        _output = recovered_model.predict(input_data)

        assert_allclose(_output, actual_output, rtol=1e-3)

    # test training mode (e.g. useful when the layer has a

    # different behavior at training and testing time).

    if has_arg(layer.call, 'training'):

        model.compile('rmsprop', 'mse')

        model.train_on_batch(input_data, actual_output)

    # test instantiation from layer config

    layer_config = layer.get_config()

    layer_config['batch_input_shape'] = input_shape

    layer = layer.__class__.from_config(layer_config)

    # for further checks in the caller function

    return actual_output
