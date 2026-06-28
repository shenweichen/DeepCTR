"""Reusable correctness contracts for DeepCTR layers and models.

These helpers deliberately test semantics rather than merely checking that a
graph can be built.  A strong test normally combines:

* an independent NumPy reference for the paper equation;
* finite (and, for small tensors, numerical) gradients;
* an invariant such as padding, causality, or permutation behaviour; and
* prediction equivalence after serialization.

The module lives under ``tests`` so the core package keeps no pytest/runtime
dependency on the test framework.
"""
from __future__ import absolute_import, division, print_function

import copy
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model


def set_test_seed(seed=2020):
    """Seed NumPy and TensorFlow for reproducible correctness fixtures."""
    np.random.seed(seed)
    if hasattr(tf.random, "set_seed"):
        tf.random.set_seed(seed)
    else:  # TensorFlow 1.x compatibility
        tf.set_random_seed(seed)


def _invoke(subject, inputs, training=False):
    """Call a Keras layer/model or a plain callable consistently."""
    if hasattr(subject, "call"):
        return subject(inputs, training=training)
    return subject(inputs)


def _to_tensor(value):
    if tf.is_tensor(value):
        return value
    return tf.convert_to_tensor(value)


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    return K.get_value(value)


def _numpy_tree(values):
    return tf.nest.map_structure(_to_numpy, values)


def assert_nested_allclose(actual, expected, rtol=1e-5, atol=1e-6):
    """Assert equality for tensor/array outputs, including nested outputs."""
    actual_flat = tf.nest.flatten(_numpy_tree(actual))
    expected_flat = tf.nest.flatten(_numpy_tree(expected))
    if len(actual_flat) != len(expected_flat):
        raise AssertionError("output structures differ: %d != %d" %
                             (len(actual_flat), len(expected_flat)))
    for index, (actual_value, expected_value) in enumerate(zip(actual_flat, expected_flat)):
        if not np.all(np.isfinite(actual_value)):
            raise AssertionError("output %d contains NaN or infinity" % index)
        assert_allclose(actual_value, expected_value, rtol=rtol, atol=atol,
                        err_msg="output %d differs from reference" % index)


def named_weights(subject):
    """Return layer weights by full variable name and, when unique, leaf name.

    Reference functions can normally use stable leaf names such as ``w1`` or
    ``W_Q`` without depending on Keras-generated layer scopes.  Ambiguous leaf
    names are intentionally omitted; their full names remain available.
    """
    result = {}
    leaf_counts = {}
    values = []
    for variable in subject.weights:
        full_name = variable.name.split(":", 1)[0]
        leaf_name = full_name.rsplit("/", 1)[-1]
        value = _to_numpy(variable)
        result[full_name] = value
        values.append((leaf_name, value))
        leaf_counts[leaf_name] = leaf_counts.get(leaf_name, 0) + 1
    for leaf_name, value in values:
        if leaf_counts[leaf_name] == 1:
            result[leaf_name] = value
    return result


def assign_named_weights(subject, values):
    """Assign a dict of deterministic arrays to weights by full or leaf name."""
    variables = {}
    leaf_counts = {}
    leaves = []
    for variable in subject.weights:
        full_name = variable.name.split(":", 1)[0]
        leaf_name = full_name.rsplit("/", 1)[-1]
        variables[full_name] = variable
        leaves.append((leaf_name, variable))
        leaf_counts[leaf_name] = leaf_counts.get(leaf_name, 0) + 1
    for leaf_name, variable in leaves:
        if leaf_counts[leaf_name] == 1:
            variables[leaf_name] = variable

    unknown = sorted(set(values) - set(variables))
    if unknown:
        raise KeyError("unknown weight names: %s" % ", ".join(unknown))
    for name, value in values.items():
        variable = variables[name]
        value = np.asarray(value, dtype=K.get_value(variable).dtype)
        expected_shape = tuple(int(dim) for dim in variable.shape)
        if value.shape != expected_shape:
            raise ValueError("weight %s has shape %s; expected %s" %
                             (name, value.shape, expected_shape))
        K.set_value(variable, value)


def assert_forward_matches(subject, inputs, reference_fn, weights=None,
                           rtol=1e-5, atol=1e-6):
    """Compare a layer/model forward pass with an independent reference.

    ``reference_fn`` receives ``(numpy_inputs, named_weight_dict)``.  It must not
    call ``subject`` or reuse the implementation under test.  ``weights`` may be
    either a Keras-ordered list or a dict accepted by
    :func:`assign_named_weights`.
    """
    tensor_inputs = tf.nest.map_structure(_to_tensor, inputs)
    _invoke(subject, tensor_inputs, training=False)  # build variables
    if weights is not None:
        if isinstance(weights, dict):
            assign_named_weights(subject, weights)
        else:
            subject.set_weights(weights)
    actual = _invoke(subject, tensor_inputs, training=False)
    expected = reference_fn(_numpy_tree(tensor_inputs), named_weights(subject))
    assert_nested_allclose(actual, expected, rtol=rtol, atol=atol)
    return _numpy_tree(actual)


def assert_finite_gradients(subject, inputs, loss_fn=None, require_all=True):
    """Assert that differentiable inputs and trainable weights have gradients."""
    if not hasattr(tf, "GradientTape") or not tf.executing_eagerly():
        raise RuntimeError("gradient contracts require TensorFlow eager execution")

    watched = []

    def make_input(value):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            variable = tf.Variable(array)
            watched.append(variable)
            return variable
        return tf.convert_to_tensor(array)

    tensor_inputs = tf.nest.map_structure(make_input, inputs)
    with tf.GradientTape() as tape:
        outputs = _invoke(subject, tensor_inputs, training=False)
        if loss_fn is None:
            terms = [tf.reduce_sum(tf.cast(value, tf.float32))
                     for value in tf.nest.flatten(outputs)]
            loss = tf.add_n(terms)
        else:
            loss = loss_fn(outputs)

    targets = watched + list(subject.trainable_weights)
    gradients = tape.gradient(loss, targets)
    for target, gradient in zip(targets, gradients):
        if gradient is None:
            if require_all:
                raise AssertionError("missing gradient for %s" % getattr(target, "name", "input"))
            continue
        gradient_value = _to_numpy(gradient)
        if not np.all(np.isfinite(gradient_value)):
            raise AssertionError("non-finite gradient for %s" % getattr(target, "name", "input"))
    return gradients


def assert_numerical_gradient(subject, inputs, rtol=2e-2, atol=2e-3, delta=1e-3):
    """Compare analytic and finite-difference input Jacobians on tiny tensors.

    This intentionally targets inputs rather than weights; weight gradients are
    covered cheaply by :func:`assert_finite_gradients`.  Keep tensors small,
    because a full Jacobian is materialized.
    """
    if not hasattr(tf.test, "compute_gradient") or not tf.executing_eagerly():
        raise RuntimeError("numerical gradients require TensorFlow eager execution")
    if isinstance(inputs, dict):
        raise TypeError("numerical gradient helper supports a tensor or list/tuple, not dict")

    is_sequence = isinstance(inputs, (list, tuple))
    input_values = list(inputs) if is_sequence else [inputs]
    tensors = [tf.convert_to_tensor(value) for value in input_values]
    if any(not tensor.dtype.is_floating for tensor in tensors):
        raise TypeError("all numerical-gradient inputs must be floating point")

    def function(*args):
        structured = type(inputs)(args) if is_sequence else args[0]
        output = _invoke(subject, structured, training=False)
        if len(tf.nest.flatten(output)) != 1:
            raise TypeError("numerical gradient helper requires one output tensor")
        return tf.nest.flatten(output)[0]

    theoretical, numerical = tf.test.compute_gradient(function, tensors, delta=delta)
    for index, (analytic, finite_difference) in enumerate(zip(theoretical, numerical)):
        assert_allclose(analytic, finite_difference, rtol=rtol, atol=atol,
                        err_msg="input %d analytic gradient differs from finite difference" % index)


def assert_invariant(subject, inputs, transform, output_selector=None,
                     rtol=1e-5, atol=1e-6):
    """Assert an output is unchanged by a semantics-preserving input transform."""
    selector = output_selector or (lambda value: value)
    baseline_inputs = copy.deepcopy(inputs)
    changed_inputs = transform(copy.deepcopy(inputs))
    baseline = selector(_invoke(subject, tf.nest.map_structure(_to_tensor, baseline_inputs),
                                training=False))
    changed = selector(_invoke(subject, tf.nest.map_structure(_to_tensor, changed_inputs),
                               training=False))
    assert_nested_allclose(changed, baseline, rtol=rtol, atol=atol)


def predict_numpy(model, inputs):
    """Predict without progress output and normalize nested results to NumPy."""
    try:
        predictions = model.predict(inputs, verbose=0)
    except TypeError:  # old Keras versions without the verbose keyword
        predictions = model.predict(inputs)
    return _numpy_tree(predictions)


def assert_serialization_predictions(model, inputs, custom_objects,
                                     rtol=1e-5, atol=1e-6):
    """Save/load a complete model and compare every prediction tensor."""
    before = predict_numpy(model, inputs)
    temp_dir = tempfile.mkdtemp(prefix="deepctr-model-")
    model_path = os.path.join(temp_dir, "model.h5")
    try:
        save_model(model, model_path)
        restored = load_model(model_path, custom_objects)
        after = predict_numpy(restored, inputs)
        assert_nested_allclose(after, before, rtol=rtol, atol=atol)
        return restored
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
