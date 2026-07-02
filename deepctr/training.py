# -*- coding:utf-8 -*-
"""Opt-in cross-framework compile and fit helpers."""

import math

import numpy as np
import tensorflow as tf


class TensorFlowAdam(object):
    """Adam with duplicate sparse indices aggregated before each update."""

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0
        self.state = {}

    def apply_gradients(self, variables, gradients):
        self.iterations += 1
        step = self.iterations
        step_size = (
            self.learning_rate
            * math.sqrt(1.0 - self.beta_2 ** step)
            / (1.0 - self.beta_1 ** step)
        )
        for index, (variable, gradient) in enumerate(zip(variables, gradients)):
            if gradient is None:
                continue
            if index not in self.state:
                self.state[index] = (
                    tf.Variable(tf.zeros_like(variable), trainable=False),
                    tf.Variable(tf.zeros_like(variable), trainable=False),
                )
            first, second = self.state[index]
            if isinstance(gradient, tf.IndexedSlices):
                unique, position = tf.unique(tf.cast(gradient.indices, tf.int32))
                values = tf.math.unsorted_segment_sum(
                    gradient.values, position, tf.shape(unique)[0]
                )
                first_rows = tf.gather(first, unique)
                second_rows = tf.gather(second, unique)
                updated_first = (
                    self.beta_1 * first_rows + (1.0 - self.beta_1) * values
                )
                updated_second = (
                    self.beta_2 * second_rows
                    + (1.0 - self.beta_2) * tf.square(values)
                )
                first.scatter_update(tf.IndexedSlices(updated_first, unique))
                second.scatter_update(tf.IndexedSlices(updated_second, unique))
                variable.scatter_sub(tf.IndexedSlices(
                    step_size * updated_first
                    / (tf.sqrt(updated_second) + self.epsilon),
                    unique,
                ))
            else:
                gradient = tf.convert_to_tensor(gradient)
                first.assign(
                    self.beta_1 * first + (1.0 - self.beta_1) * gradient
                )
                second.assign(
                    self.beta_2 * second
                    + (1.0 - self.beta_2) * tf.square(gradient)
                )
                variable.assign_sub(
                    step_size * first / (tf.sqrt(second) + self.epsilon)
                )


class NumpyBatchSequence(tf.keras.utils.Sequence):
    """Keras sequence driven by NumPy PCG64 epoch permutations."""

    def __init__(self, x, y, batch_size=256, seed=1024, shuffle=True):
        self.x = x
        self.y = np.asarray(y)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.random = np.random.default_rng(seed)
        if self.batch_size < 1 or len(self.y) < 1:
            raise ValueError("batch_size and training data must be non-empty")
        values = list(x.values()) if isinstance(x, dict) else x
        values = values if isinstance(values, (list, tuple)) else (values,)
        if any(len(value) != len(self.y) for value in values):
            raise ValueError("all feature arrays and labels must have equal lengths")
        self.order = np.arange(len(self.y))
        self._advance()

    def _advance(self):
        self.order = (
            self.random.permutation(len(self.y))
            if self.shuffle else np.arange(len(self.y))
        )

    def __len__(self):
        return int(math.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, index):
        indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        if isinstance(self.x, dict):
            features = {name: np.asarray(value)[indices] for name, value in self.x.items()}
        elif isinstance(self.x, (list, tuple)):
            features = [np.asarray(value)[indices] for value in self.x]
        else:
            features = np.asarray(self.x)[indices]
        return features, self.y[indices]

    def on_epoch_end(self):
        self._advance()


def compile_cross_framework(model, loss="binary_crossentropy", metrics=None,
                            learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                            epsilon=1e-7):
    """Compile with the Keras-2 Adam semantics mirrored by DeepCTR-Torch."""
    keras_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )
    model.compile(optimizer=keras_optimizer, loss=loss, metrics=metrics)
    model.cross_framework_optimizer = TensorFlowAdam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )
    return model


def fit_cross_framework(model, x, y, batch_size=256, epochs=1, seed=1024,
                        shuffle=True, verbose=1):
    """Fit with an explicit deterministic mini-batch timeline.

    ``train_on_batch`` avoids framework DataAdapter prefetch and epoch hooks,
    so TensorFlow and PyTorch consume exactly the same NumPy permutations.
    """
    tf.keras.utils.set_random_seed(seed)
    enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
    if enable_determinism is not None:
        enable_determinism()
    sequence = NumpyBatchSequence(
        x, y, batch_size=batch_size, seed=seed, shuffle=shuffle
    )
    history = tf.keras.callbacks.History()
    history.set_model(model)
    history.history = {}
    history.epoch = []
    for epoch in range(epochs):
        totals = {}
        rows = 0
        for index in range(len(sequence)):
            batch_x, batch_y = sequence[index]
            target = tf.convert_to_tensor(batch_y, dtype=tf.float32)
            with tf.GradientTape() as tape:
                prediction = model(batch_x, training=True)
                loss = tf.reduce_mean(
                    tf.keras.backend.binary_crossentropy(target, prediction)
                )
                if model.losses:
                    loss = loss + tf.add_n(model.losses)
            variables = list(model.trainable_weights)
            gradients = tape.gradient(loss, variables)
            model.cross_framework_optimizer.apply_gradients(variables, gradients)
            result = {"loss": float(loss.numpy())}
            batch_rows = len(batch_y)
            rows += batch_rows
            for name, value in result.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_rows
        logs = {name: value / rows for name, value in totals.items()}
        history.epoch.append(epoch)
        for name, value in logs.items():
            history.history.setdefault(name, []).append(value)
        if verbose:
            metrics = " - ".join(
                "{}: {:.4f}".format(name, value) for name, value in logs.items()
            )
            print("Epoch {}/{} - {}".format(epoch + 1, epochs, metrics))
        if epoch + 1 < epochs:
            sequence.on_epoch_end()
    return history
