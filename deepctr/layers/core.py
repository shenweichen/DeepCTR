# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2

from .activation import activation_fun


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

      Arguments
        - **hidden_size**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **keep_prob**: float between 0 and 1. Fraction of the units to keep of attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_size=(64, 32), activation='sigmoid', l2_reg=0, keep_prob=1, use_bn=False, seed=1024, **kwargs):
        self.hidden_size = hidden_size
        self.activation = activation
        self.l2_reg = l2_reg
        self.keep_prob = keep_prob
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (
                len(input_shape[0]), len(input_shape[1])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:

            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape))
        size = 4 * \
            int(input_shape[0][-1]
                ) if len(self.hidden_size) == 0 else self.hidden_size[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                      initializer=glorot_normal(
                                          seed=self.seed),
                                      name="kernel")
        self.bias = self.add_weight(
            shape=(1,), initializer=Zeros(), name="bias")
        super(LocalActivationUnit, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = MLP(self.hidden_size, self.activation, self.l2_reg,
                      self.keep_prob, self.use_bn, seed=self.seed)(att_input)
        attention_score = tf.nn.bias_add(tf.tensordot(
            att_out, self.kernel, axes=(-1, 0)), self.bias)

        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self,):
        config = {'activation': self.activation, 'hidden_size': self.hidden_size,
                  'l2_reg': self.l2_reg, 'keep_prob': self.keep_prob, 'use_bn': self.use_bn, 'seed': self.seed}
        base_config = super(LocalActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MLP(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_size**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **keep_prob**: float between 0 and 1. Fraction of the units to keep.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self,  hidden_size, activation='relu', l2_reg=0, keep_prob=1, use_bn=False, seed=1024, **kwargs):
        self.hidden_size = hidden_size
        self.activation = activation
        self.keep_prob = keep_prob
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(MLP, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_size)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i+1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_size))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_size[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_size))]

        super(MLP, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_size)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            # fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = tf.keras.layers.BatchNormalization()(fc)
            fc = activation_fun(self.activation, fc)
            #fc = tf.nn.dropout(fc, self.keep_prob)
            fc = tf.keras.layers.Dropout(1 - self.keep_prob)(fc,)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_size) > 0:
            shape = input_shape[:-1] + (self.hidden_size[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self,):
        config = {'activation': self.activation, 'hidden_size': self.hidden_size,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'keep_prob': self.keep_prob, 'seed': self.seed}
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PredictionLayer(Layer):
    """
      Arguments
         - **activation**: Activation function to use.

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, activation='sigmoid', use_bias=True, **kwargs):
        self.activation = activation
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        output = activation_fun(self.activation, x)
        output = tf.reshape(output, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self,):
        config = {'activation': self.activation, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
