# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""
import tensorflow as tf


class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            x = tf.as_string(x, )
        hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, **kwargs):

        self.l2_reg = l2_reg
        # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
        self.mode = mode
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bias = self.add_weight(name='linear_bias',
                                    shape=(1,),
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)

        self.dense = tf.keras.layers.Dense(units=1, activation=None, use_bias=False,
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs , **kwargs):

        if self.mode == 0:
            sparse_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            linear_logit = self.dense(dense_input)

        else:
            sparse_input, dense_input = inputs

            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keepdims=False) + self.dense(dense_input)

        linear_bias_logit = linear_logit + self.bias

        return linear_bias_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)
