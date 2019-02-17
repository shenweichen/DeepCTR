# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

from tensorflow.python.keras.layers import Layer, Concatenate


class NoMask(Layer):
    def __init__(self,  **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)
