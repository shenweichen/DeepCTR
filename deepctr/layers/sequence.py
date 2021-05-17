# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.layers import LSTM, Lambda, Layer

from .core import LocalActivationUnit
from .normalization import LayerNormalization

if tf.__version__ >= '2.0.0':
    from ..contrib.rnn_v2 import dynamic_rnn
else:
    from ..contrib.rnn import dynamic_rnn
from ..contrib.utils import QAAttGRUCell, VecAttGRUCell
from .utils import reduce_sum, reduce_max, div, softmax, reduce_mean


class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return reduce_max(hist, 1, keep_dims=True)

        hist = reduce_sum(uiseq_embed_list * mask, 1, keep_dims=False)

        if self.mode == "mean":
            hist = div(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedSequenceLayer(Layer):
    """The WeightedSequenceLayer is used to apply weight score on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len,seq_weight]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

        - seq_weight is a 3D tensor with shape: ``(batch_size, T, 1)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, embedding_size)``.

      Arguments
        - **weight_normalization**: bool.Whether normalize the weight score before applying to sequence.

        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, weight_normalization=True, supports_masking=False, **kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(WeightedSequenceLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, input_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            key_input, value_input = input_list
            mask = tf.expand_dims(mask[0], axis=2)
        else:
            key_input, key_length_input, value_input = input_list
            mask = tf.sequence_mask(key_length_input,
                                    self.seq_len_max, dtype=tf.bool)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = key_input.shape[-1]

        if self.weight_normalization:
            paddings = tf.ones_like(value_input) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(value_input)
        value_input = tf.where(mask, value_input, paddings)

        if self.weight_normalization:
            value_input = softmax(value_input, dim=1)

        if len(value_input.shape) == 2:
            value_input = tf.expand_dims(value_input, axis=2)
            value_input = tf.tile(value_input, [1, 1, embedding_size])

        return tf.multiply(key_input, value_input)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask):
        if self.supports_masking:
            return mask[0]
        else:
            return None

    def get_config(self, ):
        config = {'weight_normalization': self.weight_normalization, 'supports_masking': self.supports_masking}
        base_config = super(WeightedSequenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionSequencePoolingLayer(Layer):
    """The Attentional sequence pooling operation used in DIN.

      Input shape
        - A list of three tensor: [query,keys,keys_length]

        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

        - keys_length is a 2D tensor with shape: ``(batch_size, 1)``

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **att_activation**: Activation function to use in attention net.

        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

        - **supports_masking**:If True,the input need to support masking.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            if not isinstance(input_shape, list) or len(input_shape) != 3:
                raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
                                 'on a list of 3 inputs')

            if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
                raise ValueError(
                    "Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
                        len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))

            if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
                raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
                                 'inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
                                 'Got different shapes: %s' % (input_shape))
        else:
            pass
        self.local_att = LocalActivationUnit(
            self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, )
        super(AttentionSequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            queries, keys = inputs
            key_masks = tf.expand_dims(mask[-1], axis=1)

        else:

            queries, keys, keys_length = inputs
            hist_len = keys.get_shape()[1]
            key_masks = tf.sequence_mask(keys_length, hist_len)

        attention_score = self.local_att([queries, keys], training=training)

        outputs = tf.transpose(attention_score, (0, 2, 1))

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        if tf.__version__ < '1.13.0':
            outputs._uses_learning_phase = attention_score._uses_learning_phase
        else:
            outputs._uses_learning_phase = training is not None

        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[1][1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):

        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BiLSTM(Layer):
    """A multiple layer Bidirectional Residual LSTM Layer.

      Input shape
        - 3D tensor with shape ``(batch_size, timesteps, input_dim)``.

      Output shape
        - 3D tensor with shape: ``(batch_size, timesteps, units)``.

      Arguments
        - **units**: Positive integer, dimensionality of the output space.

        - **layers**:Positive integer, number of LSTM layers to stacked.

        - **res_layers**: Positive integer, number of residual connection to used in last ``res_layers``.

        - **dropout_rate**:  Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.

        - **merge_mode**: merge_mode: Mode by which outputs of the forward and backward RNNs will be combined. One of { ``'fw'`` , ``'bw'`` , ``'sum'`` , ``'mul'`` , ``'concat'`` , ``'ave'`` , ``None`` }. If None, the outputs will not be combined, they will be returned as a list.


    """

    def __init__(self, units, layers=2, res_layers=0, dropout_rate=0.2, merge_mode='ave', **kwargs):

        if merge_mode not in ['fw', 'bw', 'sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"fw","bw","sum", "mul", "ave", "concat", None}')

        self.units = units
        self.layers = layers
        self.res_layers = res_layers
        self.dropout_rate = dropout_rate
        self.merge_mode = merge_mode

        super(BiLSTM, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        self.fw_lstm = []
        self.bw_lstm = []
        for _ in range(self.layers):
            self.fw_lstm.append(
                LSTM(self.units, dropout=self.dropout_rate, bias_initializer='ones', return_sequences=True,
                     unroll=True))
            self.bw_lstm.append(
                LSTM(self.units, dropout=self.dropout_rate, bias_initializer='ones', return_sequences=True,
                     go_backwards=True, unroll=True))

        super(BiLSTM, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, **kwargs):

        input_fw = inputs
        input_bw = inputs
        for i in range(self.layers):
            output_fw = self.fw_lstm[i](input_fw)
            output_bw = self.bw_lstm[i](input_bw)
            output_bw = Lambda(lambda x: K.reverse(
                x, 1), mask=lambda inputs, mask: mask)(output_bw)

            if i >= self.layers - self.res_layers:
                output_fw += input_fw
                output_bw += input_bw
            input_fw = output_fw
            input_bw = output_bw

        output_fw = input_fw
        output_bw = input_bw

        if self.merge_mode == "fw":
            output = output_fw
        elif self.merge_mode == "bw":
            output = output_bw
        elif self.merge_mode == 'concat':
            output = K.concatenate([output_fw, output_bw])
        elif self.merge_mode == 'sum':
            output = output_fw + output_bw
        elif self.merge_mode == 'ave':
            output = (output_fw + output_bw) / 2
        elif self.merge_mode == 'mul':
            output = output_fw * output_bw
        elif self.merge_mode is None:
            output = [output_fw, output_bw]

        return output

    def compute_output_shape(self, input_shape):
        print(self.merge_mode)
        if self.merge_mode is None:
            return [input_shape, input_shape]
        elif self.merge_mode == 'concat':
            return input_shape[:-1] + (input_shape[-1] * 2,)
        else:
            return input_shape

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self, ):

        config = {'units': self.units, 'layers': self.layers,
                  'res_layers': self.res_layers, 'dropout_rate': self.dropout_rate, 'merge_mode': self.merge_mode}
        base_config = super(BiLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Transformer(Layer):
    """  Simplified version of Transformer  proposed in 《Attention is all you need》

      Input shape
        - a list of two 3D tensor with shape ``(batch_size, timesteps, input_dim)`` if ``supports_masking=True`` .
        - a list of two 4 tensors, first two tensors with shape ``(batch_size, timesteps, input_dim)``,last two tensors with shape ``(batch_size, 1)`` if ``supports_masking=False`` .


      Output shape
        - 3D tensor with shape: ``(batch_size, 1, input_dim)``  if ``output_type='mean'`` or ``output_type='sum'`` , else  ``(batch_size, timesteps, input_dim)`` .


      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **dropout_rate**: float between 0 and 1. Fraction of the units to drop.
            - **use_positional_encoding**: bool. Whether or not use positional_encoding
            - **use_res**: bool. Whether or not use standard residual connections before output.
            - **use_feed_forward**: bool. Whether or not use pointwise feed foward network.
            - **use_layer_norm**: bool. Whether or not use Layer Normalization.
            - **blinding**: bool. Whether or not use blinding.
            - **seed**: A Python integer to use as random seed.
            - **supports_masking**:bool. Whether or not support masking.
            - **attention_type**: str, Type of attention, the value must be one of { ``'scaled_dot_product'`` , ``'additive'`` }.
            - **output_type**: ``'mean'`` , ``'sum'`` or `None`. Whether or not use average/sum pooling for output.

      References
            - [Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
    """

    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=False, blinding=True, seed=1024, supports_masking=False,
                 attention_type="scaled_dot_product", output_type="mean", **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        super(Transformer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                    self.att_embedding_size, self.head_num, embedding_size))
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
        if self.attention_type == "additive":
            self.b = self.add_weight('b', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.v = self.add_weight('v', shape=[self.att_embedding_size], dtype=tf.float32,
                                     initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        # if self.use_res:
        #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
        #                                  initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))

        # if self.use_positional_encoding:
        #
        #     self.kpe = Position_Embedding(input_shape[0][-1].value)
        #     self.qpe = Position_Embedding(input_shape[1][-1].value)
        self.dropout = tf.keras.layers.Dropout(
            self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()
        # Be sure to call this somewhere!
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):

        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs

            query_masks = tf.sequence_mask(
                query_masks, self.seq_len_max, dtype=tf.float32)
            key_masks = tf.sequence_mask(
                key_masks, self.seq_len_max, dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)

        if self.use_positional_encoding:
            queries = positional_encoding(queries)
            keys = positional_encoding(queries)

        querys = tf.tensordot(queries, self.W_Query,
                              axes=(-1, 0))  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))

        # head_num*None T_q D
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":
            # head_num*None T_q T_k
            outputs = tf.matmul(querys, keys, transpose_b=True)

            outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        elif self.attention_type == "additive":
            querys_reshaped = tf.expand_dims(querys, axis=-2)
            keys_reshaped = tf.expand_dims(keys, axis=-3)
            outputs = tf.tanh(tf.nn.bias_add(querys_reshaped + keys_reshaped, self.b))
            outputs = tf.squeeze(tf.tensordot(outputs, tf.expand_dims(self.v, axis=-1), axes=[-1, 0]), axis=-1)
        else:
            NotImplementedError

        key_masks = tf.tile(key_masks, [self.head_num, 1])

        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

        # (h*N, T_q, T_k)

        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )
        if self.blinding:
            try:
                outputs = tf.matrix_set_diag(outputs, tf.ones_like(outputs)[
                                                      :, :, 0] * (-2 ** 32 + 1))
            except:
                outputs = tf.compat.v1.matrix_set_diag(outputs, tf.ones_like(outputs)[
                                                                :, :, 0] * (-2 ** 32 + 1))

        outputs -= reduce_max(outputs, axis=-1, keep_dims=True)
        outputs = softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # (h*N, T_q, T_k)
        query_masks = tf.tile(tf.expand_dims(
            query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)
        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
            result += queries
        if self.use_layer_norm:
            result = self.ln(result)

        if self.use_feed_forward:
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

        if self.output_type == "mean":
            return reduce_mean(result, axis=1, keep_dims=True)
        elif self.output_type == "sum":
            return reduce_sum(result, axis=1, keep_dims=True)
        else:
            return result

    def compute_output_shape(self, input_shape):

        return (None, 1, self.att_embedding_size * self.head_num)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num,
                  'dropout_rate': self.dropout_rate, 'use_res': self.use_res,
                  'use_positional_encoding': self.use_positional_encoding, 'use_feed_forward': self.use_feed_forward,
                  'use_layer_norm': self.use_layer_norm, 'seed': self.seed, 'supports_masking': self.supports_masking,
                  'blinding': self.blinding, 'attention_type': self.attention_type, 'output_type': self.output_type}
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def positional_encoding(inputs,
                        pos_embedding_trainable=True,
                        zero_pad=False,
                        scale=True,
                        ):
    '''Sinusoidal Positional_Encoding.

    Args:

      - inputs: A 2d Tensor with shape of (N, T).
      - num_units: Output dimensionality
      - zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      - scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      - scope: Optional scope for `variable_scope`.
      - reuse: Boolean, whether to reuse the weights of a previous layer by the same name.

    Returns:

      - A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    _, T, num_units = inputs.get_shape().as_list()
    # with tf.variable_scope(scope, reuse=reuse):
    position_ind = tf.expand_dims(tf.range(T), 0)
    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, 2. * i / num_units)
         for i in range(num_units)]
        for pos in range(T)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    # Convert to a tensor

    if pos_embedding_trainable:
        lookup_table = K.variable(position_enc, dtype=tf.float32)

    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                  lookup_table[1:, :]), 0)

    outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

    if scale:
        outputs = outputs * num_units ** 0.5
    return outputs + inputs


class BiasEncoding(Layer):
    def __init__(self, sess_max_count, seed=1024, **kwargs):
        self.sess_max_count = sess_max_count
        self.seed = seed
        super(BiasEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        if self.sess_max_count == 1:
            embed_size = input_shape[2].value
            seq_len_max = input_shape[1].value
        else:
            embed_size = input_shape[0][2].value
            seq_len_max = input_shape[0][1].value

        self.sess_bias_embedding = self.add_weight('sess_bias_embedding', shape=(self.sess_max_count, 1, 1),
                                                   initializer=TruncatedNormal(
                                                       mean=0.0, stddev=0.0001, seed=self.seed))
        self.seq_bias_embedding = self.add_weight('seq_bias_embedding', shape=(1, seq_len_max, 1),
                                                  initializer=TruncatedNormal(
                                                      mean=0.0, stddev=0.0001, seed=self.seed))
        self.item_bias_embedding = self.add_weight('item_bias_embedding', shape=(1, 1, embed_size),
                                                   initializer=TruncatedNormal(
                                                       mean=0.0, stddev=0.0001, seed=self.seed))

        # Be sure to call this somewhere!
        super(BiasEncoding, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        transformer_out = []
        for i in range(self.sess_max_count):
            transformer_out.append(
                inputs[i] + self.item_bias_embedding + self.seq_bias_embedding + self.sess_bias_embedding[i])
        return transformer_out

    def compute_output_shape(self, input_shape):

        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self, ):

        config = {'sess_max_count': self.sess_max_count, 'seed': self.seed, }
        base_config = super(BiasEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicGRU(Layer):
    def __init__(self, num_units=None, gru_type='GRU', return_sequence=True, **kwargs):

        self.num_units = num_units
        self.return_sequence = return_sequence
        self.gru_type = gru_type
        super(DynamicGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]
        if self.gru_type == "AGRU":
            self.gru_cell = QAAttGRUCell(self.num_units)
        elif self.gru_type == "AUGRU":
            self.gru_cell = VecAttGRUCell(self.num_units)
        else:
            try:
                self.gru_cell = tf.nn.rnn_cell.GRUCell(self.num_units)
            except:
                self.gru_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.num_units)

        # Be sure to call this somewhere!
        super(DynamicGRU, self).build(input_shape)

    def call(self, input_list):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        if self.gru_type == "GRU" or self.gru_type == "AIGRU":
            rnn_input, sequence_length = input_list
            att_score = None
        else:
            rnn_input, sequence_length, att_score = input_list

        rnn_output, hidden_state = dynamic_rnn(self.gru_cell, inputs=rnn_input, att_scores=att_score,
                                               sequence_length=tf.squeeze(sequence_length,
                                                                          ), dtype=tf.float32, scope=self.name)
        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(hidden_state, axis=1)

    def compute_output_shape(self, input_shape):
        rnn_input_shape = input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return (None, 1, rnn_input_shape[2])

    def get_config(self, ):
        config = {'num_units': self.num_units, 'gru_type': self.gru_type, 'return_sequence': self.return_sequence}
        base_config = super(DynamicGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KMaxPooling(Layer):
    """K Max pooling that selects the k biggest value along the specific axis.

      Input shape
        -  nD tensor with shape: ``(batch_size, ..., input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., output_dim)``.

      Arguments
        - **k**: positive integer, number of top elements to look for along the ``axis`` dimension.

        - **axis**: positive integer, the dimension to look for elements.

     """

    def __init__(self, k=1, axis=-1, **kwargs):

        self.k = k
        self.axis = axis
        super(KMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.axis < 1 or self.axis > len(input_shape):
            raise ValueError("axis must be 1~%d,now is %d" %
                             (len(input_shape), self.axis))

        if self.k < 1 or self.k > input_shape[self.axis]:
            raise ValueError("k must be in 1 ~ %d,now k is %d" %
                             (input_shape[self.axis], self.k))
        self.dims = len(input_shape)
        # Be sure to call this somewhere!
        super(KMaxPooling, self).build(input_shape)

    def call(self, inputs):

        # swap the last and the axis dimensions since top_k will be applied along the last dimension
        perm = list(range(self.dims))
        perm[-1], perm[self.axis] = perm[self.axis], perm[-1]
        shifted_input = tf.transpose(inputs, perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        output = tf.transpose(top_k, perm)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.k
        return tuple(output_shape)

    def get_config(self, ):
        config = {'k': self.k, 'axis': self.axis}
        base_config = super(KMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
