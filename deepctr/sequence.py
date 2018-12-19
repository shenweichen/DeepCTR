from tensorflow.python.keras.layers import Layer
from .layers import LocalActivationUnit
import tensorflow as tf


class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **seq_len_max**:Positive integer indicates that the max length of all the sequence feature,usually same as T.

        - **mode**:str.Pooling operation to be used,can be sum,mean or max.
    """

    def __init__(self, seq_len_max, mode='mean', **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.seq_len_max = seq_len_max
        self.mode = mode
        super(SequencePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, **kwargs):
        uiseq_embed_list, user_behavior_length = seq_value_len_list
        embedding_size = uiseq_embed_list.shape[-1]
        mask = tf.sequence_mask(user_behavior_length,
                                self.seq_len_max, dtype=tf.float32)

        mask = tf.transpose(mask, (0, 2, 1))

        mask = tf.tile(mask, [1, 1, embedding_size])
        uiseq_embed_list *= mask
        hist = uiseq_embed_list
        if self.mode == "max":
            return tf.reduce_max(hist, 1, keep_dims=True)

        hist = tf.reduce_sum(hist, 1, keep_dims=False)
        if self.mode == "mean":

            hist = tf.div(hist, user_behavior_length)
        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[0][-1])

    def get_config(self,):
        config = {'seq_len_max': self.seq_len_max, 'mode': self.mode}
        base_config = super(SequencePoolingLayer, self).get_config()
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
        - **hidden_size**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_size=(80, 40), activation='sigmoid', weight_normalization=False, **kwargs):

        self.hidden_size = hidden_size
        self.activation = activation
        self.weight_normalization = weight_normalization

        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
                             'on a list of 3 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
            raise ValueError("Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
                len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
            raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
                             'inputs of a 3 inputs with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
                             'Got different shapes: %s,%s and %s' % (input_shape))
        super(AttentionSequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        query_key_keylen_list = inputs
        queries, keys, keys_length = query_key_keylen_list
        hist_len = keys.get_shape()[1]

        attention_score = LocalActivationUnit(
            self.hidden_size, self.activation, 0, 1, False, 1024,)([queries, keys])

        outputs = tf.transpose(attention_score, (0, 2, 1))

        key_masks = tf.sequence_mask(keys_length, hist_len)

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)

        outputs = tf.matmul(outputs, keys)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[0][-1])

    def get_config(self,):

        config = {'hidden_size': self.hidden_size, 'activation': self.activation,
                  'weight_normalization': self.weight_normalization}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
