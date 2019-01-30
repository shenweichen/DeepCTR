from tensorflow.python.keras.layers import Layer,LSTM,Bidirectional,Lambda
from .layers import LocalActivationUnit
import tensorflow as tf
from tensorflow.python.keras import backend as K


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

    def __init__(self, mode='mean',supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        #self.seq_len_max = seq_len_max
        self.mode = mode
        self.eps = 1e-8
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = input_shape[0][1].value
        super(SequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!


    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.to_float(mask)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        uiseq_embed_list *= mask
        hist = uiseq_embed_list
        if self.mode == "max":
            return tf.reduce_max(hist, 1, keep_dims=True)

        hist = tf.reduce_sum(hist, 1, keep_dims=False)

        if self.mode == "mean":
            hist = tf.div(hist, user_behavior_length+self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None,1,input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self,):
        config = { 'mode': self.mode,'supports_masking':self.supports_masking}
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

        - **supports_masking**:If True,the input need to support masking.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_size=(80, 40), activation='sigmoid', weight_normalization=False,supports_masking=False,**kwargs):

        self.hidden_size = hidden_size
        self.activation = activation
        self.weight_normalization = weight_normalization

        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
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
        else:
            pass
        super(AttentionSequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None,**kwargs):

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

        attention_score = LocalActivationUnit(
            self.hidden_size, self.activation, 0, 1, False, 1024,)([queries, keys])

        outputs = tf.transpose(attention_score, (0, 2, 1))

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

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self,):

        config = {'hidden_size': self.hidden_size, 'activation': self.activation,
                  'weight_normalization': self.weight_normalization,'supports_masking':self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BiLSTM(Layer):
    """A multiple layer Bidirectional Residual LSTM Layer.

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
    """Bidirectional wrapper for RNNs.

    Arguments:
        layer: `Recurrent` instance.
        merge_mode: Mode by which outputs of the
            forward and backward RNNs will be combined.
            One of {'sum', 'mul', 'concat', 'ave', None}.
            If None, the outputs will not be combined,
            they will be returned as a list.

    Raises:
        ValueError: In case of invalid `merge_mode` argument.
    """
    def __init__(self,num_units, num_layers=2, num_res_layers=0,dropout_rate=0.2, mode='ave',**kwargs):

        if mode not in ['fw','bw','sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"fw","bw","sum", "mul", "ave", "concat", None}')

        self.num_units = num_units
        self.num_layers = num_layers
        self.num_res_layers = num_res_layers
        self.dropout_rate = dropout_rate
        self.mode = mode
        #self.hidden_size = hidden_size
        #self.activation = activation
        #self.weight_normalization = weight_normalization

        super(BiLSTM, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        # if not isinstance(input_shape, list) or len(input_shape) != 3:
        #     raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
        #                      'on a list of 3 inputs')
        #
        # if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 or len(input_shape[2]) != 2:
        #     raise ValueError("Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 2" % (
        #         len(input_shape[0]), len(input_shape[1]), len(input_shape[2])))
        #
        # if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1 or input_shape[2][1] != 1:
        #     raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
        #                      'inputs of a 3 inputs with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1)'
        #                      'Got different shapes: %s,%s and %s' % (input_shape))
        self.fw_lstm = []
        self.bw_lstm = []
        for i in range(self.num_layers):
            self.fw_lstm.append(LSTM(self.num_units, dropout=self.dropout_rate, bias_initializer='ones', return_sequences=True,
                             unroll=True))
            self.bw_lstm.append( LSTM(self.num_units, dropout=self.dropout_rate, bias_initializer='ones', return_sequences=True,
                             go_backwards=True, unroll=True))

        super(BiLSTM, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,mask=None,**kwargs):

        input_fw = inputs
        input_bw = inputs
        for i in range(self.num_layers):
            output_fw = self.fw_lstm[i](input_fw)
            output_bw = self.bw_lstm[i](input_bw)
            output_bw = Lambda(lambda x:K.reverse(x,1),mask=lambda inputs,mask: mask)(output_bw)

            if i >= self.num_layers - self.num_res_layers:
                output_fw += input_fw #= add([output_fw, input_fw])
                output_bw += input_bw  #add([output_bw, input_bw])
            input_fw = output_fw
            input_bw = output_bw

        output_fw = input_fw
        output_bw = input_bw
        #if self.rev:
        #    output_bw = K.reverse(output_bw,1)

        if self.mode == "fw":
            output = output_fw
        elif self.mode == "bw":
            output = output_bw
        elif self.mode == 'concat':
            output = K.concatenate([output_fw, output_bw])
        elif self.mode == 'sum':
            output = output_fw + output_bw
        elif self.mode == 'ave':
            output = (output_fw + output_bw) / 2
        elif self.mode == 'mul':
            output = output_fw * output_bw
        elif self.mode is None:
            output = [output_fw, output_bw]

        return output

    def compute_output_shape(self, input_shape):
        if self.mode is None:
            return [input_shape,input_shape]
        elif self.mode == 'concat':
            return input_shape[:-1]+(input_shape[-1]*2)
        else:
            return input_shape
    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self,):

        config = {'num_units': self.num_units, 'num_layers': self.num_layers,
                  'num_res_layers': self.num_res_layers,'dropout_rate':self.dropout_rate,'mode':self.mode}
        base_config = super(BiLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
