import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import LSTM, Lambda, Layer

from .layers import LocalActivationUnit,LayerNormalization


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
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self,):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
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

    def __init__(self, hidden_size=(80, 40), activation='sigmoid', weight_normalization=False, supports_masking=False, **kwargs):

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

    def call(self, inputs, mask=None, **kwargs):

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
                  'weight_normalization': self.weight_normalization, 'supports_masking': self.supports_masking}
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

        - **dropout**:  Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.

        - **merge_mode**: merge_mode: Mode by which outputs of the forward and backward RNNs will be combined. One of {``'fw'``,``'bw'``,``'sum'``, ``'mul'``, ``'concat'``, ``'ave'``, ``None``}.. If None, the outputs will not be combined, they will be returned as a list.


    """

    def __init__(self, units, layers=2, res_layers=0, dropout=0.2, merge_mode='ave', **kwargs):

        if merge_mode not in ['fw', 'bw', 'sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"fw","bw","sum", "mul", "ave", "concat", None}')

        self.units = units
        self.layers = layers
        self.res_layers = res_layers
        self.dropout = dropout
        self.merge_mode = merge_mode

        super(BiLSTM, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        self.fw_lstm = []
        self.bw_lstm = []
        for i in range(self.layers):
            self.fw_lstm.append(LSTM(self.units, dropout=self.dropout, bias_initializer='ones', return_sequences=True,
                                     unroll=True))
            self.bw_lstm.append(LSTM(self.units, dropout=self.dropout, bias_initializer='ones', return_sequences=True,
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
            return input_shape[:-1]+(input_shape[-1]*2,)
        else:
            return input_shape

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self,):

        config = {'units': self.units, 'layers': self.layers,
                  'res_layers': self.res_layers, 'dropout': self.dropout, 'merge_mode': self.merge_mode}
        base_config = super(BiLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class AttentionSequencePoolingLayerv2(Layer):
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

    def __init__(self, hidden_size=(80, 40), activation='sigmoid', weight_normalization=True,sim_type='nn', **kwargs):

        self.hidden_size = hidden_size
        self.activation = activation
        self.weight_normalization = weight_normalization
        self.sim_type = sim_type

        super(AttentionSequencePoolingLayerv2, self).__init__(**kwargs)

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
        #if self.sim_type == "mul":
        #    self.add_weight(name='kernel',
        #                                shape=(dim, 1),
        #                               initializer=glorot_normal(seed=self.seed),
        #                                regularizer=l2(self.l2_reg),
        #                                trainable=True)

        super(AttentionSequencePoolingLayerv2, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        query_key_keylen_list = inputs
        queries, keys, keys_length = query_key_keylen_list
        hist_len = keys.get_shape()[1]
        #print(queries,keys)
        if self.sim_type == "nn":
            #print(queries,keys,'---------------')
            attention_score =LocalActivationUnit(
                self.hidden_size, self.activation, 0, 1, False, 1024,)([queries, keys])
                #NaiveActivationUnit()([queries,keys])
                #LocalActivationUnit(
                #self.hidden_size, self.activation, 0, 1, False, 1024,)([queries, keys])
            #NaiveActivationUnit()([queries,keys])
        elif self.sim_type == 'mat':
            attention_score = NaiveActivationUnit()([queries,keys])
        elif self.sim_type == "cos":
            #query, keys = inputs
            def cosine_distance(vests):
                x, y = vests
                x = K.l2_normalize(x, axis=-1)
                y = K.l2_normalize(y, axis=-1)
                return K.mean(x * y, axis=-1, keepdims=True)

            def cos_dist_output_shape(shapes):
                shape1, shape2 = shapes
                return shape1[:2]+(1,)

            keys_len = keys.get_shape()[1]
            queries = K.repeat_elements(queries, keys_len, 1)
            attention_score = Lambda(cosine_distance)([queries, keys])


            #print(attention_score)
        #elif self.sim_type == "mul":

        attention_score = tf.transpose(attention_score,(0,2,1))

        key_masks = tf.sequence_mask(keys_length, hist_len)
        #print(keys_length,hist_len,key_masks)
        if self.weight_normalization:
            paddings = tf.ones_like(attention_score) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(attention_score)
        #print(attention_score,'-----------------------')
        attention_score = tf.where(key_masks, attention_score, paddings)

        attention_score = attention_score / (keys.get_shape().as_list()[-1] ** 0.5)

        if self.weight_normalization:
            attention_score = tf.nn.softmax(attention_score)
        return attention_score
        #outputs = tf.matmul(attention_score, keys)

        #return outputs

    def compute_output_shape(self, input_shape):
        #return (None, 1, input_shape[0][-1])

        return (None, 1, input_shape[1][1])

    def get_config(self,):

        config = {'hidden_size': self.hidden_size, 'activation': self.activation,
                  'weight_normalization': self.weight_normalization}
        base_config = super(AttentionSequencePoolingLayerv2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#
# class Transformer(Layer):
#     """Transformer  proposed in 《Attention is all you need》
#
#       Input shape
#         - 3D tensor with shape ``(batch_size, timesteps, input_dim)``.
#
#       Output shape
#         - 3D tensor with shape: ``(batch_size, 1, input_dim)``.
#
#
#       Arguments
#             - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
#             - **head_num**: int.The head number in multi-head  self-attention network.
#             - **dropout_rate**: float between 0 and 1. Fraction of the units to drop.
#             - **use_positional_encoding**: bool.Whether or not use positional_encoding
#             - **use_res**: bool.Whether or not use standard residual connections before output.
#             - **use_feed_forward**: bool.Whether or not use pointwise feed foward network.
#             - **use_layer_norm**: bool. Whether or not use Layer  Normalization.
#             - **seed**: A Python integer to use as random seed.
#             - **supports_masking**:bool. Whether or not support masking.
#
#       References
#             - [Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
#     """
#
#     def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0,use_positional_encoding=True,use_res=True,use_feed_forward=True,use_layer_norm=False,seed=1024,supports_masking=False, **kwargs):
#         if head_num <= 0:
#             raise ValueError('head_num must be a int > 0')
#         self.att_embedding_size = att_embedding_size
#         self.head_num = head_num
#         self.num_units = att_embedding_size * head_num
#         self.use_res = use_res
#         self.use_feed_forward = use_feed_forward
#         self.seed = seed
#         self.use_positional_encoding = use_positional_encoding
#         self.dropout_rate = dropout_rate
#         self.use_layer_norm = use_layer_norm
#         super(Transformer, self).__init__(**kwargs)
#         self.supports_masking = supports_masking
#
#     def build(self, input_shape):
#         # if len(input_shape) != 3:
#         #     raise ValueError(
#         #         "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
#
#         embedding_size = input_shape[0][-1].value
#         self.seq_len_max = input_shape[0][-2].value
#         self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
#                                        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
#         self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
#                                      initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed+1))
#         self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
#                                        initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed+2))
#         # if self.use_res:
#         #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
#         #                                  initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
#         if self.use_feed_forward:
#             self.fw1 = self.add_weight('fw1',shape=[self.num_units,4*self.num_units],dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
#             self.fw2 = self.add_weight('fw2',shape=[4*self.num_units,self.num_units],dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
#
#         self.dropout = tf.keras.layers.Dropout(self.dropout_rate,seed=self.seed)
#         self.ln = LayerNormalization()
#         # Be sure to call this somewhere!
#         super(Transformer, self).build(input_shape)
#
#     def call(self, inputs,mask=None, **kwargs):
#         # if K.ndim(inputs) != 3:
#         #     raise ValueError(
#         #         "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
#         if self.supports_masking:
#             queries, keys = inputs
#             query_masks,key_masks = mask
#         else:
#             queries, keys, query_masks, key_masks = inputs
#
#             query_masks = tf.sequence_mask(query_masks,self.seq_len_max, dtype=tf.float32)
#             key_masks = tf.sequence_mask(key_masks,self.seq_len_max, dtype=tf.float32)
#             query_masks = tf.squeeze(query_masks, axis=1)
#             key_masks = tf.squeeze(key_masks, axis=1)
#
#         if self.use_positional_encoding:
#             pe_units = keys.get_shape().as_list()[-1]
#             queries_positional_encoding = self.positional_encoding(query_masks, pe_units,pos_embedding_trainable=True)
#             queries += queries_positional_encoding
#             keys_positional_encoding = self.positional_encoding(key_masks, pe_units, pos_embedding_trainable=True)
#             keys += keys_positional_encoding
#
#         querys = tf.tensordot(queries, self.W_Query,axes=(-1, 0))  # None T_q D*head_num
#         keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
#         values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))
#
#         querys = tf.concat(tf.split(querys, self.head_num, axis=2),axis=0) #head_num*None T_q D
#         keys = tf.concat(tf.split(keys, self.head_num, axis=2),axis=0)
#         values = tf.concat(tf.split(values, self.head_num, axis=2),axis=0)
#
#         outputs = tf.matmul(querys, keys, transpose_b=True)  # head_num*None T_q T_k
#
#         outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
#
#         key_masks = tf.tile(key_masks, [self.head_num, 1])
#
#         # (h*N, T_q, T_k)
#         key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
#
#         paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
#
#         # (h*N, T_q, T_k)
#
#         outputs = tf.where(tf.equal(key_masks, 1), outputs,paddings, )
#
#         outputs -= tf.reduce_max(outputs, axis=-1, keep_dims=True)
#         outputs = tf.nn.softmax(outputs)
#         query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
#         # (h*N, T_q, T_k)
#         query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
#
#         outputs *= query_masks
#
#         outputs = self.dropout(outputs)
#         # Weighted sum
#         # ( h*N, T_q, C/h)
#         result = tf.matmul(outputs,values)
#         result = tf.concat(tf.split(result, self.head_num,axis=0 ), axis=2)
#
#         if self.use_res:
#             result += queries #tf.tensordot(queries, self.W_Res, axes=(-1, 0))
#         if self.use_layer_norm:
#             result = self.ln(result)
#
#         if self.use_feed_forward:
#             fw1 = tf.nn.relu(tf.tensordot(result,self.fw1,axes=[-1,0]))
#             fw1 = self.dropout(fw1)
#             fw2 = tf.tensordot(fw1,self.fw2,axes=[-1,0])
#             if self.use_res:
#                 result += fw2
#             if self.use_layer_norm:
#                 result = self.ln(result)
#
#         return tf.reduce_mean(result,axis=1,keep_dims=True)
#
#     def compute_output_shape(self, input_shape):
#
#         return (None, 1, self.att_embedding_size * self.head_num)
#
#     def get_config(self, ):
#         config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num, 'use_res': self.use_res,
#                   'seed': self.seed}
#         base_config = super(Transformer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def positional_encoding(self, inputs,
#                             num_units,
#                             pos_embedding_trainable=False,
#                             zero_pad=False,
#                             scale=True,
#                             ):
#         '''Sinusoidal Positional_Encoding.
#         Args:
#           inputs: A 2d Tensor with shape of (N, T).
#           num_units: Output dimensionality
#           zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
#           scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
#           scope: Optional scope for `variable_scope`.
#           reuse: Boolean, whether to reuse the weights of a previous layer
#             by the same name.
#         Returns:
#             A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
#         '''
#
#         N, T = inputs.get_shape().as_list()
#         # with tf.variable_scope(scope, reuse=reuse):
#         position_ind = tf.expand_dims(tf.range(T), 0)
#         # First part of the PE function: sin and cos argument
#         position_enc = np.array([
#             [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
#             for pos in range(T)])
#
#         # Second part, apply the cosine to even columns and sin to odds.
#         position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
#         position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
#
#         # Convert to a tensor
#         lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
#         if pos_embedding_trainable:
#             # lookup_table = tf.variable('trainable_pos_embedding', initializer=lookup_table, dtype=tf.float32)
#             lookup_table = tf.Variable(initial_value=lookup_table, dtype=tf.float32)
#
#         if zero_pad:
#             lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
#                                       lookup_table[1:, :]), 0)
#         outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
#
#         if scale:
#             outputs = outputs * num_units ** 0.5
#         return outputs


class Transformer(Layer):
    """Transformer  proposed in 《Attention is all you need》

      Input shape
        - 3D tensor with shape ``(batch_size, timesteps, input_dim)``.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, input_dim)``.


      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **dropout_rate**: float between 0 and 1. Fraction of the units to drop.
            - **use_positional_encoding**: bool.Whether or not use positional_encoding
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **use_feed_forward**: bool.Whether or not use pointwise feed foward network.
            - **use_layer_norm**: bool. Whether or not use Layer  Normalization.
            - **seed**: A Python integer to use as random seed.
            - **supports_masking**:bool. Whether or not support masking.

      References
            - [Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
    """

    def __init__(self, att_embedding_size=1, head_num=8, dropout_rate=0.0, use_positional_encoding=True, use_res=True,
                 use_feed_forward=True, use_layer_norm=False, seed=1024, supports_masking=False, **kwargs):
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
        super(Transformer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        # if len(input_shape) != 3:
        #     raise ValueError(
        #         "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        embedding_size = input_shape[0][-1].value
        self.seq_len_max = input_shape[0][-2].value
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed + 2))
        # if self.use_res:
        #     self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], dtype=tf.float32,
        #                                  initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        if self.use_feed_forward:
            self.fw1 = self.add_weight('fw1', shape=[self.num_units, 4 * self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
            self.fw2 = self.add_weight('fw2', shape=[4 * self.num_units, self.num_units], dtype=tf.float32,
                                       initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))

        if self.use_positional_encoding:
            #self.keys_idx, self.keys_pe = self.add_pe_weight(input_shape[0])
            #self.querys_idx, self.querys_pe = self.add_pe_weight(input_shape[1])
            self.kpe = Position_Embedding(input_shape[0][-1].value)
            self.qpe = Position_Embedding(input_shape[1][-1].value)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()
        # Be sure to call this somewhere!
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # if K.ndim(inputs) != 3:
        #     raise ValueError(
        #         "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
        else:
            queries, keys, query_masks, key_masks = inputs

            query_masks = tf.sequence_mask(query_masks, self.seq_len_max, dtype=tf.float32)
            key_masks = tf.sequence_mask(key_masks, self.seq_len_max, dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)
        # if scale:
        #     outputs = outputs * num_units ** 0.5
        if self.use_positional_encoding:
            pe_units = keys.get_shape().as_list()[-1]
            # queries_positional_encoding = self.positional_encoding(query_masks, pe_units,pos_embedding_trainable=True)
            #queries_positional_encoding = self.querys_pe(self.querys_idx) * pe_units ** 0.5
            #queries += queries_positional_encoding
            queries = self.qpe(queries)
            # keys_positional_encoding = self.positional_encoding(key_masks, pe_units, pos_embedding_trainable=True)
            #keys_positional_encoding = self.keys_pe(self.keys_idx) * pe_units ** 0.5
            #keys += keys_positional_encoding
            keys = self.kpe(keys)

        querys = tf.tensordot(queries, self.W_Query, axes=(-1, 0))  # None T_q D*head_num
        keys = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        values = tf.tensordot(keys, self.W_Value, axes=(-1, 0))

        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)  # head_num*None T_q D
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0)

        outputs = tf.matmul(querys, keys, transpose_b=True)  # head_num*None T_q T_k

        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.tile(key_masks, [self.head_num, 1])

        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

        # (h*N, T_q, T_k)

        outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )

        outputs -= tf.reduce_max(outputs, axis=-1, keep_dims=True)
        outputs = tf.nn.softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # (h*N, T_q, T_k)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])

        outputs *= query_masks

        outputs = self.dropout(outputs)
        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, values)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            result += queries  # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
        if self.use_layer_norm:
            result = self.ln(result)

        if self.use_feed_forward:
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

        return tf.reduce_mean(result, axis=1, keep_dims=True)

    def compute_output_shape(self, input_shape):

        return (None, 1, self.att_embedding_size * self.head_num)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num,
                  'dropout_rate': self.dropout_rate, 'use_res': self.use_res,
                  'use_positional_encoding': self.use_positional_encoding, 'use_feed_forward': self.use_feed_forward,
                  'use_layer_norm': self.use_layer_norm, 'seed': self.seed, 'supports_masking': self.supports_masking}
        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def positional_encoding(self, inputs,
                            num_units,
                            pos_embedding_trainable=True,
                            zero_pad=False,
                            scale=True,
                            ):
        '''Sinusoidal Positional_Encoding.
        Args:
          inputs: A 2d Tensor with shape of (N, T).
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
            A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
        '''

        N, T = inputs.get_shape().as_list()
        # with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.expand_dims(tf.range(T), 0)
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        if pos_embedding_trainable:
            # lookup_table = tf.variable('trainable_pos_embedding', initializer=lookup_table, dtype=tf.float32)
            lookup_table = tf.Variable(initial_value=lookup_table, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)

        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5
        return outputs

    def add_pe_weight(self, input_shape, pos_embedding_trainable=True, zero_pad=False, scale=True):
        # pe_units = input_shape[-1].value
        N, T, num_units = input_shape.as_list()
        # with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.expand_dims(tf.range(T), 0)
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.keras.layers.Embedding(T, num_units, weights=[position_enc],
                                                 trainable=pos_embedding_trainable)
        return position_ind, lookup_table


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32' ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = tf.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        #tf.cum
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)