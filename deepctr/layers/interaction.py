# -*- coding:utf-8 -*-
"""

Authors:
    Weichen Shen,weichenswc@163.com,
    Harshit Pande,
    Yi He, heyi_jack@163.com

"""

import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import batch_dot

try:
    from tensorflow.python.ops.init_ops import Zeros, Ones, Constant, TruncatedNormal, \
        glorot_normal_initializer as glorot_normal, \
        glorot_uniform_initializer as glorot_uniform
except ImportError:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, Constant, TruncatedNormal, glorot_normal, glorot_uniform

from tensorflow.python.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers import utils

from .activation import activation_layer
from .utils import concat_func, reduce_sum, softmax, reduce_mean
from .core import DNN


class AFMLayer(Layer):
    """Attentonal Factorization Machine models pairwise (order-2) feature
    interactions without linear term and bias.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      Arguments
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.

        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.

        - **dropout_rate** : float between in [0,1). Fraction of the attention net output units to dropout.

        - **seed** : A Python integer to use as random seed.

      References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(self, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            # input_shape = input_shape[0]
            # if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')

        shape_set = set()
        reduced_input_shape = [shape.as_list() for shape in input_shape]
        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_input_shape[i]))

        if len(shape_set) > 1:
            raise ValueError('A `AttentionalFM` layer requires '
                             'inputs with same shapes '
                             'Got different shapes: %s' % (shape_set))

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A `AttentionalFM` layer requires '
                             'inputs of a list with same shape tensor like\
                             (None, 1, embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))

        embedding_size = int(input_shape[0][-1])

        self.attention_W = self.add_weight(shape=(embedding_size,
                                                  self.attention_factor), initializer=glorot_normal(seed=self.seed),
                                           regularizer=l2(self.l2_reg_w), name="attention_W")
        self.attention_b = self.add_weight(
            shape=(self.attention_factor,), initializer=Zeros(), name="attention_b")
        self.projection_h = self.add_weight(shape=(self.attention_factor, 1),
                                            initializer=glorot_normal(seed=self.seed), name="projection_h")
        self.projection_p = self.add_weight(shape=(
            embedding_size, 1), initializer=glorot_normal(seed=self.seed), name="projection_p")
        self.dropout = Dropout(
            self.dropout_rate, seed=self.seed)

        self.tensordot = Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(AFMLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embeds_vec_list = inputs
        row = []
        col = []

        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        p = tf.concat(row, axis=1)
        q = tf.concat(col, axis=1)
        inner_product = p * q

        bi_interaction = inner_product
        attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
            bi_interaction, self.attention_W, axes=(-1, 0)), self.attention_b))
        #  Dense(self.attention_factor,'relu',kernel_regularizer=l2(self.l2_reg_w))(bi_interaction)
        self.normalized_att_score = softmax(tf.tensordot(
            attention_temp, self.projection_h, axes=(-1, 0)), dim=1)
        attention_output = reduce_sum(
            self.normalized_att_score * bi_interaction, axis=1)

        attention_output = self.dropout(attention_output, training=training)  # training

        afm_out = self.tensordot([attention_output, self.projection_p])
        return afm_out

    def compute_output_shape(self, input_shape):

        if not isinstance(input_shape, list):
            raise ValueError('A `AFMLayer` layer should be called '
                             'on a list of inputs.')
        return (None, 1)

    def get_config(self, ):
        config = {'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(AFMLayer, self).get_config()
        base_config.update(config)
        return base_config


class BiInteractionPooling(Layer):
    """Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector.

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      References
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """

    def __init__(self, **kwargs):

        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        super(BiInteractionPooling, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        concated_embeds_value = inputs
        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[-1])


class CIN(Layer):
    """Compressed Interaction Network used in xDeepFM.This implemention is
    adapted from code that the author of the paper published on https://github.com/Leavingseason/xDeepFM.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .

      Arguments
        - **layer_size** : list of int.Feature maps in each layer.

        - **activation** : activation function used on feature maps.

        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.

        - **seed** : A Python integer to use as random seed.

      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024, **kwargs):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.split_half = split_half
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        self.field_nums = [int(input_shape[1])]
        self.filters = []
        self.bias = []
        for i, size in enumerate(self.layer_size):

            self.filters.append(self.add_weight(name='filter' + str(i),
                                                shape=[1, self.field_nums[-1]
                                                       * self.field_nums[0], size],
                                                dtype=tf.float32, initializer=glorot_uniform(
                    seed=self.seed + i),
                                                regularizer=l2(self.l2_reg)))

            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=Zeros()))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.activation_layers = [activation_layer(
            self.activation) for _ in self.layer_size]

        super(CIN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        dim = int(inputs.get_shape()[-1])
        hidden_nn_layers = [inputs]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)

            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)

            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])

            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')

            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = reduce_sum(result, -1, keep_dims=False)

        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(
                self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return (None, featuremap_num)

    def get_config(self, ):

        config = {'layer_size': self.layer_size, 'split_half': self.split_half, 'activation': self.activation,
                  'seed': self.seed}
        base_config = super(CIN, self).get_config()
        base_config.update(config)
        return base_config


class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.

      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Arguments
        - **layer_num**: Positive integer, the cross layer number

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix

        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.

        - **seed**: A Python integer to use as random seed.

      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.l2_reg = l2_reg
        self.seed = seed
        print('CrossNet parameterization:', self.parameterization)
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])
        if self.parameterization == 'vector':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, 1),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        elif self.parameterization == 'matrix':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, dim),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
                dot_ = tf.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)  # W * xi  (bs, dim, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'layer_num': self.layer_num, 'parameterization': self.parameterization,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNet, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class CrossNetMix(Layer):
    """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space

      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Arguments
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.

        - **num_experts** : Positive integer, number of experts.

        - **layer_num**: Positive integer, the cross layer number

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix

        - **seed**: A Python integer to use as random seed.

      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, low_rank=32, num_experts=4, layer_num=2, l2_reg=0, seed=1024, **kwargs):
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.seed = seed
        super(CrossNetMix, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])

        # U: (dim, low_rank)
        self.U_list = [self.add_weight(name='U_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]
        # V: (dim, low_rank)
        self.V_list = [self.add_weight(name='V_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]
        # C: (low_rank, low_rank)
        self.C_list = [self.add_weight(name='C_list' + str(i),
                                       shape=(self.num_experts, self.low_rank, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]

        self.gating = [Dense(1, use_bias=False) for i in range(self.num_experts)]

        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNetMix, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](tf.squeeze(x_l, axis=2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = tf.einsum('ij,bjk->bik', tf.transpose(self.V_list[i][expert_id]), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = tf.nn.tanh(v_x)
                v_x = tf.einsum('ij,bjk->bik', self.C_list[i][expert_id], v_x)  # (bs, low_rank, 1)
                v_x = tf.nn.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = tf.einsum('ij,bjk->bik', self.U_list[i][expert_id], v_x)  # (bs, dim, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(tf.squeeze(dot_, axis=2))

            # (3) mixture of low-rank experts
            output_of_experts = tf.stack(output_of_experts, 2)  # (bs, dim, num_experts)
            gating_score_of_experts = tf.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = tf.matmul(output_of_experts, tf.nn.softmax(gating_score_of_experts, 1))
            x_l = moe_out + x_l  # (bs, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'low_rank': self.low_rank, 'num_experts': self.num_experts, 'layer_num': self.layer_num,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNetMix, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


class InnerProductLayer(Layer):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.

      Input shape
        - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape: ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.

      Arguments
        - **reduce_sum**: bool. Whether return inner product or element-wise product

      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, reduce_sum=True, **kwargs):
        self.reduce_sum = reduce_sum
        super(InnerProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `InnerProductLayer` layer should be called '
                             'on a list of at least 2 inputs')

        reduced_inputs_shapes = [shape.as_list() for shape in input_shape]
        shape_set = set()

        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_inputs_shapes[i]))

        if len(shape_set) > 1:
            raise ValueError('A `InnerProductLayer` layer requires '
                             'inputs with same shapes '
                             'Got different shapes: %s' % (shape_set))

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A `InnerProductLayer` layer requires '
                             'inputs of a list with same shape tensor like (None,1,embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))
        super(InnerProductLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([embed_list[idx]
                       for idx in row], axis=1)  # batch num_pairs k
        q = tf.concat([embed_list[idx]
                       for idx in col], axis=1)

        inner_product = p * q
        if self.reduce_sum:
            inner_product = reduce_sum(
                inner_product, axis=2, keep_dims=True)
        return inner_product

    def compute_output_shape(self, input_shape):
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        input_shape = input_shape[0]
        embed_size = input_shape[-1]
        if self.reduce_sum:
            return (input_shape[0], num_pairs, 1)
        else:
            return (input_shape[0], num_pairs, embed_size)

    def get_config(self, ):
        config = {'reduce_sum': self.reduce_sum, }
        base_config = super(InnerProductLayer, self).get_config()
        base_config.update(config)
        return base_config


class InteractingLayer(Layer):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.

      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
            - 3D tensor with shape:``(batch_size,field_size,att_embedding_size * head_num)``.


      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.

      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling
        super(InteractingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                         dtype=tf.float32,
                                         initializer=TruncatedNormal(seed=self.seed))

        # Be sure to call this somewhere!
        super(InteractingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        querys = tf.tensordot(inputs, self.W_Query,
                              axes=(-1, 0))  # None F D*head_num
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        # head_num None F D
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(
            querys, keys, transpose_b=True)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores,
                           values)  # head_num None F D
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0)  # None F D*head_num

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        result = tf.nn.relu(result)

        return result

    def compute_output_shape(self, input_shape):

        return (None, input_shape[1], self.att_embedding_size * self.head_num)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num, 'use_res': self.use_res,
                  'seed': self.seed}
        base_config = super(InteractingLayer, self).get_config()
        base_config.update(config)
        return base_config


class OutterProductLayer(Layer):
    """OutterProduct Layer used in PNN.This implemention is
    adapted from code that the author of the paper published on https://github.com/Atomu2014/product-nets.

      Input shape
            - A list of N 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
            - 2D tensor with shape:``(batch_size,N*(N-1)/2 )``.

      Arguments
            - **kernel_type**: str. The kernel weight matrix type to use,can be mat,vec or num

            - **seed**: A Python integer to use as random seed.

      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, kernel_type='mat', seed=1024, **kwargs):
        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")
        self.kernel_type = kernel_type
        self.seed = seed
        super(OutterProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `OutterProductLayer` layer should be called '
                             'on a list of at least 2 inputs')

        reduced_inputs_shapes = [shape.as_list() for shape in input_shape]
        shape_set = set()

        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_inputs_shapes[i]))

        if len(shape_set) > 1:
            raise ValueError('A `OutterProductLayer` layer requires '
                             'inputs with same shapes '
                             'Got different shapes: %s' % (shape_set))

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A `OutterProductLayer` layer requires '
                             'inputs of a list with same shape tensor like (None,1,embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        input_shape = input_shape[0]
        embed_size = int(input_shape[-1])
        if self.kernel_type == 'mat':

            self.kernel = self.add_weight(shape=(embed_size, num_pairs, embed_size),
                                          initializer=glorot_uniform(
                                              seed=self.seed),
                                          name='kernel')
        elif self.kernel_type == 'vec':
            self.kernel = self.add_weight(shape=(num_pairs, embed_size,), initializer=glorot_uniform(self.seed),
                                          name='kernel'
                                          )
        elif self.kernel_type == 'num':
            self.kernel = self.add_weight(
                shape=(num_pairs, 1), initializer=glorot_uniform(self.seed), name='kernel')

        super(OutterProductLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([embed_list[idx]
                       for idx in row], axis=1)  # batch num_pairs k
        # Reshape([num_pairs, self.embedding_size])
        q = tf.concat([embed_list[idx] for idx in col], axis=1)

        # -------------------------
        if self.kernel_type == 'mat':
            p = tf.expand_dims(p, 1)
            # k     k* pair* k
            # batch * pair
            kp = reduce_sum(

                # batch * pair * k

                tf.multiply(

                    # batch * pair * k

                    tf.transpose(

                        # batch * k * pair

                        reduce_sum(

                            # batch * k * pair * k

                            tf.multiply(

                                p, self.kernel),

                            -1),

                        [0, 2, 1]),

                    q),

                -1)
        else:
            # 1 * pair * (k or 1)

            k = tf.expand_dims(self.kernel, 0)

            # batch * pair

            kp = reduce_sum(p * q * k, -1)

            # p q # b * p * k

        return kp

    def compute_output_shape(self, input_shape):
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        return (None, num_pairs)

    def get_config(self, ):
        config = {'kernel_type': self.kernel_type, 'seed': self.seed}
        base_config = super(OutterProductLayer, self).get_config()
        base_config.update(config)
        return base_config


class FGCNNLayer(Layer):
    """Feature Generation Layer used in FGCNN,including Convolution,MaxPooling and Recombination.

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,new_feture_num,embedding_size)``.

      References
        - [Liu B, Tang R, Chen Y, et al. Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1904.04447, 2019.](https://arxiv.org/pdf/1904.04447)

    """

    def __init__(self, filters=(14, 16,), kernel_width=(7, 7,), new_maps=(3, 3,), pooling_width=(2, 2),
                 **kwargs):
        if not (len(filters) == len(kernel_width) == len(new_maps) == len(pooling_width)):
            raise ValueError("length of argument must be equal")
        self.filters = filters
        self.kernel_width = kernel_width
        self.new_maps = new_maps
        self.pooling_width = pooling_width

        super(FGCNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        self.conv_layers = []
        self.pooling_layers = []
        self.dense_layers = []
        pooling_shape = input_shape.as_list() + [1, ]
        embedding_size = int(input_shape[-1])
        for i in range(1, len(self.filters) + 1):
            filters = self.filters[i - 1]
            width = self.kernel_width[i - 1]
            new_filters = self.new_maps[i - 1]
            pooling_width = self.pooling_width[i - 1]
            conv_output_shape = self._conv_output_shape(
                pooling_shape, (width, 1))
            pooling_shape = self._pooling_output_shape(
                conv_output_shape, (pooling_width, 1))
            self.conv_layers.append(Conv2D(filters=filters, kernel_size=(width, 1), strides=(1, 1),
                                           padding='same',
                                           activation='tanh', use_bias=True, ))
            self.pooling_layers.append(
                MaxPooling2D(pool_size=(pooling_width, 1)))
            self.dense_layers.append(Dense(pooling_shape[1] * embedding_size * new_filters,
                                           activation='tanh', use_bias=True))

        self.flatten = Flatten()

        super(FGCNNLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embedding_size = int(inputs.shape[-1])
        pooling_result = tf.expand_dims(inputs, axis=3)

        new_feature_list = []

        for i in range(1, len(self.filters) + 1):
            new_filters = self.new_maps[i - 1]

            conv_result = self.conv_layers[i - 1](pooling_result)

            pooling_result = self.pooling_layers[i - 1](conv_result)

            flatten_result = self.flatten(pooling_result)

            new_result = self.dense_layers[i - 1](flatten_result)

            new_feature_list.append(
                tf.reshape(new_result, (-1, int(pooling_result.shape[1]) * new_filters, embedding_size)))

        new_features = concat_func(new_feature_list, axis=1)
        return new_features

    def compute_output_shape(self, input_shape):

        new_features_num = 0
        features_num = input_shape[1]

        for i in range(0, len(self.kernel_width)):
            pooled_features_num = features_num // self.pooling_width[i]
            new_features_num += self.new_maps[i] * pooled_features_num
            features_num = pooled_features_num

        return (None, new_features_num, input_shape[-1])

    def get_config(self, ):
        config = {'kernel_width': self.kernel_width, 'filters': self.filters, 'new_maps': self.new_maps,
                  'pooling_width': self.pooling_width}
        base_config = super(FGCNNLayer, self).get_config()
        base_config.update(config)
        return base_config

    def _conv_output_shape(self, input_shape, kernel_size):
        # channels_last
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='same',
                stride=1,
                dilation=1)
            new_space.append(new_dim)
        return ([input_shape[0]] + new_space + [self.filters])

    def _pooling_output_shape(self, input_shape, pool_size):
        # channels_last

        rows = input_shape[1]
        cols = input_shape[2]
        rows = utils.conv_output_length(rows, pool_size[0], 'valid',
                                        pool_size[0])
        cols = utils.conv_output_length(cols, pool_size[1], 'valid',
                                        pool_size[1])
        return [input_shape[0], rows, cols, input_shape[3]]


class SENETLayer(Layer):
    """SENETLayer used in FiBiNET.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.

        - **seed** : A Python integer to use as random seed.

      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio

        self.seed = seed
        super(SENETLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')

        self.filed_size = len(input_shape)
        self.embedding_size = input_shape[0][-1]
        reduction_size = max(1, self.filed_size // self.reduction_ratio)

        self.W_1 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        inputs = concat_func(inputs, axis=1)
        Z = reduce_mean(inputs, axis=-1, )

        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))

        return tf.split(V, self.filed_size, axis=1)

    def compute_output_shape(self, input_shape):

        return input_shape

    def compute_mask(self, inputs, mask=None):
        return [None] * self.filed_size

    def get_config(self, ):
        config = {'reduction_ratio': self.reduction_ratio, 'seed': self.seed}
        base_config = super(SENETLayer, self).get_config()
        base_config.update(config)
        return base_config


class BilinearInteraction(Layer):
    """BilinearInteraction Layer used in FiBiNET.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``. Its length is ``filed_size``.

      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2,embedding_size)``.

      Arguments
        - **bilinear_type** : String, types of bilinear functions used in this layer.

        - **seed** : A Python integer to use as random seed.

      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)

    """

    def __init__(self, bilinear_type="interaction", seed=1024, **kwargs):
        self.bilinear_type = bilinear_type
        self.seed = seed

        super(BilinearInteraction, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')
        embedding_size = int(input_shape[0][-1])

        if self.bilinear_type == "all":
            self.W = self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(len(input_shape) - 1)]
        elif self.bilinear_type == "interaction":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(len(input_shape)), 2)]
        else:
            raise NotImplementedError

        super(BilinearInteraction, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        n = len(inputs)
        if self.bilinear_type == "all":
            vidots = [tf.tensordot(inputs[i], self.W, axes=(-1, 0)) for i in range(n)]
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "each":
            vidots = [tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)) for i in range(n - 1)]
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError
        output = concat_func(p, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        filed_size = len(input_shape)
        embedding_size = input_shape[0][-1]

        return (None, filed_size * (filed_size - 1) // 2, embedding_size)

    def get_config(self, ):
        config = {'bilinear_type': self.bilinear_type, 'seed': self.seed}
        base_config = super(BilinearInteraction, self).get_config()
        base_config.update(config)
        return base_config


class FieldWiseBiInteraction(Layer):
    """Field-Wise Bi-Interaction Layer used in FLEN,compress the
     pairwise element-wise product of features into one single vector.

      Input shape
        - A list of 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size,embedding_size)``.

      Arguments
        - **use_bias** : Boolean, if use bias.
        - **seed** : A Python integer to use as random seed.

      References
        - [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690)

    """

    def __init__(self, use_bias=True, seed=1024, **kwargs):
        self.use_bias = use_bias
        self.seed = seed

        super(FieldWiseBiInteraction, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError(
                'A `Field-Wise Bi-Interaction` layer should be called '
                'on a list of at least 2 inputs')

        self.num_fields = len(input_shape)
        embedding_size = input_shape[0][-1]

        self.kernel_mf = self.add_weight(
            name='kernel_mf',
            shape=(int(self.num_fields * (self.num_fields - 1) / 2), 1),
            initializer=Ones(),
            regularizer=None,
            trainable=True)

        self.kernel_fm = self.add_weight(
            name='kernel_fm',
            shape=(self.num_fields, 1),
            initializer=Constant(value=0.5),
            regularizer=None,
            trainable=True)
        if self.use_bias:
            self.bias_mf = self.add_weight(name='bias_mf',
                                           shape=(embedding_size),
                                           initializer=Zeros())
            self.bias_fm = self.add_weight(name='bias_fm',
                                           shape=(embedding_size),
                                           initializer=Zeros())

        super(FieldWiseBiInteraction,
              self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" %
                (K.ndim(inputs)))

        field_wise_embeds_list = inputs

        # MF module
        field_wise_vectors = tf.concat([
            reduce_sum(field_i_vectors, axis=1, keep_dims=True)
            for field_i_vectors in field_wise_embeds_list
        ], 1)

        left = []
        right = []

        for i, j in itertools.combinations(list(range(self.num_fields)), 2):
            left.append(i)
            right.append(j)

        embeddings_left = tf.gather(params=field_wise_vectors,
                                    indices=left,
                                    axis=1)
        embeddings_right = tf.gather(params=field_wise_vectors,
                                     indices=right,
                                     axis=1)

        embeddings_prod = embeddings_left * embeddings_right
        field_weighted_embedding = embeddings_prod * self.kernel_mf
        h_mf = reduce_sum(field_weighted_embedding, axis=1)
        if self.use_bias:
            h_mf = tf.nn.bias_add(h_mf, self.bias_mf)

        # FM module
        square_of_sum_list = [
            tf.square(reduce_sum(field_i_vectors, axis=1, keep_dims=True))
            for field_i_vectors in field_wise_embeds_list
        ]
        sum_of_square_list = [
            reduce_sum(field_i_vectors * field_i_vectors,
                       axis=1,
                       keep_dims=True)
            for field_i_vectors in field_wise_embeds_list
        ]

        field_fm = tf.concat([
            square_of_sum - sum_of_square for square_of_sum, sum_of_square in
            zip(square_of_sum_list, sum_of_square_list)
        ], 1)

        h_fm = reduce_sum(field_fm * self.kernel_fm, axis=1)
        if self.use_bias:
            h_fm = tf.nn.bias_add(h_fm, self.bias_fm)

        return h_mf + h_fm

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][-1])

    def get_config(self, ):
        config = {'use_bias': self.use_bias, 'seed': self.seed}
        base_config = super(FieldWiseBiInteraction, self).get_config()
        base_config.update(config)
        return base_config


class FwFMLayer(Layer):
    """Field-weighted Factorization Machines

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      Arguments
        - **num_fields** : integer for number of fields
        - **regularizer** : L2 regularizer weight for the field strength parameters of FwFM

      References
        - [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising]
        https://arxiv.org/pdf/1806.03514.pdf
    """

    def __init__(self, num_fields=4, regularizer=0.000001, **kwargs):
        self.num_fields = num_fields
        self.regularizer = regularizer
        super(FwFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        if input_shape[1] != self.num_fields:
            raise ValueError("Mismatch in number of fields {} and \
                 concatenated embeddings dims {}".format(self.num_fields, input_shape[1]))

        self.field_strengths = self.add_weight(name='field_pair_strengths',
                                               shape=(self.num_fields, self.num_fields),
                                               initializer=TruncatedNormal(),
                                               regularizer=l2(self.regularizer),
                                               trainable=True)

        super(FwFMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        if inputs.shape[1] != self.num_fields:
            raise ValueError("Mismatch in number of fields {} and \
                 concatenated embeddings dims {}".format(self.num_fields, inputs.shape[1]))

        pairwise_inner_prods = []
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            # get field strength for pair fi and fj
            r_ij = self.field_strengths[fi, fj]

            # get embeddings for the features of both the fields
            feat_embed_i = tf.squeeze(inputs[0:, fi:fi + 1, 0:], axis=1)
            feat_embed_j = tf.squeeze(inputs[0:, fj:fj + 1, 0:], axis=1)

            f = tf.scalar_mul(r_ij, batch_dot(feat_embed_i, feat_embed_j, axes=1))
            pairwise_inner_prods.append(f)

        sum_ = tf.add_n(pairwise_inner_prods)
        return sum_

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = super(FwFMLayer, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields,
            'regularizer': self.regularizer
        })
        return config


class FEFMLayer(Layer):
    """Field-Embedded Factorization Machines

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape:
            ``(batch_size, (num_fields * (num_fields-1))/2)`` # concatenated FEFM interaction embeddings

      Arguments
        - **regularizer** : L2 regularizer weight for the field pair matrix embeddings parameters of FEFM

      References
        - [Field-Embedded Factorization Machines for Click-through Rate Prediction]
         https://arxiv.org/pdf/2009.09931.pdf
    """

    def __init__(self, regularizer, **kwargs):
        self.regularizer = regularizer
        super(FEFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                                expect to be 3 dimensions" % (len(input_shape)))

        self.num_fields = int(input_shape[1])
        embedding_size = int(input_shape[2])

        self.field_embeddings = {}
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            field_pair_id = str(fi) + "-" + str(fj)
            self.field_embeddings[field_pair_id] = self.add_weight(name='field_embeddings' + field_pair_id,
                                                                   shape=(embedding_size, embedding_size),
                                                                   initializer=TruncatedNormal(),
                                                                   regularizer=l2(self.regularizer),
                                                                   trainable=True)

        super(FEFMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        pairwise_inner_prods = []
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            field_pair_id = str(fi) + "-" + str(fj)
            feat_embed_i = tf.squeeze(inputs[0:, fi:fi + 1, 0:], axis=1)
            feat_embed_j = tf.squeeze(inputs[0:, fj:fj + 1, 0:], axis=1)
            field_pair_embed_ij = self.field_embeddings[field_pair_id]

            feat_embed_i_tr = tf.matmul(feat_embed_i, field_pair_embed_ij + tf.transpose(field_pair_embed_ij))

            f = batch_dot(feat_embed_i_tr, feat_embed_j, axes=1)
            pairwise_inner_prods.append(f)

        concat_vec = tf.concat(pairwise_inner_prods, axis=1)
        return concat_vec

    def compute_output_shape(self, input_shape):
        num_fields = int(input_shape[1])
        return (None, (num_fields * (num_fields - 1)) / 2)

    def get_config(self):
        config = super(FEFMLayer, self).get_config().copy()
        config.update({
            'regularizer': self.regularizer,
        })
        return config


class BridgeModule(Layer):
    """Bridge Module used in EDCN

      Input shape
        - A list of two 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

    Arguments
        - **bridge_type**: The type of bridge interaction, one of 'pointwise_addition', 'hadamard_product', 'concatenation', 'attention_pooling'

        - **activation**: Activation function to use.

      References
        - [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models.](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)

    """

    def __init__(self, bridge_type='hadamard_product', activation='relu', **kwargs):
        self.bridge_type = bridge_type
        self.activation = activation

        super(BridgeModule, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError(
                'A `BridgeModule` layer should be called '
                'on a list of 2 inputs')

        self.dnn_dim = int(input_shape[0][-1])
        if self.bridge_type == "concatenation":
            self.dense = Dense(self.dnn_dim, self.activation)
        elif self.bridge_type == "attention_pooling":
            self.dense_x = DNN([self.dnn_dim, self.dnn_dim], self.activation, output_activation='softmax')
            self.dense_h = DNN([self.dnn_dim, self.dnn_dim], self.activation, output_activation='softmax')

        super(BridgeModule, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        x, h = inputs
        if self.bridge_type == "pointwise_addition":
            return x + h
        elif self.bridge_type == "hadamard_product":
            return x * h
        elif self.bridge_type == "concatenation":
            return self.dense(tf.concat([x, h], axis=-1))
        elif self.bridge_type == "attention_pooling":
            a_x = self.dense_x(x)
            a_h = self.dense_h(h)
            return a_x * x + a_h * h

    def compute_output_shape(self, input_shape):
        return (None, self.dnn_dim)

    def get_config(self):
        base_config = super(BridgeModule, self).get_config().copy()
        config = {
            'bridge_type': self.bridge_type,
            'activation': self.activation
        }
        config.update(base_config)
        return config
