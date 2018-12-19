import itertools
from tensorflow.python.keras.layers import Layer,Activation,BatchNormalization
from tensorflow.python.keras.regularizers import  l2
from tensorflow.python.keras.initializers import Zeros,glorot_normal,glorot_uniform
from tensorflow.python.keras import backend as K

import tensorflow as tf


class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

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
        if len(input_shape) !=3 :
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions"% (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):

        if K.ndim(inputs) !=3 :
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions"% (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum =  tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term
    
    def compute_output_shape(self, input_shape):
        return (None, 1)


class AFMLayer(Layer):
    """Attentonal Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      Arguments

        - **attention_factor** : Positive integer, dimensionality of the attention network output space.

        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength applied to attention network.

        - **keep_prob** : float between 0 and 1. Fraction of the attention net output units to keep.

        - **seed** : A Python integer to use as random seed.

      References
        - [Attentional Factorization Machines : Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(self, attention_factor=4, l2_reg_w=0,keep_prob=1.0,seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.keep_prob = keep_prob
        self.seed = seed
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
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
                             'inputs of a list with same shape tensor like (None,1,embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))



        embedding_size = input_shape[0][-1].value

        self.attention_W = self.add_weight(shape=(embedding_size, self.attention_factor), initializer=glorot_normal(seed=self.seed),regularizer=l2(self.l2_reg_w),
                                           name="attention_W")
        self.attention_b = self.add_weight(shape=(self.attention_factor,), initializer=Zeros(), name="attention_b")
        self.projection_h = self.add_weight(shape=(self.attention_factor, 1), initializer=glorot_normal(seed=self.seed),
                                            name="projection_h")
        self.projection_p = self.add_weight(shape=(embedding_size, 1), initializer=glorot_normal(seed=self.seed), name="projection_p")


        super(AFMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embeds_vec_list = inputs
        row = []
        col = []
        # num_inputs = len(embeds_vec_list)
        # for i in range(num_inputs - 1):
        #     for j in range(i + 1, num_inputs):
        #         row.append(i)
        #         col.append(j)
        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)
        #p = tf.concat([embeds_vec_list[idx] for idx in row],axis=1)
        #q = tf.concat([embeds_vec_list[idx] for idx in col], axis=1)
        p = tf.concat(row,axis=1)
        q = tf.concat(col,axis=1)
        inner_product = p * q

        bi_interaction = inner_product
        attention_temp =  tf.nn.relu(tf.nn.bias_add(tf.tensordot(bi_interaction,self.attention_W,axes=(-1,0)),self.attention_b))
        #  Dense(self.attention_factor,'relu',kernel_regularizer=l2(self.l2_reg_w))(bi_interaction)
        self.normalized_att_score =tf.nn.softmax(tf.tensordot(attention_temp,self.projection_h,axes=(-1,0)),dim=1)
        attention_output = tf.reduce_sum(self.normalized_att_score*bi_interaction,axis=1)

        attention_output = tf.nn.dropout(attention_output,self.keep_prob,seed=1024)
        # Dropout(1-self.keep_prob)(attention_output)
        afm_out = tf.tensordot(attention_output,self.projection_p,axes=(-1,0))

        return afm_out

    def compute_output_shape(self, input_shape):

        if not isinstance(input_shape, list):
            raise ValueError('A `AFMLayer` layer should be called '
                             'on a list of inputs.')
        return (None, 1)

    def get_config(self,):
        config = {'attention_factor': self.attention_factor,'l2_reg_w':self.l2_reg_w, 'keep_prob':self.keep_prob, 'seed':self.seed}
        base_config = super(AFMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PredictionLayer(Layer):

    def __init__(self, activation='sigmoid',use_bias=True, **kwargs):
        self.activation = activation
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(shape=(1,), initializer=Zeros(),name="global_bias")

        super(PredictionLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x,self.global_bias,data_format='NHWC')

        if isinstance(self.activation,str):
            output = Activation(self.activation)(x)
        elif issubclass(self.activation,Layer):
            output = self.activation()(x)
        else:
            raise ValueError("Invalid activation of MLP,found %s.You should use a str or a Activation Layer Class." % (self.activation))
        output = tf.reshape(output,(-1,1))

        return output

    def compute_output_shape(self, input_shape):
        return (None,1)

    def get_config(self,):
        config = {'activation': self.activation,'use_bias':self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model,which leans both low and high degree cross feature.

      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Arguments
        - **layer_num**: Positive integer, the cross layer number

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix

        - **seed**: A Python integer to use as random seed.

      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """
    def __init__(self, layer_num=2,l2_reg=0,seed=1024, **kwargs):
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.seed = seed
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = input_shape[-1].value
        self.kernels = [self.add_weight(name='kernel'+str(i),
                                        shape=(dim, 1),
                                       initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(self.layer_num)]
        self.bias = [self.add_weight(name='bias'+str(i) ,
                                     shape=(dim,1),
                                    initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        super(CrossNet, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):
        if K.ndim(inputs) !=2 :
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions"% (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs,axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = tf.tensordot(tf.transpose(x_l,[0,2,1]),self.kernels[i],axes=(-1,0))
            dot_ = tf.matmul(x_0,xl_w)
            x_l = dot_ + x_l + self.bias[i]
        x_l = tf.squeeze(x_l,axis=2)
        return x_l

    def get_config(self,):

        config = {'layer_num': self.layer_num,'l2_reg':self.l2_reg,'seed':self.seed}
        base_config = super(CrossNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


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

    def __init__(self,  hidden_size, activation='relu',l2_reg=0, keep_prob=1, use_bn=False,seed=1024,**kwargs):
        self.hidden_size = hidden_size
        self.activation =activation
        self.keep_prob = keep_prob
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(MLP, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_size)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(hidden_units[i], hidden_units[i+1]),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_size))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_size[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_size))]

        super(MLP, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):
        deep_input = inputs

        for i in range(len(self.hidden_size)):
            fc = tf.nn.bias_add(tf.tensordot(deep_input,self.kernels[i],axes=(-1,0)),self.bias[i])
            #fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = BatchNormalization()(fc)

            if isinstance(self.activation,str):
                fc = Activation(self.activation)(fc)
            elif issubclass(self.activation,Layer):
                fc = self.activation()(fc)
            else:
                raise ValueError("Invalid activation of MLP,found %s.You should use a str or a Activation Layer Class."%(self.activation))
            fc = tf.nn.dropout(fc,self.keep_prob)

            deep_input = fc

        return deep_input
    def compute_output_shape(self, input_shape):
        if len(self.hidden_size) > 0:
            shape = input_shape[:-1] + (self.hidden_size[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self,):
        config = {'activation': self.activation,'hidden_size':self.hidden_size, 'l2_reg':self.l2_reg,'use_bn':self.use_bn, 'keep_prob':self.keep_prob,'seed': self.seed}
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BiInteractionPooling(Layer):
    """Bi-Interaction Layer used in Neural FM,compress the pairwise element-wise product of features into one single vector.

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

        if len(input_shape) != 3 :
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions"% (len(input_shape)))

        super(BiInteractionPooling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):

        if K.ndim(inputs) !=3 :
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions"% (K.ndim(inputs)))

        concated_embeds_value = inputs
        square_of_sum =  tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = 0.5*(square_of_sum - sum_of_square)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[-1])

class OutterProductLayer(Layer):
    """OutterProduct Layer used in PNN.This implemention  is adapted from code that the author of the paper published on https://github.com/Atomu2014/product-nets.

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

        if  len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A `OutterProductLayer` layer requires '
                             'inputs of a list with same shape tensor like (None,1,embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        input_shape = input_shape[0]
        embed_size = input_shape[-1].value
        if self.kernel_type == 'mat':

            self.kernel = self.add_weight(shape=(embed_size,num_pairs,embed_size), initializer=glorot_uniform(seed=self.seed),
                                      name='kernel')
        elif self.kernel_type == 'vec':
            self.kernel = self.add_weight(shape=(num_pairs,embed_size,),initializer=glorot_uniform(self.seed),name='kernel'
                                          )
        elif self.kernel_type == 'num':
            self.kernel = self.add_weight(shape=(num_pairs,1),initializer=glorot_uniform(self.seed),name='kernel')

        super(OutterProductLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([embed_list[idx] for idx in row],axis=1)  # batch num_pairs k
        q = tf.concat([embed_list[idx] for idx in col],axis=1)  # Reshape([num_pairs, self.embedding_size])

        #-------------------------
        if self.kernel_type == 'mat':
            p = tf.expand_dims(p, 1)
            # k     k* pair* k
            # batch * pair
            kp = tf.reduce_sum(

                # batch * pair * k

                tf.multiply(

                    # batch * pair * k

                    tf.transpose(

                        # batch * k * pair

                        tf.reduce_sum(

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

            kp = tf.reduce_sum(p * q * k, -1)

            # p q # b * p * k

        return kp

    def compute_output_shape(self, input_shape):
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        return (None, num_pairs)

    def get_config(self,):
        config = {'kernel_type': self.kernel_type,'seed': self.seed}
        base_config = super(OutterProductLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class InnerProductLayer(Layer):
    """InnerProduct Layer used in PNN that compute the element-wise product or inner product between feature vectors.

      Input shape
        - A list of N 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape: ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.

      Arguments
        - **reduce_sum**: bool. Whether return inner product or element-wise product

      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self,reduce_sum=True,**kwargs):
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
        super(InnerProductLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):
        if K.ndim(inputs[0]) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)
        #num_pairs = int(num_inputs * (num_inputs - 1) / 2)


        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([embed_list[idx] for idx in row],axis=1)# batch num_pairs k
        q = tf.concat([embed_list[idx] for idx in col],axis=1)  # Reshape([num_pairs, self.embedding_size])
        inner_product = p * q
        if self.reduce_sum:
            inner_product = tf.reduce_sum(inner_product, axis=2, keep_dims=True)
        return inner_product


    def compute_output_shape(self, input_shape):
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        input_shape = input_shape[0]
        embed_size = input_shape[-1]
        if self.reduce_sum:
            return (input_shape[0],num_pairs,1)
        else:
            return (input_shape[0],num_pairs,embed_size)

    def get_config(self,):
        config = {'reduce_sum': self.reduce_sum,}
        base_config = super(InnerProductLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of user interests varies adaptively given different candidate items.

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

    def __init__(self,hidden_size=(64,32), activation='sigmoid',l2_reg=0, keep_prob=1, use_bn=False,seed=1024,**kwargs):
        self.hidden_size = hidden_size
        self.activation = activation
        self.l2_reg = l2_reg
        self.keep_prob = keep_prob
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `LocalActivationUnit` layer should be called '
                             'on a list of 2 inputs')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3 :
                raise ValueError("Unexpected inputs dimensions %d and %d, expect to be 3 dimensions" % (len(input_shape[0]),len(input_shape[1])))

        if input_shape[0][-1]!=input_shape[1][-1] or input_shape[0][1]!=1:

            raise ValueError('A `LocalActivationUnit` layer requires '
                             'inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)'
                             'Got different shapes: %s,%s' % (input_shape))
        size = 4*int(input_shape[0][-1]) if len(self.hidden_size) == 0 else self.hidden_size[-1]
        self.kernel = self.add_weight(shape=(size, 1),
                                           initializer=glorot_normal(seed=self.seed),
                                           name="kernel")
        self.bias = self.add_weight(shape=(1,), initializer=Zeros(), name="bias")
        super(LocalActivationUnit, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs,**kwargs):


        query,keys = inputs

        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query,keys_len,1)

        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        att_input = tf.layers.batch_normalization(att_input)
        att_out = MLP(self.hidden_size, self.activation, self.l2_reg, self.keep_prob, self.use_bn, seed=self.seed)(att_input)
        attention_score = tf.nn.bias_add(tf.tensordot(att_out,self.kernel,axes=(-1,0)),self.bias)

        return attention_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def get_config(self,):
        config = {'activation': self.activation,'hidden_size':self.hidden_size, 'l2_reg':self.l2_reg, 'keep_prob':self.keep_prob,'use_bn':self.use_bn,'seed': self.seed}
        base_config = super(LocalActivationUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

