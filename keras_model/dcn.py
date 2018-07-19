# -*- coding:utf-8 -*-
"""
@author: shenweichen,wcshen1994@163.com
A keras implementation of Deep & Cross Network
Reference:
[1] Deep & Cross Network for Ad Click Predictions (https://arxiv.org/abs/1708.05123)
"""
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Embedding, Concatenate, Activation, Reshape, Flatten, Dropout
from keras.models import Model
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal, TruncatedNormal
from keras.layers import add
from keras.engine.topology import Layer
from keras.regularizers import l2


class CrossLayer(Layer):

    def __init__(self, layer_num=1, init_std=0.0001, l2_reg=0, **kwargs):
        self.dim = None
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.dim = input_shape[1]
        self.kernels = [self.add_weight(name='kernel',
                                        shape=(self.dim, 1),
                                        initializer='random_normal',
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(self.layer_num)]
        self.bias = [self.add_weight(name='kernel',
                                     shape=(1, 1),
                                     initializer='zeros',
                                     trainable=True) for i in range(self.layer_num)]
        super(CrossLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        x_0 = Reshape([self.dim, 1])(x)
        x_l = x_0
        for i in range(self.layer_num):
            dot_ = tf.matmul(x_0, tf.transpose(x_l, [0, 2, 1]))  # K.dot(x_0,K.transpose(x_l))
            dot_ = K.dot(dot_, self.kernels[i])
            x_l = add([dot_, self.bias[i], x_l])  # K.bias_add(dot_, self.bias)

        x_l = K.reshape(x_l, [-1, self.dim])
        return x_l


class OutputLayer(Layer):
    def __init__(self,activation='sigmoid', **kwargs):
        self.activation = activation
        super(OutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.global_bias = self.add_weight(shape=(1,),initializer='zeros',name="global_bias")
        super(OutputLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x,):
        """

        :param x:  None * X
        :return: None * X
        """
        output = Activation(self.activation)(K.bias_add(x,self.global_bias,data_format='channels_last'))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class DCN():
    def __init__(self,  feature_dim_dict, max_embedding_size=256, embedding_factor=6,
                 cross_num=2, hidden_size=(32,), l2_reg_embedding=0.00002, l2_reg_cross=0.00002, l2_reg_deep=0.00002,
                 init_std=0.0001, seed=1024, keep_prob=0.5, use_bn=False, final_activation='sigmoid',
                 checkpoint_path=None,  bias_dim={"sparse":{},"dense":[]},):
        if not isinstance(feature_dim_dict,dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
            raise ValueError("feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")
        #self.field_dim = feature_dim_dict["sparse"]

        #if cate_feature_name is not None and len(cate_feature_name) != field_dim:
        #    raise ValueError('cate_feature_name error')
        #self.field_dim = field_dim
        self.feature_dim = feature_dim_dict
        self.embedding_size = max_embedding_size  # 这个特征貌似改了没用
        self.embedding_factor = embedding_factor
        self.cross_num = cross_num
        self.hidden_size = hidden_size
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_cross = l2_reg_cross
        self.l2_reg_deep = l2_reg_deep
        self.init_std = init_std
        self.seed = seed
        self.keep_prob = keep_prob
        self.use_bn = use_bn
        self.activation = "relu"#activation
        self.final_activation = final_activation
        self.checkpoint_path = checkpoint_path
        self.model = self.create_model()

    def get_model(self, ):
        return self.model

    def create_model(self, ):
        cate_input, continuous_input= self.get_input()
        cate_embedding, linear_embedding = self.create_cate_embedding()
        embed_list = [cate_embedding[i](cate_input[i]) for i in range(len(cate_input))]
        # input_list = embed_list + continuous_input

        linear_term = Flatten()(add([linear_embedding[i](cate_input[i]) for i in range(len(cate_input))]))

        deep_input = Flatten()(Concatenate()(embed_list))
        if len(continuous_input) > 0:
            if len(continuous_input) == 1:
                continuous_list = continuous_input[0]
            else:
                continuous_list = Concatenate()(continuous_input)

            deep_input = Concatenate()([deep_input, continuous_list])

        #if self.bias_dim > 0:
        #    bias_input_list = Concatenate()(bias_input)

        final_logit = linear_term  # TODO add bias term
        if len(self.hidden_size) > 0 and self.cross_num > 0:  # Deep & Cross
            deep_out = self.deep_layer(deep_input, self.hidden_size, self.activation, self.init_std, self.keep_prob,
                                       self.seed)
            cross_out = CrossLayer(self.cross_num)(deep_input)
            stack_out = Concatenate()([cross_out, deep_out])
            final_logit = Dense(1, activation=None)(stack_out)
        elif len(self.hidden_size) > 0:  # Only Deep
            deep_out = self.deep_layer(deep_input, self.hidden_size, self.activation, self.init_std, self.keep_prob,
                                       self.seed)
            final_logit = Dense(1, activation=None)(deep_out)
        elif self.cross_num > 0:  # Only Cross
            cross_out = CrossLayer(self.cross_num, init_std=self.init_std, l2_reg=self.l2_reg_cross)(deep_input)
            final_logit = Dense(1, activation=None)(cross_out)
        else:
            raise NotImplementedError
        final_logit = add([final_logit, linear_term], )
        #if self.bias_dim > 0:#TODO:添加偏差特征处理
        #    wide_out = self.wide_layer(bias_input_list, self.init_std, self.seed)
        #    final_logit = add([final_logit, wide_out])

        output = OutputLayer(self.final_activation)(final_logit)#Activation(self.final_activation)(final_logit)
        model = Model(inputs=cate_input + continuous_input , outputs=output)
        return model

    def get_input(self, ):
        cate_input = [Input(shape=(1,), name='cate_' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["sparse"])]
        continuous_input = [Input(shape=(1,), name='continuous' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["dense"])]
        #bias_input = [Input(shape=(1,), name='bias' + str(i)) for i in range(bias_term)]
        return cate_input, continuous_input#, bias_input

    def create_cate_embedding(self,  ):
        """

        :param field_dim:
        :param feature_dim:
        :param embedding_size:
        :param init_std:
        :return:
        """
        #if self.cate_feature_name is not None:
        #    cate_embed_name = ['embed_' + str(i) + '_' + feat for i, feat in enumerate(self.cate_feature_name)]
        #else:
        #    cate_embed_name = ['embed_' + str(i) for i in range(self.field_dim)]

        cate_embedding = [
            Embedding(self.feature_dim["sparse"][feat], min(self.embedding_size, int(self.embedding_factor * pow(self.feature_dim["sparse"][feat], 0.25))),
                      embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed), \
                      embeddings_regularizer=l2(self.l2_reg_embedding), name='cate_emb_' + str(i) + '-'+feat) for i,feat in
            enumerate(self.feature_dim["sparse"])]
        print("total embed size", sum(
            [min(self.embedding_size, int(self.embedding_factor * pow(self.feature_dim["sparse"][k], 0.25))) for k,v in self.feature_dim["sparse"].items()]))
        linear_embedding = [Embedding(self.feature_dim["sparse"][feat], 1,
                                      embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed),
                                      embeddings_regularizer=l2(self.l2_reg_embedding), name='linear_emb_' + str(i) + '-'+feat)
                            for i,feat in enumerate(self.feature_dim["sparse"])]
        return cate_embedding, linear_embedding

    def deep_layer(self, flatten_embeds_value, hidden_size, activation, init_std, keep_prob, seed):
        """

        :param flatten_embeds_value: batch_size * (feature_dim * feild_dim)
        :return:
        """
        deep_input = flatten_embeds_value
        for l in range(len(hidden_size)):
            fc = Dense(hidden_size[l], activation=None, use_bias=False, \
                       kernel_initializer=TruncatedNormal(mean=0.0, stddev=init_std, seed=seed) \
                       , kernel_regularizer=l2(self.l2_reg_deep))(deep_input)
            if self.use_bn:
                fc = BatchNormalization()(fc)
            fc = Activation('relu')(fc)

            # if l < len(hidden_size) - 1:
            fc = Dropout(1 - keep_prob)(fc)
            deep_input = fc
        deep_out = deep_input
        return deep_out

    #def wide_layer(self, bias_input, init_std, seed):
    #    wide_out = Dense(1, activation=None, )(bias_input)
    #    return wide_out


if __name__ == "__main__":
    model = DCN({"sparse":{"field_1":5,"field_2": 5,"field_3": 5,"field_4": 5},"dense":[]}, cross_num=2, hidden_size=[32, ], keep_prob=0.5,
                ).model
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
    print("DCN compile done")