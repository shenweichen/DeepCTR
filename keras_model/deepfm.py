# -*- coding:utf-8 -*-
"""
@author: shenweichen,wcshen1994@163.com
A keras implementation of DeepFM
Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
 (https://arxiv.org/abs/1703.04247)
"""
import keras.backend as K
from keras.layers import Input, Dense, Embedding, Concatenate, Activation, Lambda, Reshape, Flatten, Dropout, add, \
    subtract
from keras.models import Model
from keras.initializers import RandomNormal, TruncatedNormal,Zeros
from keras.layers import multiply
from keras.engine.topology import Layer
from keras.regularizers import l2



class FMLayer(Layer):
    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, concated_embeds_value):
        """

        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """

        temp_a = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), )(concated_embeds_value)

        temp_a = Lambda(lambda x: K.square(x))(temp_a)

        temp_b = multiply([concated_embeds_value, concated_embeds_value])

        temp_b = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(temp_b)

        cross_term = subtract([temp_a, temp_b])

        cross_term = Lambda(lambda x: 0.5 * K.sum(x, axis=2, keepdims=False))(cross_term)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


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

class DeepFM():
    def __init__(self, feature_dim_dict, embedding_size=4,
                 use_fm=True, hidden_size=[], l2_reg_linear=0.00002, l2_reg_fm=0.00002, l2_reg_deep=0.00002,
                 init_std=0.0001, seed=1024, keep_prob=0.5,final_activation='sigmoid',deep_input_mode='concat',
                 checkpoint_path=None, bias_feature_dict={'sparse':{},'dense':[]}):
        """
       :param feature_dim_dict:
       :param embedding_size:
       :param use_fm:
       :param hidden_size:
       :param l2_reg_linear:
       :param l2_reg_fm:
       :param l2_reg_deep:
       :param init_std:
       :param seed:
       :param keep_prob:
       :param final_activation:
       :param deep_input_mode:
       :param checkpoint_path:
       :param bias_feature_dict:
        """
        if not isinstance(feature_dim_dict,
                          dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
            raise ValueError(
                "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

        #self.field_dim = field_dim
        self.feature_dim = feature_dim_dict
        self.embedding_size = embedding_size
        self.use_fm = use_fm
        self.hidden_size = hidden_size
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_fm = l2_reg_fm
        self.l2_reg_deep = l2_reg_deep
        self.init_std = init_std
        self.seed = seed
        self.keep_prob = keep_prob
        self.activation = "relu"
        self.final_activation = final_activation
        self.bias_feature_dim = bias_feature_dict
        self.checkpoint_path = checkpoint_path
        self.deep_input_mode = deep_input_mode
        self.model = self.create_model()

    def get_model(self, ):
        return self.model

    def create_model(self, ):
        cate_input, continuous_input, bias_cate_input, bias_continuous_input = self.get_input ()
        cate_embedding, linear_embedding, bias_cate_embedding = self.create_cate_embedding()

        embed_list = [cate_embedding[i](cate_input[i]) for i in range(len(cate_input))]
        bias_embed_list = [bias_cate_embedding[i](bias_cate_input[i]) for i in range(len(bias_cate_input))]
        linear_term = add([linear_embedding[i](cate_input[i]) for i in range(len(cate_input))])


        fm_input = Concatenate(axis=1)(embed_list)
        deep_input = Flatten()(fm_input)

        if len(continuous_input) > 0:
            if len(continuous_input) == 1:
                continuous_list = continuous_input[0]
            else:
                continuous_list = Concatenate()(continuous_input)
            deep_input = Concatenate()([deep_input, continuous_list])


        fm_out = FMLayer()(fm_input)
        deep_out = self.deep_layer(deep_input, self.hidden_size, self.activation,  self.keep_prob,
                                   )
        if len(self.hidden_size) ==0 and self.use_fm == False:#only linear
            final_logit = linear_term,
        elif len(self.hidden_size) ==0 and self.use_fm == True:# linear + FM
            final_logit = add([linear_term,fm_out])
        elif len(self.hidden_size)>0 and self.use_fm == False:# linear +　Deep
            final_logit = add([linear_term,deep_out])
        elif len(self.hidden_size) >0 and self.use_fm == True:# linear + FM + Deep
            final_logit = add([linear_term,fm_out,deep_out])
        else:
            raise NotImplementedError
        self.bias_continuous_dim = len(self.bias_feature_dim["dense"])
        self.bias_cate_dim = len(self.bias_feature_dim["sparse"])
        if self.bias_continuous_dim>0:
            bias_continuous_out = Dense(1, activation=None, )(bias_continuous_input)
            final_logit = add([final_logit,bias_continuous_out])
        if self.bias_cate_dim> 0:
            if self.bias_cate_dim == 1:
                bias_cate_out = Lambda(lambda x: x)(bias_embed_list)  # add(bias_embed_list)
            if self.bias_cate_dim > 1:
                bias_cate_out = add(bias_embed_list)

            final_logit = add([final_logit, bias_cate_out])

        output = OutputLayer(self.final_activation)(final_logit)#Activation('sigmoid', name="final_activation", )(final_logit)
        output = Reshape([1])(output)
        model = Model(inputs=cate_input + continuous_input + bias_cate_input + bias_continuous_input, outputs=output)
        return model

    def get_input(self, ):
        cate_input = [Input(shape=(1,), name='cate' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["sparse"])]
        continuous_input = [Input(shape=(1,), name='continuous' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["dense"])]
        bias_cate_input = [Input(shape=(1,), name='bias_cate' + str(i)+'-'+feat) for i,feat in enumerate(self.bias_feature_dim["sparse"])]
        bias_continuous_input = [Input(shape=(1,), name='bias_continuous' + str(i)+'-'+feat) for i,feat in
                                 enumerate(self.bias_feature_dim["dense"])]

        return cate_input, continuous_input, bias_cate_input, bias_continuous_input,

    def create_cate_embedding(self,):
        """

        :param field_dim:
        :param feature_dim:
        :param embedding_size:
        :param init_std:
        :return:
        """

        cate_embedding = [Embedding(self.feature_dim["sparse"][feat], self.embedding_size,
                                    embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed), embeddings_regularizer=l2(self.l2_reg_fm),
                                    name='embed_cate' + str(i)+'-'+feat) for i,feat in
                          enumerate(self.feature_dim["sparse"])]
        linear_embedding = [Embedding(self.feature_dim["sparse"][feat], 1,
                                      embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed)
                                      , embeddings_regularizer=l2(self.l2_reg_linear), name='embed_linear' + str(i)) for
                            i,feat in enumerate(self.feature_dim["sparse"])]
        bias_cate_embedding = [Embedding(self.bias_feature_dim["sparse"][feat], 1,
                                         embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed)
                                         , embeddings_regularizer=l2(self.l2_reg_linear), name='embed_bias' + str(i))
                               for
                               i,feat in enumerate(self.bias_feature_dim["sparse"])]
        return cate_embedding, linear_embedding, bias_cate_embedding#,global_bias

    """
    def fm_layer(self, concated_embeds_value):


        :param concated_embeds_value:  batch_size * feature_dim * feild_dim
        :return:

        temp_a = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), )(concated_embeds_value)
        temp_b = Multiply()([concated_embeds_value, concated_embeds_value])
        temp_b = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(temp_b)
        cross_term = subtract([temp_a, temp_b])
        cross_term = Lambda(lambda x: 0.5 * K.sum(x, axis=2, keepdims=False))(cross_term)
        return cross_term
    """

    def deep_layer(self, flatten_embeds_value, hidden_size, activation,  keep_prob,):
        """

        :param flatten_embeds_value: batch_size * (feature_dim * feild_dim)
        :return:
        """
        deep_input = flatten_embeds_value
        for l in range(len(hidden_size)):
            fc = Dense(hidden_size[l], activation=activation, \
                       kernel_initializer=TruncatedNormal(mean=0.0, stddev=self.init_std, seed=self.seed), \
                       kernel_regularizer=l2(self.l2_reg_deep))(deep_input)

            # if l < len(hidden_size) - 1:
            fc = Dropout(1 - keep_prob)(fc)
            deep_input = fc

        deep_out = Dense(1, activation=None)(deep_input)
        return deep_out

if __name__ == "__main__":
    model = DeepFM({'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}, embedding_size=4, use_fm=True, hidden_size=[4, 4, 4], keep_prob=0.6,).model
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
    print("DeepFM compile done")