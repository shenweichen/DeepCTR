# -*- coding:utf-8 -*-
"""
@author: shenweichen,wcshen1994@163.com
A keras implementation of Neural Factorization Machines
Reference:
[1] Neural Factorization Machines for Sparse Predictive Analytics (https://arxiv.org/abs/1708.05027)
"""
import keras.backend as K
from keras.layers import Input,Dense,Embedding,Concatenate,Activation,Lambda,Reshape,Flatten,Dropout,add,subtract,multiply
from keras.models import  Model
from keras.initializers import RandomNormal,TruncatedNormal
from keras.engine.topology import Layer




class FMLayer(Layer):
    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, concated_embeds_value):
        """

        :param concated_embeds_value: None * field_size * embedding_size
        :return:
        """
        temp_a = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), )(concated_embeds_value)
        temp_b = multiply([concated_embeds_value, concated_embeds_value])
        temp_b = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(temp_b)
        cross_term = subtract([temp_a, temp_b])
        cross_term = Lambda(lambda x: 0.5 * K.sum(x, axis=2, keepdims=False))(cross_term)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

class BiInteractionLayer(Layer):
    def __init__(self, **kwargs):
        self.factor_size = None
        super(BiInteractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.factor_size = input_shape[2]
        super(BiInteractionLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, concated_embeds_value):
        """

        :param concated_embeds_value: None * field_size * embedding_size
        :return: None * embedding_size
        """
        temp_a = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), )(concated_embeds_value)
        temp_b = multiply([concated_embeds_value, concated_embeds_value])
        temp_b = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(temp_b)
        cross_term = subtract([temp_a, temp_b])
        cross_term = Lambda(lambda x: 0.5 * cross_term)(cross_term)
        cross_term = Reshape([self.factor_size])(cross_term)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.factor_size)


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

class NFM():
    def __init__(self,  feature_dim_dict, embedding_size=4,
                 hidden_size=[], l2_reg_w=0.00002, l2_reg_V=0.00002,
                 init_std=0.0001, seed=1024, keep_prob=0.5, final_activation='sigmoid',
                 checkpoint_path=None, bias_feature_dim={"sparse":{},"dense":[]}):
        if not isinstance(feature_dim_dict,
                          dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
            raise ValueError(
                "feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")

        #self.field_dim = field_dim
        self.feature_dim = feature_dim_dict
        #print(feature_dim_dict)
        self.embedding_size = embedding_size
        #self.use_cross = use_cross
        self.hidden_size = hidden_size
        self.l2_reg_w = 1.0
        self.l2_reg_V = l2_reg_V
        self.init_std = init_std
        self.seed = seed
        self.keep_prob = keep_prob
        self.activation = "relu"#activation
        self.final_activation = final_activation
        self.checkpoint_path = checkpoint_path
        #self.continuous_dim = continuous_dim
        self.bias_dim = bias_feature_dim
        self.model = self.create_model()

    def get_model(self, ):
        return self.model

    def create_model(self,  ):
        cate_input, continous_input, bias_input = self.get_input()
        cate_embedding, linear_embedding = self.create_cate_embedding()


        embed_list = [cate_embedding[i](cate_input[i]) for i in range(len(cate_input))]
        linear_term = add([linear_embedding[i](cate_input[i]) for i in range(len(cate_input))])
        linear_term = Reshape([1])(linear_term)

        fm_input = Concatenate(axis=1)(embed_list)


        bi_out = BiInteractionLayer()(fm_input)
        bi_out = Dropout(1-self.keep_prob)(bi_out)
        deep_out = self.deep_layer(bi_out, self.hidden_size, self.activation, self.init_std, self.keep_prob,
                                   self.seed)
        #K.bias_add()
        final_logit = linear_term  # TODO add bias term

        if len(self.hidden_size) > 0:
            final_logit = add([final_logit, deep_out])

        output = OutputLayer(self.final_activation)(final_logit)#Activation(self.final_activation, name="final_activation", )(final_logit)
        model = Model(inputs=cate_input + continous_input + bias_input, outputs=output)
        return model

    def get_input(self, ):
        cate_input = [Input(shape=(1,), name='cate_' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["sparse"])]
        continous_input = [Input(shape=(1,), name='continous' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["dense"])]
        bias_input = [Input(shape=(1,), name='bias' + str(i)+'-'+feat) for i,feat in enumerate(self.bias_dim["sparse"])]
        return cate_input, continous_input, bias_input

    def create_cate_embedding(self,):
        """

        :param field_dim:
        :param feature_dim:
        :param embedding_size:
        :param init_std:
        :return:
        """

        cate_embedding = [Embedding(self.feature_dim["sparse"][feat], self.embedding_size,
                                    embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed), name='cate_emb_' + str(i) + '-'+feat) for i,feat in
                          enumerate(self.feature_dim["sparse"])]
        linear_embedding = [Embedding(self.feature_dim["sparse"][feat], 1,
                                      embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed)
                                      , name='linear_emb_' + str(i) + '-'+feat) for i,feat in enumerate(self.feature_dim["sparse"])]
        return cate_embedding, linear_embedding


    def deep_layer(self, flatten_embeds_value, hidden_size, activation, init_std, keep_prob, seed):
        """

        :param flatten_embeds_value: batch_size * (feature_dim * feild_dim)
        :return:
        """
        deep_input = flatten_embeds_value
        for l in range(len(hidden_size)):
            fc = Dense(hidden_size[l], activation=activation, \
                       kernel_initializer=TruncatedNormal(mean=0.0, stddev=init_std, seed=seed))(deep_input)

            # if l < len(hidden_size) - 1:
            fc = Dropout(1 - keep_prob)(fc)
            deep_input = fc

        deep_out = Dense(1, activation=None)(deep_input)
        return deep_out

if __name__ == "__main__":
    model = NFM({"sparse":{"field1":4,"field2":4,"field3":4,"field4":4},"dense":[]}, embedding_size=3,  hidden_size=[4, 4, 4], keep_prob=0.6,).model
    model.compile('adam', 'binary_crossentropy', )
    print("nfm compile done")