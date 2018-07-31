# -*- coding:utf-8 -*-
"""
@author: shenweichen,wcshen1994@163.com
A keras implementation of Attentional Factorization Machines
Reference:
[1] Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks
(https://arxiv.org/abs/1708.04617)
"""
import keras.backend as K
from keras.layers import Input, Dense, Embedding, Concatenate, Activation, Lambda, Reshape, Flatten, Dropout, subtract, Dot, concatenate, add
from keras.models import Model
from keras.initializers import RandomNormal, TruncatedNormal
from keras.layers import multiply
from keras.engine.topology import Layer
from keras.regularizers import l2

try:
    from tf_model.utils import FMLayer, IdentityLayer, AFMLayer, ActivationWeightedSumLayer, SoftmaxWeightedSumLayer
except:
    pass


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


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



class AFMLayer(Layer):
    def __init__(self, attention_factor=4,pair_interaction_keep_prob=1.0, **kwargs):
        self.attention_factor = attention_factor
        self.keep_prob = pair_interaction_keep_prob
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim = input_shape[0][-1]
        self.attention_W = self.add_weight(shape=(input_dim,self.attention_factor),initializer='glorot_uniform',name="attention_W")
        self.attention_b = self.add_weight(shape=(self.attention_factor,),initializer='zeros',name="attention_b")
        self.projection_h = self.add_weight(shape=(self.attention_factor,1),initializer='glorot_uniform',name="projection_h")
        self.projection_p = self.add_weight(shape=(input_dim,1),initializer='glorot_uniform',name="projection_p")
        super(AFMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, embeds_vec_list,):
        """

        :param embeds_vec_list:  list of len(field_size) ,elem: None*1*embedding_size
        :return: None * 1
        """
        assert isinstance(embeds_vec_list,list),"input must be list"
        # TODO: biinteraction
        bi_interaction = []
        for i in range(len(embeds_vec_list)):
            for j in range(i + 1, len(embeds_vec_list)):
                prod_merged = multiply([embeds_vec_list[i], embeds_vec_list[j]])
                bi_interaction.append(prod_merged)
        bi_interaction = concatenate(bi_interaction, axis=1)
        #bi_interaction = Dropout(1-self.keep_prob)(bi_interaction)
        attention_temp = Activation(activation='relu')(K.bias_add(K.dot(bi_interaction,self.attention_W),self.attention_b,data_format='channels_last'))
        #Dense(self.attention_factor, activation='relu')(bi_interaction)
        #attention_a = Dense(1, activation=None)(attention_temp)
        #attention_weight = Activation(activation='softmax')(attention_a)
        attention_weight = Activation(activation='softmax')(K.dot(attention_temp,self.projection_h))
        #print(attention_temp,self.projection_h,attention_weight)
        attention_output = Dot(axes=1)([attention_weight, bi_interaction])
        #afm_out = Reshape([1])(Dense(1, )(attention_output))
        afm_out = Reshape([1])(K.dot(attention_output,self.projection_p))
        return afm_out

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


class AFM():
    """
    when set use_attention=False,deep_input_mode='concat' it is deepfm
    """
    def __init__(self, feature_dim_dict, embedding_size=4,
                 use_fm=True, use_attention=True,attention_factor=4,deep_input_mode='inner_product', hidden_size=[32],
                 l2_reg_linear=0.00002, l2_reg_fm=0.00002, l2_reg_deep=0.00002,
                 init_std=0.0001, seed=1024, keep_prob=0.5,deep_input_keep_prob=0.5,pair_out_keep_prob=0.5, final_activation='sigmoid',
                 checkpoint_path=None, bias_feature_dim = {'sparse':{},'dense':[]}): #continuous_dim=0, bias_field_dim=0, bias_feature_dim=[], bias_continuous_dim=0, \
               # ):
        """
# feature_dim={"cate":{'user_id':4,'ad_id':3,'gender':2},"continuous":["age",]}
        :param field_dim:
        :param feature_dim: dict  feature_name:feature_dim
        :param embedding_size:
        :param use_fm:
        :param use_merged_pooling:
        :param use_afm:
        :param deep_input_mode: str ["inner_product","elementwise_product","concat"]
        :param hidden_size:
        :param l2_reg_linear:
        :param l2_reg_fm:
        :param l2_reg_deep:
        :param init_std:
        :param seed:
        :param keep_prob:
        :param activation:
        :param checkpoint_path:
        :param continuous_dim:
        :param bias_field_dim:
        :param bias_feature_dim:
        :param bias_continuous_dim:
        :param variable_feature_dim:
        :param variable_feature_input_length:
        :param variable_feature_embedding_size:
        """
        if not isinstance(feature_dim_dict,dict) or "sparse" not in feature_dim_dict or "dense" not in feature_dim_dict:
            raise ValueError("feature_dim must be a dict like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_5',]}")
        #self.field_dim = field_dim
        self.feature_dim = feature_dim_dict
        self.embedding_size = embedding_size
        self.use_cross = use_fm
        self.hidden_size = hidden_size
        self.l2_reg_linear = l2_reg_linear
        self.l2_reg_fm = l2_reg_fm
        self.l2_reg_deep = l2_reg_deep
        self.init_std = init_std
        self.seed = seed
        self.keep_prob = keep_prob
        self.use_fm = use_fm
        self.use_attention= use_attention
        self.attention_factor = attention_factor
        self.activation = "relu"#activation
        self.deep_input_mode = deep_input_mode
        self.final_activation = final_activation
        self.checkpoint_path = checkpoint_path
        self.pair_out_keep_prob = pair_out_keep_prob
        #self.continuous_dim = continuous_dim
        #self.bias_continuous_dim = bias_continuous_dim
        #self.bias_cate_dim = bias_field_dim
        self.bias_feature_dim = bias_feature_dim
        #self.variable_feature_dim = variable_feature_dim
        #self.use_variable_inner_prod = True
        #self.variable_feature_input_length = variable_feature_input_length
        self.deep_input_keep_prob = deep_input_keep_prob
        self.model = self.create_model()

    def get_model(self, ):
        return self.model

    def create_model(self, ):
        cate_input, continuous_input, bias_cate_input, bias_continuous_input = self.get_input()
        cate_embedding, linear_embedding, bias_cate_embedding = self.create_cate_embedding()

        embed_list = [cate_embedding[i](cate_input[i]) for i in range(len(self.feature_dim["sparse"]))]


        # TODO: deep part 输入准备
        vec_elemwise_prod_list = []
        vec_inner_prod_list = []
        # 下面是普通的交叉
        for i in range(len(self.feature_dim["sparse"])):
            for j in range(i + 1, len(self.feature_dim["sparse"])):
                prod = multiply([embed_list[i], embed_list[j]])
                vec_elemwise_prod_list.append(prod)
                vec_inner_prod_list.append(Lambda(lambda x: K.sum(x, axis=2, keepdims=False))(prod))


        bias_embed_list = [bias_cate_embedding[i](bias_cate_input[i]) for i in range(len(self.bias_feature_dim["sparse"]))]
        linear_term = add([linear_embedding[i](cate_input[i]) for i in range(len(self.feature_dim["sparse"]))])
        # linear_term = Reshape([1])(linear_term)

        fm_input = Concatenate(axis=1)(embed_list)

        if self.use_attention:
            fm_out = AFMLayer(self.attention_factor,self.pair_out_keep_prob)(embed_list)
        else:
            fm_out = FMLayer()(fm_input)

        if self.deep_input_mode == "inner_product":
            deep_input = Concatenate(axis=-1)(vec_inner_prod_list)#Flatten()(fm_input)
        elif self.deep_input_mode == "concat":
            deep_input = Flatten()(fm_input)
        elif self.deep_input_mode == "elementwise_product":
            deep_input = Flatten()(Concatenate(axis=-1)(vec_elemwise_prod_list))
        else:
            raise  ValueError("deep_input_mode invalid")
        #print(deep_input)
        if len(continuous_input) > 0:
            if len(continuous_input) == 1:
                continuous_list = continuous_input[0]
            else:
                continuous_list = Concatenate()(continuous_input)
            deep_input = Concatenate()([deep_input, continuous_list])




        deep_out = self.deep_layer(deep_input, self.hidden_size, self.activation, self.init_std, self.keep_prob,
                                   self.seed)
        if len(self.hidden_size) == 0 and self.use_fm == False:  # only linear
            final_logit = linear_term
        elif len(self.hidden_size) == 0 and self.use_fm == True:  # only FM
            final_logit = add([linear_term, fm_out])
        elif len(self.hidden_size) > 0 and self.use_fm == False:  # linear +　Deep
            final_logit = add([linear_term, deep_out])
        elif len(self.hidden_size) > 0 and self.use_fm == True:  # Deep FM
            final_logit = add([linear_term, fm_out, deep_out])
        else:
            raise NotImplementedError
        bias_cate_dim = len(self.bias_feature_dim["sparse"])
        bias_continuous_dim = len(self.bias_feature_dim["dense"])
        if bias_cate_dim + bias_continuous_dim > 0:
            if bias_continuous_dim > 0:  # TODO:添加连续偏差特征
                bias_continuous_out = Dense(1, activation=None, )(bias_continuous_input)
            if bias_cate_dim == 1:
                bias_cate_out = Lambda(lambda x: x)(bias_embed_list)  # add(bias_embed_list)
            if bias_cate_dim > 1:
                bias_cate_out = add(bias_embed_list)
            wide_out = bias_cate_out
            final_logit = add([final_logit, wide_out])
        #print(final_logit)
        #output = Activation(self.final_activation, name="final_activation", )(final_logit)
        output = OutputLayer(self.final_activation)(final_logit)
        #print(output,output2)
        output = Reshape([1])(output)
        model = Model(inputs=cate_input + continuous_input + bias_cate_input + bias_continuous_input,
                      outputs=output)
        return model

    def get_input(self, ):
        cate_input = [Input(shape=(1,), name='cate_' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["sparse"])]
        continuous_input = [Input(shape=(1,), name='continuous' + str(i)+'-'+feat) for i,feat in enumerate(self.feature_dim["dense"])]
        bias_cate_input = [Input(shape=(1,), name='bias_cate' + str(i)+'-'+feat) for i,feat in enumerate(self.bias_feature_dim["sparse"])]
        bias_continuous_input = [Input(shape=(1,), name='bias_continuous' + str(i)+'-'+feat) for i,feat in
                                 enumerate(self.bias_feature_dim["dense"])]


        return cate_input, continuous_input, bias_cate_input, bias_continuous_input

    def create_cate_embedding(self, ):
        """

        :param field_dim:
        :param feature_dim:
        :param embedding_size:
        :param init_std:
        :return:
        """

        cate_embedding = [Embedding(self.feature_dim["sparse"][feat], self.embedding_size,
                                    embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed), embeddings_regularizer=l2(self.l2_reg_fm),
                                    name='cate_emb_' + str(i) + '-'+feat) for i,feat in
                          enumerate(self.feature_dim["sparse"])]
        linear_embedding = [Embedding(self.feature_dim["sparse"][feat], 1,
                                      embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed)
                                      , embeddings_regularizer=l2(self.l2_reg_linear), name='linear_emb_' + str(i)+'-'+feat) for
                            i,feat in enumerate(self.feature_dim["sparse"])]
        bias_cate_embedding = [Embedding(self.bias_feature_dim["sparse"][feat], 1,
                                         embeddings_initializer=RandomNormal(mean=0.0, stddev=self.init_std, seed=self.seed)
                                         , embeddings_regularizer=l2(self.l2_reg_linear), name='embed_bias' + str(i)+'-'+feat)
                               for
                               i,feat in enumerate(self.bias_feature_dim["sparse"])]
        return cate_embedding, linear_embedding, bias_cate_embedding


    def deep_layer(self, flatten_embeds_value, hidden_size, activation, init_std, keep_prob, seed):
        """

        :param flatten_embeds_value: batch_size * (feature_dim * feild_dim)
        :return:
        """
        deep_input = flatten_embeds_value
        deep_input = Dropout(1-self.deep_input_keep_prob)(deep_input)
        for l in range(len(hidden_size)):
            fc = Dense(hidden_size[l], activation=activation, \
                       kernel_initializer=TruncatedNormal(mean=0.0, stddev=init_std, seed=seed), \
                       kernel_regularizer=l2(self.l2_reg_deep))(deep_input)

            # if l < len(hidden_size) - 1:
            fc = Dropout(1 - keep_prob)(fc)
            deep_input = fc

        deep_out = Dense(1, activation=None)(deep_input)
        return deep_out

    def wide_layer(self, bias_continuous_input, bias_cate_input, init_std, seed):
        wide_out = Dense(1, activation=None, )(bias_continuous_input)
        wide_out = add([wide_out, bias_cate_input])
        return wide_out


if __name__ == "__main__":

    model = AFM({"sparse":{"field1":4,"field2":4, "field3":4,"field4": 4},'dense':["field_5"]}, embedding_size=4, use_fm=True, hidden_size=[], keep_prob=0.5,).model
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])
    print("AFM compile done")#