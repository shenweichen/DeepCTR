# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Deep Interest Network for Click-Through Rate Prediction (https://arxiv.org/pdf/1706.06978.pdf)
"""

from tensorflow.python.keras.layers import Input, Dense, Embedding, Concatenate, Reshape,Lambda,Layer,multiply, Permute
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2
import  tensorflow as tf
from deepctr.contrib.utils import QAAttGRUCell, VecAttGRUCell
from deepctr.contrib.rnn import dynamic_rnn
from ..layers.core import MLP  #,NaiveActivationUnit
from ..layers.sequence import AttentionSequencePoolingLayer
from ..input_embedding import create_singlefeat_inputdict
from ..layers.activation import Dice
from ..utils import check_feature_config_dict



class DynamicGRU(Layer):
    def __init__(self, num_units=None,type='GRU',return_sequence=True, name="gru",**kwargs):

        self.num_units = num_units
        self.return_sequence = return_sequence
        #self.name = name
        self.type  = type
        super(DynamicGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]

        super(DynamicGRU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input_list):
        """
        :param concated_embeds_value: None * field_size * embedding_size
        :return: None*1
        """
        if self.type == "GRU" or self.type == "AIGRU":
            rnn_input, sequence_length = input_list
            att_score = None
        else:
            rnn_input, sequence_length,att_score = input_list

        if self.type == "AGRU":
            gru_cell = QAAttGRUCell(self.num_units)
        elif self.type == "AUGRU":
            gru_cell = VecAttGRUCell(self.num_units)
        else:
            gru_cell = tf.nn.rnn_cell.GRUCell(self.num_units)


        rnn_output,hidden_state =   dynamic_rnn(gru_cell, inputs=rnn_input,att_scores=att_score,
                                                            sequence_length=tf.squeeze(sequence_length,
                                                                                      ), dtype=tf.float32,scope=self.name)
        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(hidden_state,axis=1)
    def compute_output_shape(self, input_shape):
        rnn_input_shape =  input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return (None,1,rnn_input_shape[2])




def get_input(feature_dim_dict, seq_feature_list, seq_max_len):

    sparse_input,dense_input = create_singlefeat_inputdict(feature_dim_dict)
    user_behavior_input = {feat: Input(shape=(seq_max_len,), name='seq_' + str(i) + '-' + feat) for i, feat in
                           enumerate(seq_feature_list)}

    user_behavior_length = Input(shape=(1,), name='seq_length')

    return sparse_input,dense_input, user_behavior_input, user_behavior_length

def auxiliary_loss( h_states, click_seq, noclick_seq, mask, stag = None):
    """

    :param h_states:
    :param click_seq:
    :param noclick_seq: #[B,T-1,E]
    :param mask:#[B,1]
    :param stag:
    :return:
    """
    hist_len, _ = click_seq.get_shape().as_list()[1:]
    mask = tf.sequence_mask(mask, hist_len)
    mask = mask[:,0,:]

    mask = tf.cast(mask, tf.float32)

    click_input_ = tf.concat([h_states, click_seq], -1)

    noclick_input_ = tf.concat([h_states, noclick_seq], -1)

    click_prop_ = auxiliary_net(click_input_, stag = stag)[:, :, 0]

    noclick_prop_ = auxiliary_net(noclick_input_, stag = stag)[:, :, 0]#[B,T-1]

    click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask

    noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask

    loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)

    return loss_
def auxiliary_net(in_, stag='auxiliary_net'):

    bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)

    dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)

    dnn1 = tf.nn.sigmoid(dnn1)

    dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)

    dnn2 = tf.nn.sigmoid(dnn2)

    dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)

    y_hat = tf.nn.sigmoid(dnn3)#tf.nn.softmax(dnn3) + 0.00000001

    return y_hat

def interest_evolution(concat_behavior, deep_input_item, user_behavior_length, gru_type="GRU", use_neg=False,
                       neg_concat_behavior=None,embedding_size=8,att_hidden_size=(64,16),att_activation='sigmoid',att_weight_normalization=False,):
    if gru_type not in ["GRU", "AIGRU", "AGRU", "AUGRU"]:
        raise ValueError("gru_type error ")
    aux_loss_1 = None
    # if gru_type == "GRUv2":
    #     return DynamicGRU(embedding_size*2, return_sequence=False, name="gru1") \
    #     ([concat_behavior, user_behavior_length]),aux_loss_1

    rnn_outputs = DynamicGRU(embedding_size*2, return_sequence=True, name="gru1") \
        ([concat_behavior, user_behavior_length])
    #rnn_outputs = concat_behavior
    #hist_len = concat_behavior.get_shape()[1]

    #key_masks = Lambda(lambda x:tf.sequence_mask(x,hist_len))(user_behavior_length)
    #print(user_behavior_length, hist_len,key_masks)
    if gru_type == "AUGRU" and use_neg:
        aux_loss_1 = auxiliary_loss(rnn_outputs[:, :-1, :], concat_behavior[:, 1:, :],

                                              neg_concat_behavior[:, 1:, :],

                                              tf.subtract(user_behavior_length, 1), stag="gru")  # [:, 1:]

    if gru_type == "GRU":
        rnn_outputs2 = DynamicGRU(embedding_size*2, return_sequence=True, name="gru2") \
            ([rnn_outputs, user_behavior_length])
        #rnn_outputs2 = rnn_outputs
        attention_score = AttentionSequencePoolingLayer(hidden_size=att_hidden_size,activation=att_activation,weight_normalization=att_weight_normalization,return_score=True)([deep_input_item,rnn_outputs2,user_behavior_length])
        outputs = Lambda(lambda x: tf.matmul(x[0],x[1]))([attention_score, rnn_outputs2])
        hist = outputs
        #hist = SequencePoolingLayer(300)([rnn_outputs,user_behavior_length])

    else:

        scores = AttentionSequencePoolingLayer(hidden_size=att_hidden_size,activation=att_activation,weight_normalization=att_weight_normalization,return_score=True)([deep_input_item,rnn_outputs,user_behavior_length])

        if gru_type == "AIGRU":# or gru_type == "AIGRUv2":
            #print(rnn_outputs,scores,Permute([2,1])(scores))
            if gru_type == "AIGRU":
                hist = multiply([rnn_outputs, Permute([2,1])(scores)])
            else:
                scores = AttentionSequencePoolingLayer(hidden_size=att_hidden_size,activation=att_activation, weight_normalization=att_weight_normalization,
                                                       return_score=True)(
                    [deep_input_item, concat_behavior, user_behavior_length])
                hist = multiply([concat_behavior,Permute([2,1])(scores)])
            #print(hist)
            final_state2 = DynamicGRU(embedding_size*2, type="GRU", return_sequence=False, name='gru2')(
                [hist, user_behavior_length])
        else:
            final_state2 = DynamicGRU(embedding_size * 2, type=gru_type, return_sequence=False,
                                      name='gru2')([rnn_outputs, user_behavior_length, Permute([2,1])(scores)])

        # rnn_outputs2, final_state2 = GRU(self.embedding_size * 2, return_sequences=False, return_state=True,
        #                                 name="gru2",
        #                                 kernel_initializer="he_normal")(hist)
        # hist = Lambda(lambda x: tf.expand_dims(x, axis=1))(final_state2)
        hist = final_state2
    return hist,aux_loss_1


def DIEN(feature_dim_dict, seq_feature_list, embedding_size=8, hist_len_max=16,
         gru_type="GRU",use_negsampling=False,alpha=1, use_bn=False, hidden_size=(200, 80), activation='sigmoid', att_hidden_size=[64, 16], att_activation=Dice, att_weight_normalization=True,
        l2_reg_deep=0, l2_reg_embedding=1e-5, final_activation='sigmoid', keep_prob=1, init_std=0.0001, seed=1024, ):
    """Instantiates the Deep Interest Network architecture.

    :param feature_dim_dict: dict,to indicate sparse field (**now only support sparse feature**)like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':[]}
    :param seq_feature_list: list,to indicate  sequence sparse field (**now only support sparse feature**),must be a subset of ``feature_dim_dict["sparse"]``
    :param embedding_size: positive integer,sparse feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param use_din: bool, whether use din pooling or not.If set to ``False``,use **sum pooling**
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_deep: float. L2 regularizer strength applied to deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """
    check_feature_config_dict(feature_dim_dict)
    sparse_input,dense_input, user_behavior_input,user_behavior_length = get_input(
        feature_dim_dict, seq_feature_list, hist_len_max)
    sparse_embedding_dict = {feat.name: Embedding(feat.dimension, embedding_size,
                                             embeddings_initializer=RandomNormal(
                                                 mean=0.0, stddev=init_std, seed=seed),
                                             embeddings_regularizer=l2(
                                                 l2_reg_embedding),
                                             name='sparse_emb_' + str(i) + '-' + feat.name) for i, feat in
                             enumerate(feature_dim_dict["sparse"])}
    query_emb_list = [sparse_embedding_dict[feat](
        sparse_input[feat]) for feat in seq_feature_list]
    keys_emb_list = [sparse_embedding_dict[feat](
        user_behavior_input[feat]) for feat in seq_feature_list]
    deep_input_emb_list = [sparse_embedding_dict[feat.name](
        sparse_input[feat.name]) for feat in feature_dim_dict["sparse"]]

    query_emb = Concatenate()(query_emb_list) if len(
        query_emb_list) > 1 else query_emb_list[0]
    keys_emb = Concatenate()(keys_emb_list) if len(
        keys_emb_list) > 1 else keys_emb_list[0]
    deep_input_emb = Concatenate()(deep_input_emb_list) if len(
        deep_input_emb_list) > 1 else deep_input_emb_list[0]

    if use_negsampling:
        neg_user_behavior_input = {feat: Input(shape=(hist_len_max,), name='neg_seq_' + str(i) + '-' + feat) for i, feat in
                               enumerate(seq_feature_list)}
        neg_uiseq_embed_list = [sparse_embedding_dict[feat](
            neg_user_behavior_input[feat]) for feat in seq_feature_list]
        neg_concat_behavior = Concatenate()(neg_uiseq_embed_list) if len(neg_uiseq_embed_list) > 1 else \
        neg_uiseq_embed_list[0]
    else:
        neg_concat_behavior = None

    hist,aux_loss_1 = interest_evolution(keys_emb, query_emb, user_behavior_length, gru_type=gru_type,
                                   use_neg=use_negsampling, neg_concat_behavior=neg_concat_behavior,embedding_size=embedding_size,att_hidden_size=att_hidden_size,att_activation=att_activation,att_weight_normalization=att_weight_normalization,)

    deep_input_emb = Concatenate()([deep_input_emb, hist])

    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
    if len(dense_input) > 0:
        deep_input_emb = Concatenate()([deep_input_emb]+list(dense_input.values()))

    output = MLP(hidden_size, activation, l2_reg_deep,
                 keep_prob, use_bn, seed)(deep_input_emb)
    output = Dense(1, final_activation)(output)
    output = Reshape([1])(output)
    model_input_list = list(sparse_input.values(
    ))+list(dense_input.values())+list(user_behavior_input.values())
    if use_negsampling:
        model_input_list += list(neg_user_behavior_input.values())

    model_input_list +=  [user_behavior_length]

    model = Model(inputs=model_input_list, outputs=output)

    if use_negsampling:
        model.add_loss(alpha * aux_loss_1)
    return model


