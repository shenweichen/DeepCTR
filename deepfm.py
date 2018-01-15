from __future__ import print_function

import tensorflow as tf

from .base import TFBaseModel
from .utils import tf_weighted_sigmoid_ce_with_logits


class DeepFM(TFBaseModel):
    def __init__(self, field_dim, feature_dim, embedding_size=4,
                 use_cross=True, hidden_size=[], l2_reg_w=1.0, l2_reg_V=1.0,
                 init_std=0.01, seed=1024, keep_prob=0.5,
                 checkpoint_path=None, continuous_dim=0,):

        super(DeepFM, self).__init__(
            seed=seed, checkpoint_path=checkpoint_path,)
        self.params = locals()
        self._build_graph()

    def _get_output_target(self, ):
        return tf.sigmoid(self.logit)

    def _get_optimizer_loss(self,):
        return self.loss

    def _build_graph(self, ):
        with self.graph.as_default():  # , tf.device('/cpu:0'):

            self._create_placeholders()
            self._create_variable()
            self._forward_pass()
            self._create_loss()

    def _get_input_data(self, ):
        if self.params['continuous_dim'] == 0:
            return self.placeholders['X']
        else:
            return self.placeholders['X'], self.placeholders['continuous_bias']

    def _get_input_target(self, ):
        return self.placeholders['Y']

    def _create_placeholders(self, ):
        self.placeholders = {}
        self.placeholders['X'] = tf.placeholder(
            tf.int32, shape=[None, self.params['field_dim']], name='input_X')
        self.placeholders['Y'] = tf.placeholder(
            tf.float32, shape=[None, ], name='input_Y')
        # addd
        self.placeholders['continuous_bias'] = tf.placeholder(
            tf.float32, shape=[None, self.params['continuous_dim']], name='input_bias')
        #-----
        self.train_flag = tf.placeholder(tf.bool, name='train_flag')

    def _create_variable(self, ):

        self.b = tf.get_variable(name='bias', initializer=tf.constant(0.0),)
        # TODO:  self.init_std/ math.sqrt(float(dim))
        self.embeddings = tf.get_variable(name='cross_weight', initializer=tf.random_normal([self.params['feature_dim'], self.params['embedding_size']],
                                                                                            stddev=self.params['init_std'], seed=self.seed),
                                          )
        self.single_embedding = tf.get_variable(name='linear_weight',
                                                initializer=tf.zeros((self.params['feature_dim'], 1), ), )
        # TODO: normal
        self._l2_reg_w = tf.constant(
            self.params['l2_reg_w'], shape=(1,), name='l2_reg_w')
        self._l2_reg_V = tf.constant(
            self.params['l2_reg_V'], shape=(1,), name='l2_reg_V')

    def _forward_pass(self, ):

        def inverted_dropout(fc, keep_prob):
            return tf.divide(tf.nn.dropout(fc, keep_prob), keep_prob)

        with tf.name_scope("linear_term"):
            w = tf.nn.embedding_lookup(
                self.single_embedding, self.placeholders['X'], partition_strategy='div')
            linear_term = tf.reduce_sum(w, axis=(1, 2))

        with tf.name_scope("cross_term"):
            embeds = tf.nn.embedding_lookup(
                self.embeddings, self.placeholders['X'], partition_strategy='div')
            temp_a = tf.reduce_sum(
                tf.matmul(embeds, embeds, transpose_b=True), axis=(1, 2))
            temp_b = tf.reduce_sum(tf.square(embeds), axis=(1, 2))
            cross_term = 0.5 * (temp_a - temp_b)
        with tf.name_scope('deep_network'):
            if len(self.params['hidden_size']) > 0:
                fc_input = tf.reshape(
                    embeds, (-1, self.params['field_dim'] * self.params['embedding_size']))
                if self.params['continuous_dim'] > 0:
                    fc_input = tf.concat(
                        [fc_input, self.placeholders['continuous_bias']], axis=1)
                for l in range(len(self.params['hidden_size'])):
                    fc = tf.contrib.layers.fully_connected(fc_input, self.params['hidden_size'][l],
                                                           activation_fn=tf.nn.relu,
                                                           weights_initializer=tf.truncated_normal_initializer(
                                                               stddev=self.params['init_std']),
                                                           )
                    if l < len(self.params['hidden_size']) - 1:
                        fc = tf.cond(self.train_flag, lambda: inverted_dropout(
                            fc, self.params['keep_prob']), lambda: fc)

                    fc_input = fc

                nn_logit = tf.contrib.layers.fully_connected(fc_input, 1, activation_fn=None,
                                                             weights_initializer=tf.truncated_normal_initializer(
                                                                 stddev=self.params['init_std']),
                                                             )
                nn_logit = tf.reshape(nn_logit, (-1,))

        with tf.name_scope("model_logit"):
            self.logit = self.b + linear_term
            if self.params['use_cross']:
                self.logit += cross_term
            if len(self.params['hidden_size']) > 0:
                self.logit += nn_logit

    def _create_loss(self, ):
        l2_reg_w_loss = 1 * self._l2_reg_w * \
            tf.nn.l2_loss(self.single_embedding)
        l2_reg_V_loss = 1 * self._l2_reg_V * tf.nn.l2_loss(self.embeddings)
        # tf.contrib.layers.l2_regularizer()

        # self.log_loss = tf.reduce_sum(tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(
        #    labels=self.placeholders['Y'], logits=self.logit),self.sample_weight))  # weighted total_loss
        self.log_loss = tf.reduce_sum(tf_weighted_sigmoid_ce_with_logits(
            self.placeholders['Y'], self.logit, self.sample_weight))

        self.loss = self.log_loss  # + l2_reg_w_loss
        if self.params['use_cross']:
            # self.loss += l2_reg_V_loss
            pass


if __name__ == '__main__':
    model = DeepFM(2, 3)
    model.compile('ftrl',)
    print('DeepFM test pass')
