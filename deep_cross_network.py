import tensorflow as tf
from .base import TFBaseModel

class DeepCrossNetwork(TFBaseModel):
    def __init__(self, field_dim, feature_dim,embedding_size=4,
                 cross_layer_num=1, hidden_size=[], use_batchnorm=True,deep_l2_reg=0.0,
                 init_std=0.01, seed=1024, keep_prob=0.5,
                 checkpoint_path=None,  ):
        super(DeepCrossNetwork, self).__init__(
            seed=seed, checkpoint_path=checkpoint_path)

        self.field_dim = field_dim
        self.feature_dim = feature_dim
        self.embedding_size = embedding_size

        self.deep_l2_reg = deep_l2_reg
        self.init_std = init_std

        self.seed = seed
        self.keep_prob = keep_prob
        self.cross_layer_num = cross_layer_num
        self.hidden_size = hidden_size

        #self.feature_list = feature_list
        #self.feature_count = feature_count
        self.use_batchnorm = use_batchnorm

        self._build_graph()

    def _get_optimizer_loss(self):
        return self.loss

    def _get_input_data(self, ):
        return self.X

    def _get_input_target(self, ):
        return self.Y
    def _get_output_target(self, ):
        return tf.sigmoid(self.logit)

    def _build_graph(self,):

        with self.graph.as_default():  # , tf.device('/cpu:0'):
            self._create_placeholders()

            self._create_variable()
            self._forward_pass()
            self._create_loss()


    def _create_placeholders(self, ):

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.field_dim], name='input_X')
        self.Y = tf.placeholder(tf.float32, shape=[None, ], name='input_Y')
        self.train_flag = tf.placeholder(tf.bool, name='train_flag')

    def _create_variable(self, ):

        self.b = tf.Variable(tf.constant(0.0), name='bias')
        # TODO:  self.init_std/ math.sqrt(float(dim))
        self.embedding_list = []
        self.total_size = self.field_dim * self.embedding_size
        # for feat in self.feature_list:
        #    cardinality = self.feature_count[feat]
        #    embedding_size = int(6 * np.power(cardinality, 0.25))
        #    self.total_size += embedding_size
        #    self.embedding_list.append(
        #        tf.Variable(tf.random_normal([cardinality, embedding_size], stddev=self.init_std, seed=self.seed),
        #                    name='embed' + feat))
        self.embeddings = tf.Variable(tf.random_normal(
            [self.feature_dim, self.embedding_size], stddev=self.init_std, seed=self.seed), name='cross_embed_weight')

        self.cross_layer_weight = [
            tf.Variable(tf.random_normal([self.total_size, 1], stddev=self.init_std, seed=self.seed)) for i in
            range(self.cross_layer_num)]
        self.cross_layer_bias = [
            tf.Variable(tf.random_normal([self.total_size, 1], stddev=self.init_std, seed=self.seed)) for i in
            range(self.cross_layer_num)]

    def f_cross_l(self, x_l, w_l, b_l):
        dot = tf.matmul(self._x_0, x_l, transpose_b=True)
        return tf.tensordot(dot, w_l, 1) + b_l

    def _forward_pass(self, ):

        def inverted_dropout(fc, keep_prob):
            return tf.divide(tf.nn.dropout(fc, keep_prob), keep_prob)
        

        with tf.name_scope("cross_network"):
            #embeds = []
            # for i in range(len(self.feature_list)):
            #    temp = tf.nn.embedding_lookup(self.embedding_list[i], self.X[:, i], )
            #    embeds.append(temp)
            #embeds = tf.concat(embeds, axis=1)
            embeds = tf.nn.embedding_lookup(
                self.embeddings, self.X, partition_strategy='div')

            self._x_0 = tf.reshape(embeds, (-1, self.total_size, 1))
            x_l = self._x_0
            for l in range(self.cross_layer_num):
                x_l = self.f_cross_l(
                    x_l, self.cross_layer_weight[l], self.cross_layer_bias[l]) + x_l

            cross_network_out = tf.reshape(x_l, (-1, self.total_size))

        with tf.name_scope('deep_network'):
            if len(self.hidden_size) > 0:
                fc_input = tf.reshape(
                    embeds, (-1, self.field_dim * self.embedding_size))

                for l in range(len(self.hidden_size)):
                    if self.use_batchnorm:
                        weight = tf.get_variable(name='deep_weight'+str(l),
                                              shape=[fc_input.get_shape().as_list()[1],self.hidden_size[l]],
                                            initializer=tf.random_normal_initializer(stddev=self.init_std,seed=self.seed))
                        #bias = tf.Variable(0.0,name='bias'+str(l))
                        H = tf.matmul(fc_input,weight)#,bias
                        H_hat = tf.layers.batch_normalization(H,training=self.train_flag)
                        fc = tf.nn.relu(H_hat)
                    else:
                        fc = tf.contrib.layers.fully_connected(fc_input, self.hidden_size[l],
                                                               activation_fn=tf.nn.relu,
                                                               weights_initializer=tf.truncated_normal_initializer(
                                                                   stddev=self.init_std),
                                                               weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                   self.deep_l2_reg))
                    if l < len(self.hidden_size) - 1:
                        fc = tf.cond(self.train_flag, lambda: inverted_dropout(
                            fc, self.keep_prob), lambda: fc)
                    fc_input = fc
                deep_network_out = fc_input

        with tf.name_scope("combination_output_layer"):
            x_stack = cross_network_out
            if len(self.hidden_size) > 0:
                x_stack = tf.concat([x_stack, deep_network_out], axis=1)

            self.logit = tf.contrib.layers.fully_connected(x_stack, 1, activation_fn=None,
                                                           weights_initializer=tf.truncated_normal_initializer(
                                                               stddev=self.init_std),
                                                           weights_regularizer=None)
            self.logit = tf.reshape(self.logit, (-1,))


    def _create_loss(self, ): 

        self.log_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.Y, logits=self.logit))  # total_loss
        # TODO: tf.summary.FileWriter
        #tf.summary.scalar('loss',self.log_loss)
        #self.merged = tf.summary.merge_all()
        #self.train_writer = tf.summary.FileWriter('../check/DCN/train',self.graph)
        #test_writer = tf.summary.FileWriter('../check/DCN/test')
        #https://www.tensorflow.org/get_started/summaries_and_tensorboard
        self.loss = self.log_loss  # + l2_reg_w_loss
if __name__ == '__main__':
    model = DeepCrossNetwork(2, 3)
    model.compile('adam',)
    print('DeepCrossNetwork test pass')