import time
from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.model_selection import train_test_split

from .utils import sigmoid_cross_entropy_with_probs 


class TFBaseModel(metaclass=ABCMeta):
    def __init__(self, seed=1024, checkpoint_path=None,):
        self.seed = seed

        if checkpoint_path and checkpoint_path.count('/') < 2:
            raise ValueError('checkpoint_path must be dir/model_name format')
        self.checkpoint_path = checkpoint_path
        self.train_flag = True
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)
        

    @abstractmethod
    def _get_input_data(self, ):
        raise NotImplementedError
    @abstractmethod
    def _get_input_target(self, ):
        raise NotImplementedError

    @abstractmethod
    def _get_output_target(self, ):
        raise NotImplementedError
    @ abstractmethod
    def _get_optimizer_loss(self,):
        """
        return the loss tensor that the optimizer wants to minimize
        :return:
        """
    @abstractmethod
    def _build_graph(self):
        """
        该方法必须在子类的初始化方法末尾被调用
        子类的方法在默认图中构建计算图
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            tf.set_random_seed(self.seed)
            #构建计算图
            #...
            #...
        """
        raise NotImplementedError

    def compile(self, optimizer='sgd', loss='logloss', metrics=None, loss_weights=None, sample_weight_mode=None):
        """
        compile the model with optimizer and loss function
        :param optimizer:str or predefined optimizer in tensorflow
        ['sgd','adam','adagrad','rmsprop','moment','ftrl']
        :param loss: str 
        :param metrics: str ['logloss','mse','mean_squared_error','logloss_with_logits']
        :param loss_weights:
        :param sample_weight_mode:
        :return:
        """
        with self.graph.as_default():# , tf.device('/cpu:0'):
            #根据指定的优化器和损失函数初始化
            self.metric_list = self._create_metrics(metrics)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#for the use of BN
            with tf.control_dependencies(update_ops):#for the use of BN
                self.optimizer = self._create_optimizer(optimizer).minimize(self.loss, global_step=self.global_step)
            #执行初始化操作
            init = tf.global_variables_initializer()  # init
            self.sess.run(init)  # sess defined in scope

    def _create_metrics(self,metric):
        if metric is None:#若不指定，则以训练时的损失函数作为度量
            return [self._get_optimizer_loss()]

        if metric not in ['logloss','mse','mean_squared_error','logloss_with_logits']:
            raise ValueError('invalid param metrics')
        # TODO:添加更多度量函数和函数作为参数
        metrics_list = []

        if metric == 'logloss':
            metrics_list.append(tf.reduce_sum(sigmoid_cross_entropy_with_probs(
            labels=self._get_input_target(), probs=self._get_output_target())))
        elif metric=='mse' or metric == 'mean_squared_error':
            metrics_list.append(tf.reduce_sum(tf.squared_difference(self._get_input_target(),self._get_output_target())))
        elif metric=='logloss_with_logits':
            metrics_list.append(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._get_input_target(), logits=self.logit)))
        return metrics_list

    def _create_optimizer(self,optimizer='sgd'):
        """

        :param optimizer: str of optimizer or predefined optimizer in tensorflow
        :return: optimizer object
        """

        optimizer_dict = {'sgd':tf.train.GradientDescentOptimizer(0.01),
                          'adam':tf.train.AdamOptimizer(0.001),
                          'adagrad':tf.train.AdagradOptimizer(0.01),
                          #'adagradda':tf.train.AdagradDAOptimizer(),
                          'rmsprop':tf.train.RMSPropOptimizer(0.001),
                          'moment':tf.train.MomentumOptimizer(0.01,0.9),
                          'ftrl':tf.train.FtrlOptimizer(0.01)
                          #tf.train.ProximalAdagradOptimizer#padagrad
                           #tf.train.ProximalGradientDescentOptimizer#pgd
        }
        if isinstance(optimizer,str):
            if optimizer in optimizer_dict.keys():
                return optimizer_dict[optimizer]
            else:
                raise ValueError('invalid optimizer name')
        elif isinstance(optimizer,tf.train.Optimizer):
            return  optimizer
        else:
            raise ValueError('invalid parm for optimizer')

   

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path + '.ckpt', self.global_step)

    def load_model(self, meta_graph_path, ckpt_dir=None, ckpt_path=None):
        """
        :ckpt_dir 最新的检查点
        :ckpt_path 指定检查点
        """
        if ckpt_dir is None and ckpt_path is None:
            raise ValueError('Must specify ckpt_dir or ckpt_path')
        restore_saver = tf.train.import_meta_graph(meta_graph_path, )
        if ckpt_path is None:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        restore_saver.restore(self.sess, ckpt_path)

    def train_on_batch(self, x, y ):  # fit a batch
        feed_dict_ = {self._get_input_data(): x,
                      self._get_input_target(): y, self.train_flag: True}
        self.sess.run((self._get_optimizer_loss(), self.optimizer), feed_dict=feed_dict_)

    def fit(self,x, y, batch_size=1024, epochs=50, validation_split = 0.0, validation_data=None,
            val_size=2 ** 18, shuffle=True,initial_epoch=0,min_display=50,max_iter=-1):
        
        if validation_split < 0 or validation_split >= 1:
            raise ValueError("validation_split must be a float number >= 0 and < 1")
        
        n_samples = x.shape[0]
        iters = (n_samples - 1) // batch_size + 1
        self.tr_loss_list = []
        self.val_loss_list = []
        print(iters, "steps per epoch")
        print(batch_size, "samples per step")
        start_time = time.time()
        stop_flag = False
        self.best_loss = np.inf
        self.best_ckpt = None
        if not validation_data and validation_split > 0:
            x,val_x,y,val_y = train_test_split(x,y,test_size = validation_split,random_state = self.seed)
            validation_data = [(val_x,val_y)]
        
        
        for i in range(epochs):
            if i < initial_epoch:
                continue
            if shuffle:
                x,y = sklearn_shuffle(x,y, random_state=self.seed) 
            for j in range(iters):
                batch_x = x[j * batch_size:(j + 1) * batch_size]
                batch_y = y[j * batch_size:(j + 1) * batch_size]
                
                self.train_on_batch(batch_x, batch_y )
                if j % min_display == 0:
                    tr_loss = self.evaluate(x, y, val_size)
                    self.tr_loss_list.append(tr_loss)
                    total_time = time.time() - start_time
                    if validation_data is None:
                        print("Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} tr_time {3: 0.1f}".format(i, j, tr_loss,
                                                                                                       total_time))
                    else:
                        val_loss = self.evaluate(validation_data[0][0], validation_data[0][1], val_size)
                        self.val_loss_list.append(val_loss)
                        print(
                            "Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} va_loss {3: 0.6f} tr_time {4: 0.1f}".format(
                                i, j, tr_loss, val_loss, total_time))

                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            # self.save_model(self.checkpoint_path+'best')

                # self.save_model(self.checkpoint_path)

                if (i * iters) + j == max_iter:
                    stop_flag = True
                    break
            if stop_flag:
                break


    def test_on_batch(self,x,y, ):
        """
        evaluate sum of batch loss
        """
        feed_dict_ = {self._get_input_data(): x,
                      self._get_input_target(): y, self.train_flag: False}
        loss = self.sess.run(self.metric_list, feed_dict=feed_dict_)
        return loss[0]

    def evaluate(self, x,y, val_size=2 ** 18):
        """
        evaluate the model and return mean loss
        :param data: DataFrame
        :param feature_list: list of features
        :param target_str:
        :param val_size:
        :return: mean loss
        """
        val_samples = x.shape[0]
        val_iters = (val_samples - 1) // val_size + 1
        total_val_loss = 0
        for i in range(0, val_iters):
            batch_x = x[i * val_size:(i + 1) * val_size]
            batch_y = y[i * val_size:(i + 1) * val_size]
            val_loss = self.test_on_batch(batch_x,batch_y )
            total_val_loss += val_loss
        return total_val_loss / val_samples

    def predict_on_batch(self, x, ):
        feed_dict_ = {self._get_input_data(): x, self.train_flag: False}
        prob = self.sess.run([self._get_output_target()], feed_dict=feed_dict_)
        return prob[0]

    def predict(self, x, batch_size=2 ** 18):
        n_samples = x.shape[0]
        iters = (n_samples - 1) // batch_size + 1
        pred_prob = np.array([])
        for j in range(iters):
            batch_x = x[j * batch_size:(j + 1) * batch_size]
            batch_prob = self.predict_on_batch(batch_x, )
            pred_prob = np.concatenate((pred_prob, batch_prob))
        return pred_prob

