import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def sigmoid_cross_entropy_with_probs(labels=None,probs=None,name=None):
    try:
        labels.get_shape().merge_with(probs.get_shape())
    except ValueError:
        raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))
    return  -tf.reduce_sum(labels * tf.log(probs,)+(1-labels)*tf.log(1-probs), name=name)

def tf_weighted_sigmoid_ce_with_logits(labels=None, logits=None,sample_weight=None):
    return tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits),sample_weight)

def guarantee_initialized_variables(session, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   session.run(tf.report_uninitialized_variables(list_of_variables)))
    session.run(tf.initialize_variables(uninitialized_variables))
    #return unintialized_variables
def init_uninitialized(sess):
	sess.run(tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in set(sess.run(tf.report_uninitialized_variables()))
]))
def new_variable_initializer(sess):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            _ = sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    return init_new_vars_op
    #sess.run(init_new_vars_op)
def get_sample_num(x):
    if isinstance(x,list):
        return x[0].shape[0]
    else:
        return x.shape[0]
def sklearn_shuffle(x,y,random_state):
    if isinstance(x,list):
        l = len(x)
        res = shuffle(*x,y,random_state=random_state)
        return res[:l],res[-1]
    else:
        return shuffle(x,y,random_state=random_state)
def sklearn_split(x,y,test_size,random_state):
    if isinstance(x,list):
        l = len(x)
        res = train_test_split(*x,y,test_size = test_size,random_state = random_state)
        return res[0:2*l:2],res[1:2*l:2],res[-2],res[-1]
    else:
        return train_test_split(x,y,test_size = test_size,random_state = random_state)
