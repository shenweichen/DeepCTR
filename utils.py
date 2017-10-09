import tensorflow as tf
def sigmoid_cross_entropy_with_probs(labels=None,probs=None,name=None):
    try:
        labels.get_shape().merge_with(probs.get_shape())
    except ValueError:
        raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))
    return  -tf.reduce_sum(labels * tf.log(probs,)+(1-labels)*tf.log(1-probs), name=name)