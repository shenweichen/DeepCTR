import tensorflow as tf
from tensorflow.python.estimator.canned.head import _Head
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance

LINEAR_SCOPE_NAME = 'linear'
DNN_SCOPE_NAME = 'dnn'


def _summary_key(head_name, val):
    return '%s/%s' % (val, head_name) if head_name else val


class Head(_Head):

    def __init__(self, task,
                 name=None):
        self._task = task
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def logits_dimension(self):
        return 1

    def _eval_metric_ops(self,
                         labels,
                         logits,
                         predictions,
                         unweighted_loss,
                         weights=None):

        labels = to_float(labels)
        predictions = to_float(predictions)

        # with name_scope(None, 'metrics', (labels, logits, predictions,
        # unweighted_loss, weights)):
        metrics = get_metrics()
        losses = get_losses()

        metric_ops = {
            _summary_key(self._name, "prediction/mean"): metrics.mean(predictions, weights=weights),
            _summary_key(self._name, "label/mean"): metrics.mean(labels, weights=weights),
        }
        tf.summary.scalar("prediction/mean", metric_ops[_summary_key(self._name, "prediction/mean")][1])
        tf.summary.scalar("label/mean", metric_ops[_summary_key(self._name, "label/mean")][1])

        mean_loss = losses.compute_weighted_loss(
            unweighted_loss, weights=1.0, reduction=losses.Reduction.MEAN)

        if self._task == "binary":
            metric_ops[_summary_key(self._name, "LogLoss")] = metrics.mean(mean_loss, weights=weights, )
            tf.summary.scalar("LogLoss", mean_loss)

            metric_ops[_summary_key(self._name, "AUC")] = metrics.auc(labels, predictions, weights=weights)
            tf.summary.scalar("AUC", metric_ops[_summary_key(self._name, "AUC")][1])
        else:

            metric_ops[_summary_key(self._name, "MSE")] = metrics.mean_squared_error(labels, predictions,
                                                                                     weights=weights)
            tf.summary.scalar("MSE", mean_loss)

            metric_ops[_summary_key(self._name, "MAE")] = metrics.mean_absolute_error(labels, predictions,
                                                                                      weights=weights)
            tf.summary.scalar("MAE", metric_ops[_summary_key(self._name, "MAE")][1])

        return metric_ops

    def create_loss(self, features, mode, logits, labels):
        del mode, features  # Unused for this head.
        losses = get_losses()
        if self._task == "binary":
            loss = losses.sigmoid_cross_entropy(labels, logits, reduction=losses.Reduction.NONE)
        else:
            loss = losses.mean_squared_error(labels, logits, reduction=losses.Reduction.NONE)
        return loss

    def create_estimator_spec(
            self, features, mode, logits, labels=None, train_op_fn=None, training_chief_hooks=None):
        # with name_scope('head'):
        logits = tf.reshape(logits, [-1, 1])
        if self._task == 'binary':
            pred = tf.sigmoid(logits)
        else:
            pred = logits

        predictions = {"pred": pred, "logits": logits}
        export_outputs = {"predict": tf.estimator.export.PredictOutput(predictions)}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

        labels = tf.reshape(labels, [-1, 1])

        unweighted_loss = self.create_loss(features, mode, logits, labels)

        losses = get_losses()
        loss = losses.compute_weighted_loss(
            unweighted_loss, weights=1.0, reduction=losses.Reduction.SUM)
        reg_loss = losses.get_regularization_loss()

        training_loss = loss + reg_loss

        eval_metric_ops = self._eval_metric_ops(labels, logits, pred, unweighted_loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=training_loss,
            train_op=train_op_fn(training_loss),
            eval_metric_ops=eval_metric_ops,
            training_chief_hooks=training_chief_hooks)


def deepctr_model_fn(features, mode, logits, labels, task, linear_optimizer, dnn_optimizer, training_chief_hooks):
    linear_optimizer = get_optimizer_instance(linear_optimizer, 0.005)
    dnn_optimizer = get_optimizer_instance(dnn_optimizer, 0.01)
    train_op_fn = get_train_op_fn(linear_optimizer, dnn_optimizer)

    head = Head(task)
    return head.create_estimator_spec(features=features,
                                      mode=mode,
                                      labels=labels,
                                      train_op_fn=train_op_fn,
                                      logits=logits, training_chief_hooks=training_chief_hooks)


def get_train_op_fn(linear_optimizer, dnn_optimizer):
    def _train_op_fn(loss):
        train_ops = []
        global_step = tf.train.get_global_step() if tf.__version__ < "2.0.0" else tf.compat.v1.train.get_global_step()
        linear_var_list = get_collection(get_GraphKeys().TRAINABLE_VARIABLES, LINEAR_SCOPE_NAME)
        dnn_var_list = get_collection(get_GraphKeys().TRAINABLE_VARIABLES, DNN_SCOPE_NAME)

        if len(dnn_var_list) > 0:
            train_ops.append(
                dnn_optimizer.minimize(
                    loss,
                    var_list=dnn_var_list))
        if len(linear_var_list) > 0:
            train_ops.append(
                linear_optimizer.minimize(
                    loss,
                    var_list=linear_var_list))

        train_op = tf.group(*train_ops)
        with tf.control_dependencies([train_op]):
            return tf.assign_add(global_step, 1).op if tf.__version__ < "2.0.0" else tf.compat.v1.assign_add(
                global_step, 1).op

    return _train_op_fn


def variable_scope(name_or_scope):
    if tf.__version__ < "2.0.0":
        return tf.variable_scope(name_or_scope)
    else:
        return tf.compat.v1.variable_scope(name_or_scope)


def name_scope(name, default_name=None, values=None):
    if tf.__version__ < "2.0.0":
        return tf.name_scope(name, default_name, values)
    else:
        return tf.compat.v1.name_scope(name, default_name, values)


def get_collection(key, scope=None):
    if tf.__version__ < "2.0.0":
        return tf.get_collection(key, scope=scope)
    else:
        return tf.compat.v1.get_collection(key, scope=scope)


def get_GraphKeys():
    if tf.__version__ < "2.0.0":
        return tf.GraphKeys
    else:
        return tf.compat.v1.GraphKeys


def get_losses():
    if tf.__version__ < "2.0.0":
        return tf.losses
    else:
        return tf.compat.v1.losses


def input_layer(features, feature_columns):
    if tf.__version__ < "2.0.0":
        return tf.feature_column.input_layer(features, feature_columns)
    else:
        return tf.compat.v1.feature_column.input_layer(features, feature_columns)


def get_metrics():
    if tf.__version__ < "2.0.0":
        return tf.metrics
    else:
        return tf.compat.v1.metrics


def to_float(x, name="ToFloat"):
    if tf.__version__ < "2.0.0":
        return tf.to_float(x, name)
    else:
        return tf.compat.v1.to_float(x, name)
