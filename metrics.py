import tensorflow as tf
from functools import wraps
import numpy as np


def add_reset_op(sc=None):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tf.variable_scope(sc) as scope:
                metric_op, update_op = func(*args, **kwargs)
                vars = tf.contrib.framework.get_variables(
                    scope.original_name_scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
                reset_op = tf.variables_initializer(vars)
                with tf.control_dependencies([reset_op]):
                    one_shot_op = tf.identity(update_op)
            return metric_op, update_op, reset_op, one_shot_op

        return wrapper

    return decorate


@add_reset_op('metrics')
def _metrics_accuracy(labels, predictions, weights=None,
                      metrics_collections=None, updates_collections=None,
                      name=None):
    return tf.metrics.accuracy(labels, predictions, weights,
                               metrics_collections, updates_collections,
                               name)


def metrics_accuracy(labels, logits, weights=None,
                     metrics_collections=None, updates_collections=None,
                     name=None):
    predictions = tf.argmax(logits, axis=-1)
    return _metrics_accuracy(labels, predictions, weights,
                             metrics_collections, updates_collections,
                             name)


@add_reset_op('metrics')
def _metrics_mean_per_class_accuracy(labels, predictions, num_classes, weights=None,
                                     metrics_collections=None, updates_collections=None, name=None):
    return tf.metrics.mean_per_class_accuracy(labels, predictions, num_classes, weights,
                                              metrics_collections, updates_collections, name)


def metrics_mean_per_class_accuracy(labels, logits, num_classes, weights=None,
                                    metrics_collections=None, updates_collections=None, name=None):
    predictions = tf.argmax(logits, axis=-1)
    return _metrics_mean_per_class_accuracy(labels, predictions, num_classes, weights,
                                            metrics_collections, updates_collections, name)


@add_reset_op('metrics')
def _metrics_auc(labels, predictions, weights=None, num_thresholds=200,
                 metrics_collections=None, updates_collections=None,
                 curve='ROC', name=None, summation_method='trapezoidal'):
    return tf.metrics.auc(labels, predictions, weights, num_thresholds,
                          metrics_collections, updates_collections,
                          curve, name, summation_method)


def metrics_auc(labels, logits, weights=None, num_thresholds=200,
                metrics_collections=None, updates_collections=None,
                curve='ROC', name=None, summation_method='trapezoidal'):
    predictions = tf.nn.softmax(logits)[:, -1]
    return _metrics_auc(labels, predictions, weights, num_thresholds,
                        metrics_collections, updates_collections,
                        curve, name, summation_method)


def metrics_apc_np(confusion_matrix):
    corrects = np.diag(confusion_matrix)
    total_num_per_class = np.sum(confusion_matrix, axis=1)
    return np.concatenate([corrects[None].T, total_num_per_class[None].T], axis=1)


def rankdata(a, method='average', scope='rankdata'):
    with tf.name_scope(scope):
        arr = tf.reshape(a, shape=[-1])
        _, sorter = tf.nn.top_k(-arr, arr.get_shape()[-1])
        inv = tf.invert_permutation(sorter)
        if method == 'ordinal':
            res = inv + 1
        else:
            arr = tf.gather(arr, sorter)
            obs = tf.cast(tf.not_equal(arr[1:], arr[:-1]), dtype=tf.int32)
            obs = tf.concat([[1], obs], axis=0)
            dense = tf.gather(tf.cumsum(obs), inv)
            if method == 'dense':
                res = dense
            else:
                count = tf.reshape(tf.where(tf.not_equal(obs, tf.zeros_like(obs))), [-1])
                count = tf.concat([count, obs.get_shape()], axis=0)
                if method == 'max':
                    res = tf.gather(count, dense)
                elif method == 'min':
                    res = tf.gather(count, dense - 1) + 1
                else:
                    res = (tf.gather(count, dense) + tf.gather(count, dense - 1) + 1) / 2
        return tf.cast(res, tf.float32)
