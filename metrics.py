import tensorflow as tf
from functools import wraps
import numpy as np
from tensorflow.python.ops.metrics_impl import _create_local, _remove_squeezable_dimensions
from tensorflow.python.ops import confusion_matrix


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
                    op1 = tf.identity(update_op)
                with tf.control_dependencies([op1]):
                    op2 = tf.identity(metric_op)
            return metric_op, update_op, reset_op, op1, op2

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


def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None,
                                metrics_collections=None,
                                updates_collections=None,
                                name=None):
    """Calculate a streaming confusion matrix.

    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.

    Args:
      labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
      predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
      num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).

    Returns:
      total_cm: A `Tensor` representing the confusion matrix.
      update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    with tf.variable_scope(name, 'confusion_matrix'):
        total_cm = _create_local(
            'total_confusion_matrix',
            shape=[num_classes, num_classes],
            dtype=tf.float64)

        # Cast the type to int64 required by confusion_matrix_ops.
        predictions = tf.to_int64(predictions)
        labels = tf.to_int64(labels)
        num_classes = tf.to_int64(num_classes)

        # Flatten the input if its rank > 1.
        if predictions.get_shape().ndims > 1:
            predictions = tf.reshape(predictions, [-1])

        if labels.get_shape().ndims > 1:
            labels = tf.reshape(labels, [-1])

        if (weights is not None) and (weights.get_shape().ndims > 1):
            weights = tf.reshape(weights, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            labels, predictions, num_classes, weights=weights, dtype=tf.float64)
        update_op = tf.assign_add(total_cm, current_cm)
        if metrics_collections:
            tf.add_to_collection(metrics_collections, total_cm)
        if updates_collections:
            tf.add_to_collection(updates_collections, update_op)
        return total_cm, update_op


@add_reset_op('metrics')
def _metrics_streaming_confusion_matrix(labels, predictions, num_classes, weights=None,
                                        metrics_collections=None, updates_collections=None,
                                        name=None):
    return _streaming_confusion_matrix(labels, predictions, num_classes, weights,
                                       metrics_collections, updates_collections,
                                       name)


def metrics_confusion_matrix(labels, logits, num_classes, weights=None,
                             metrics_collections=None, updates_collections=None,
                             name=None):
    predictions = tf.argmax(logits, axis=-1)
    return _metrics_streaming_confusion_matrix(labels, predictions, num_classes, weights,
                                               metrics_collections, updates_collections,
                                               name)


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
        _, sorter = tf.nn.top_k(-arr, tf.shape(arr)[-1])
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
                count = tf.concat([tf.cast(count, tf.int32), tf.shape(obs)], axis=0)
                if method == 'max':
                    res = tf.gather(count, dense)
                elif method == 'min':
                    res = tf.gather(count, dense - 1) + 1
                else:
                    res = (tf.gather(count, dense) + tf.gather(count, dense - 1) + 1) / 2
        return tf.cast(res, tf.float32)


def streaming_spearman_correlation(labels, predictions, weights=None,
                                   rank_method='average',
                                   metrics_collections=None,
                                   updates_collections=None,
                                   name=None):
    def _weighted_pearson(x, y, w):
        xd = x - tf.reduce_sum(x * w) / tf.reduce_sum(w)
        yd = y - tf.reduce_sum(y * w) / tf.reduce_sum(w)
        corr = ((tf.reduce_sum(w * yd * xd) / tf.reduce_sum(w)) /
                tf.sqrt((tf.reduce_sum(w * yd ** 2) * tf.reduce_sum(w * xd ** 2)) / (tf.reduce_sum(w) ** 2)))
        return corr

    with tf.variable_scope(name, 'spearman_r'):
        predictions, labels, weights = _remove_squeezable_dimensions(
            predictions, labels, weights)
        y1 = _create_local('y1', shape=[1], validate_shape=False)
        y2 = _create_local('y2', shape=[1], validate_shape=False)
        w = _create_local('weights', shape=[1], validate_shape=False)
        if weights is None:
            weights = tf.ones_like(labels, dtype=labels.dtype)
        update_y1 = tf.assign(y1, tf.concat([y1, labels], axis=-1), validate_shape=False)[1:]
        update_y2 = tf.assign(y2, tf.concat([y2, predictions], axis=-1), validate_shape=False)[1:]
        update_w = tf.assign(w, tf.concat([w, weights], axis=-1), validate_shape=False)[1:]
        update_op = tf.group(update_y1, update_y2, update_w, name='update_op')
        ry1 = rankdata(y1[1:], rank_method)
        ry2 = rankdata(y2[1:], rank_method)
        cur_w = w[1:]
        spearman_r = tf.identity(_weighted_pearson(ry1, ry2, cur_w), name='spearman_r')
        if metrics_collections:
            tf.add_to_collection(metrics_collections, spearman_r)
        if updates_collections:
            tf.add_to_collection(updates_collections, update_op)
    return spearman_r, update_op


@add_reset_op('metrics')
def metrics_spearman_correlation(labels, predictions, weights=None,
                                 rank_method='average',
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
    return streaming_spearman_correlation(labels, predictions, weights,
                                          rank_method,
                                          metrics_collections,
                                          updates_collections,
                                          name)
