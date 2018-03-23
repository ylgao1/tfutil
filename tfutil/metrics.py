import tensorflow as tf
from functools import wraps
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import confusion_matrix

__all__ = ['metric_variable', 'get_metric_name', 'metrics_accuracy', 'metrics_mean_per_class_accuracy',
           'metrics_apc_np', 'metrics_auc', 'metrics_confusion_matrix', 'metrics_spearman_correlation',
           'metrics_mean_squared_error', 'metrics_mean_absolute_error']


def metric_variable(shape, dtype=tf.float32, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""

    return variable_scope.variable(
        lambda: tf.zeros(shape, dtype),
        trainable=False,
        collections=[
            tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES
        ],
        validate_shape=validate_shape,
        name=name)


def _remove_squeezable_dimensions(predictions, labels, weights):
    """Squeeze or expand last dim if needed.

    Squeezes last dim of `predictions` or `labels` if their rank differs by 1
    (using confusion_matrix.remove_squeezable_dimensions).
    Squeezes or expands last dim of `weights` if its rank differs by 1 from the
    new rank of `predictions`.

    If `weights` is scalar, it is kept scalar.

    This will use static shape if available. Otherwise, it will add graph
    operations, which could result in a performance hit.

    Args:
      predictions: Predicted values, a `Tensor` of arbitrary dimensions.
      labels: Optional label `Tensor` whose dimensions match `predictions`.
      weights: Optional weight scalar or `Tensor` whose dimensions match
        `predictions`.

    Returns:
      Tuple of `predictions`, `labels` and `weights`. Each of them possibly has
      the last dimension squeezed, `weights` could be extended by one dimension.
    """
    predictions = tf.convert_to_tensor(predictions)
    if labels is not None:
        labels, predictions = confusion_matrix.remove_squeezable_dimensions(
            labels, predictions)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if weights is None:
        return predictions, labels, None

    weights = tf.convert_to_tensor(weights)
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return predictions, labels, weights

    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    if (predictions_rank is not None) and (weights_rank is not None):
        # Use static rank.
        if weights_rank - predictions_rank == 1:
            weights = tf.squeeze(weights, [-1])
        elif predictions_rank - weights_rank == 1:
            weights = tf.expand_dims(weights, [-1])
    else:
        # Use dynamic rank.
        weights_rank_tensor = tf.rank(weights)
        rank_diff = weights_rank_tensor - tf.rank(predictions)

        def _maybe_expand_weights():
            return tf.cond(
                tf.equal(rank_diff, -1),
                lambda: tf.expand_dims(weights, [-1]),
                lambda: weights)

        # Don't attempt squeeze if it will fail based on static check.
        if ((weights_rank is not None) and
                (not weights_shape.dims[-1].is_compatible_with(1))):
            maybe_squeeze_weights = lambda: weights
        else:
            maybe_squeeze_weights = lambda: tf.squeeze(weights, [-1])

        def _maybe_adjust_weights():
            return tf.cond(
                tf.equal(rank_diff, 1),
                maybe_squeeze_weights,
                _maybe_expand_weights)

        # If weights are scalar, do nothing. Otherwise, try to add or remove a
        # dimension to match predictions.
        weights = tf.cond(
            tf.equal(weights_rank_tensor, 0),
            lambda: weights, _maybe_adjust_weights)
    return predictions, labels, weights


def get_metric_name(metric_op):
    return metric_op.name.split('/')[-2]


def add_reset_op(sc=None):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tf.variable_scope(sc) as scope:
                metric_op, update_op = func(*args, **kwargs)
                vars = tf.contrib.framework.get_variables(
                    scope.original_name_scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
                reset_op = tf.variables_initializer(vars)
            metric_name = get_metric_name(metric_op)
            metric_summ_ph = tf.placeholder(dtype=tf.float32)
            metric_summ_op = tf.summary.scalar(metric_name, metric_summ_ph)
            return metric_op, update_op, reset_op, metric_summ_ph, metric_summ_op

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
        total_cm = metric_variable(
            [num_classes, num_classes], tf.float64, name='total_confusion_matrix')

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
        y1 = metric_variable(name='y1', shape=[1], validate_shape=False)
        y2 = metric_variable(name='y2', shape=[1], validate_shape=False)
        w = metric_variable(name='weights', shape=[1], validate_shape=False)
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


@add_reset_op('metrics')
def metrics_mean_squared_error(labels, predictions, weights=None, metrics_collections=None, updates_collections=None,
                               name=None):
    return tf.metrics.mean_squared_error(labels, predictions, weights, metrics_collections, updates_collections, name)


@add_reset_op('metrics')
def metrics_mean_absolute_error(labels, predictions, weights=None, metrics_collections=None, updates_collections=None,
                                name=None):
    return tf.metrics.mean_absolute_error(labels, predictions, weights, metrics_collections, updates_collections, name)
