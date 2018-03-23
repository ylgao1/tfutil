import tensorflow as tf
import numpy as np
import pytest
import os

from tfutil.tftrain import create_init_op
from tfutil.metrics import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N = 4


@pytest.fixture(scope='module')
def classification_data_preparation():
    np.random.seed(1337)
    a1 = np.random.choice(np.arange(N), replace=True, size=[20]).astype(np.int32)
    a2 = np.random.choice(np.arange(N), replace=True, size=[20]).astype(np.int32)
    b = np.random.normal(size=[20, N]).astype(np.float32)
    return a1, a2, b


@pytest.fixture(scope='class')
def acc_cgraph_preparation():
    a_ph = tf.placeholder(shape=[None], dtype=tf.int32)
    b_ph = tf.placeholder(shape=[None, N], dtype=tf.float32)
    metric_op, update_op, reset_op, _ = metrics_accuracy(labels=a_ph, logits=b_ph)
    yield (a_ph, b_ph), (metric_op, update_op, reset_op)
    tf.reset_default_graph()


class TestAccuracy:
    def test_reset(self, classification_data_preparation, acc_cgraph_preparation):
        a1, _, b = classification_data_preparation
        (a_ph, b_ph), (metric_op, update_op, reset_op) = acc_cgraph_preparation
        fd = {a_ph: a1, b_ph: b}
        total_t, count_t = tf.local_variables()
        with tf.Session() as sess:
            sess.run(create_init_op())
            sess.run(update_op, feed_dict=fd)
            count = sess.run(count_t)
            np.testing.assert_almost_equal(len(a1), count, 3)
            sess.run(reset_op)
            resetted_total = sess.run(total_t)
            resetted_count = sess.run(count_t)
            np.testing.assert_almost_equal(0, resetted_total, 3)
            np.testing.assert_almost_equal(0, resetted_count, 3)

    def test_update(self, classification_data_preparation, acc_cgraph_preparation):
        a1, a2, b = classification_data_preparation
        (a_ph, b_ph), (metric_op, update_op, reset_op) = acc_cgraph_preparation
        fd1 = {a_ph: a1, b_ph: b}
        fd2 = {a_ph: a2, b_ph: b}
        with tf.Session() as sess:
            sess.run(create_init_op())
            # a1 a2
            sess.run(update_op, feed_dict=fd1)
            sess.run(update_op, feed_dict=fd2)
            acc1_2 = sess.run(metric_op)
            # reset
            sess.run(reset_op)
            # a2 a1
            sess.run(update_op, feed_dict=fd2)
            sess.run(update_op, feed_dict=fd1)
            acc2_2 = sess.run(metric_op)
            np.testing.assert_almost_equal(acc1_2, acc2_2, 3)


@pytest.fixture(scope='module')
def regression_data_preparation():
    np.random.seed(1337)
    a1 = np.random.rand(20)
    b1 = np.random.rand(20)
    a2 = np.random.rand(20)
    b2 = np.random.rand(20)
    return a1, b1, a2, b2


@pytest.fixture(scope='class')
def mse_cgraph_preparation():
    a_ph = tf.placeholder(shape=[None], dtype=tf.float32)
    b_ph = tf.placeholder(shape=[None], dtype=tf.float32)
    metric_op, update_op, reset_op, _ = metrics_mean_squared_error(labels=a_ph, predictions=b_ph)
    print('\n{0}'.format(get_metric_name(metric_op)))
    yield (a_ph, b_ph), (metric_op, update_op, reset_op)
    tf.reset_default_graph()


class TestMSE:
    def test_reset(self, regression_data_preparation, mse_cgraph_preparation):
        a1, b1, _, _ = regression_data_preparation
        (a_ph, b_ph), (metric_op, update_op, reset_op) = mse_cgraph_preparation
        fd = {a_ph: a1, b_ph: b1}
        total_t, count_t = tf.local_variables()
        with tf.Session() as sess:
            sess.run(create_init_op())
            sess.run(update_op, feed_dict=fd)
            count = sess.run(count_t)
            np.testing.assert_almost_equal(len(a1), count, 3)
            sess.run(reset_op)
            resetted_total = sess.run(total_t)
            resetted_count = sess.run(count_t)
            np.testing.assert_almost_equal(0, resetted_total, 3)
            np.testing.assert_almost_equal(0, resetted_count, 3)

    def test_update(self, regression_data_preparation, mse_cgraph_preparation):
        a1, b1, a2, b2 = regression_data_preparation
        (a_ph, b_ph), (metric_op, update_op, reset_op) = mse_cgraph_preparation
        fd1 = {a_ph: a1, b_ph: b1}
        fd2 = {a_ph: a2, b_ph: b2}
        with tf.Session() as sess:
            sess.run(create_init_op())
            # a1 a2
            sess.run(update_op, feed_dict=fd1)
            sess.run(update_op, feed_dict=fd2)
            mse1_2 = sess.run(metric_op)
            # reset
            sess.run(reset_op)
            # a2 a1
            sess.run(update_op, feed_dict=fd2)
            sess.run(update_op, feed_dict=fd1)
            mse2_2 = sess.run(metric_op)
            np.testing.assert_almost_equal(mse1_2, mse2_2, 5)
            # manual calculation
            mse_man = (np.sum((a1 - b1) ** 2) + np.sum((a2 - b2) ** 2)) / (len(a1) + len(a2))
            np.testing.assert_almost_equal(mse1_2, mse_man, 5)
