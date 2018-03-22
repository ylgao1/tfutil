import tensorflow as tf
import numpy as np
import pytest
import os

from ..tftrain import create_init_op
from ..metrics import metrics_accuracy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


N = 4

@pytest.fixture(scope='module')
def data_preparation():
    np.random.seed(1337)
    a1 = np.random.choice(np.arange(N), replace=True, size=[20]).astype(np.int32)
    a2 = np.random.choice(np.arange(N), replace=True, size=[20]).astype(np.int32)
    b = np.random.normal(size=[20, N]).astype(np.float32)
    return a1, a2, b

@pytest.fixture(scope='function')
def cgraph_preparation():
    a_ph = tf.placeholder(shape=[None], dtype=tf.int32)
    b_ph = tf.placeholder(shape=[None, N], dtype=tf.float32)
    metric_op, update_op, reset_op = metrics_accuracy(labels=a_ph, logits=b_ph)
    return (a_ph, b_ph), (metric_op, update_op, reset_op)

class TestAccuracy:
    def test_reset(self, data_preparation, cgraph_preparation):
        a1, _,  b = data_preparation
        (a_ph, b_ph), (metric_op, update_op, reset_op) = cgraph_preparation
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

    def test_update(self, data_preparation, cgraph_preparation):
        a1, a2, b = data_preparation
        (a_ph, b_ph), (metric_op, update_op, reset_op) = cgraph_preparation
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



















