import tensorflow as tf
from tensorflow.contrib import slim

def loss(logits, label_pl, is_one_hot=False, scope=None):
    with tf.name_scope(scope):
        if is_one_hot:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_pl, logits=logits))
        else:
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_pl, logits=logits))
        total_loss = cross_entropy + tf.add_n(tf.losses.get_regularization_losses())
    return total_loss


def create_train_op(total_loss, optimizer):
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    return train_op


def cal_accuracy(logits, label_pl, top_k=1):
    top_k_op = tf.nn.in_top_k(logits, label_pl, top_k)
    correct_num = tf.reduce_sum(tf.cast(top_k_op, tf.int32), name='correct_num')
    accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32), name='accuracy')
    return correct_num, accuracy

























