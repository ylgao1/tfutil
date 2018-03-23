import tensorflow as tf
import sonnet as snt
from tensorflow.contrib import slim


class TestNet(snt.AbstractModule):
    def __init__(self,
                 num_classes,
                 dropout_keep_prob=0.5,
                 name='testnet'):
        super(TestNet, self).__init__(name=name)
        self._num_classes = num_classes
        self._dropout_keep_prob = dropout_keep_prob

    @classmethod
    def arg_score(cls, is_training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([slim.dropout], is_training=is_training):
                with slim.arg_scope([slim.conv2d], padding='SAME'):
                    with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                        return arg_sc

    def _build(self, inputs, is_training):
        with slim.arg_scope(self.arg_score(is_training)):
            net = slim.conv2d(inputs, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='maxpool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='maxpool2')
            net = snt.BatchFlatten()(net)
            net = slim.fully_connected(net, 1024, scope='fc1')
            net = slim.dropout(net, keep_prob=self._dropout_keep_prob, scope='dp1')
            net = slim.fully_connected(net, self._num_classes, activation_fn=None,
                                       scope='fc2')
            return net
