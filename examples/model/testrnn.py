import tensorflow as tf
import sonnet as snt


class TestRNN(snt.AbstractModule):
    def __init__(self, cell, name='testrnn'):
        super(TestRNN, self).__init__(name=name)
        self._cell = cell

    def _build(self, inputs, is_training):
        outputs, states = tf.nn.dynamic_rnn(self._cell, inputs, dtype=tf.float32)
        return outputs, states

