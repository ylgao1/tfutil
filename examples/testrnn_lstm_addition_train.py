import tensorflow as tf
import numpy as np

import tfutil
from examples.model.testrnn import TestRNN


def data_generator(time_steps=100, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def gen():
        while True:
            add_values = np.random.rand(time_steps)
            add_indices = np.zeros_like(add_values)

            half = time_steps // 2
            first_half = np.random.randint(half)
            second_half = np.random.randint(half, time_steps)
            add_indices[[first_half, second_half]] = 1
            inputs = np.dstack((add_values, add_indices)).astype(np.float32)
            labels = np.sum(add_values * add_indices).astype(np.float32)
            yield inputs[0], labels

    return gen

tf.reset_default_graph()

inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 2])
labels = tf.placeholder(dtype=tf.float32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)

cell_units = 128

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=cell_units, num_proj=1)
network = TestRNN(lstm_cell, name='lstm_network')

logits_series, states = network(inputs, is_training)
logits = logits_series[:,-1, 0]


loss = tf.losses.mean_squared_error(labels, logits, scope='loss')
learing_rate_init = 2e-3
learning_rate_decay_steps = 20000
learning_rate = tfutil.dynamic_learning_rate(learing_rate_init, learning_rate_decay_steps)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = tfutil.create_train_op(loss, optimizer)

mes_op = tfutil.metrics_mean_squared_error(labels, logits)
mabse_op = tfutil.metrics_mean_absolute_error(labels, logits)

metric_opdefs = [mes_op, mabse_op]

summ_learing_rate = tf.summary.scalar('train/learning_rate', learning_rate)
summ_loss = tf.summary.scalar('train/loss', loss)
extra_summ_op = [summ_learing_rate, summ_loss]

model_tensors = tfutil.ModelTensors(inputs, labels, is_training, logits)
model_dir = 'temp/testrnn_lstm/addition'

batch_size = 50
num_steps = 20000
steps_per_checkpoint = 100

gntr = tfutil.read_tfrec_infinite_generator(data_generator(100), batch_size)
gntv = tfutil.read_tfrec_infinite_generator(data_generator(50), batch_size)
gnte = tfutil.read_tfrec_infinite_generator(data_generator(100), batch_size)

valid_listener = tfutil.Listener('valid', gntv, [mes_op])
test_listener = tfutil.Listener('test', gnte, metric_opdefs)
listeners = [valid_listener, test_listener]

model = tfutil.TFModel(model_tensors, model_dir)
model.train_stepwise(train_op, gntr, num_steps, steps_per_checkpoint,
                     metric_opdefs, extra_summ_op, listeners, from_scratch=True)






















