import tensorflow as tf
from examples.testnet import TestNet
import tfutil

num_classes = 10
inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
labels = tf.placeholder(dtype=tf.int32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)

network = TestNet(num_classes, dropout_keep_prob=0.5)
logits = network(inputs, is_training)

loss = tfutil.loss(labels, logits)

learning_rate_init = 1e-3
learning_rate_decay_steps = 500
learning_rate = tfutil.dynamic_learning_rate(learning_rate_init, learning_rate_decay_steps)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = tfutil.create_train_op(loss, optimizer)

acc_op = tfutil.metrics_accuracy(labels, logits)
mpca_op = tfutil.metrics_mean_per_class_accuracy(labels, logits, num_classes)

metric_opdefs = [acc_op, mpca_op]

summ_learning_rate = tf.summary.scalar('train/learning_rate', learning_rate)
summ_loss = tf.summary.scalar('train/loss', loss)
summ_img = tf.summary.image('image', inputs, max_outputs=6)
extra_summ_op = [summ_learning_rate, summ_loss, summ_img]

model_tensors = tfutil.ModelTensors(inputs, labels, is_training, logits)

model_dir = 'temp/testnet/mnist'
batch_size = 100
num_epochs = 2

data_dir = '/data/testdata/mnist'
data_tr = f'{data_dir}/mnist_tr.55000_28_28_1.10.tfrec'
data_tv = f'{data_dir}/mnist_tv.5000_28_28_1.10.tfrec'
data_te = f'{data_dir}/mnist_te.10000_28_28_1.10.tfrec'

gntr = tfutil.read_tfrec(data_tr, batch_size, num_epochs)
gntv = tfutil.read_tfrec(data_tv, batch_size * 2, is_test=True)
gnte = tfutil.read_tfrec(data_te, batch_size * 2, is_test=True)

valid_listener = tfutil.Listener('validation', gntv, [acc_op])
test_listener = tfutil.Listener('test', gnte, metric_opdefs)
listeners = [valid_listener, test_listener]

model = tfutil.TFModel(model_tensors, model_dir)

model.train(train_op, gntr, num_epochs, metric_opdefs, extra_summ_op, listeners, max_checkpoint_to_keep=5,
            summ_steps=10, graph=tfutil.create_op_graph(loss), from_scratch=True)
