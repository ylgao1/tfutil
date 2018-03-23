import tensorflow as tf
import tfutil
from examples.model.testnet import TestNet
import numpy as np
from sklearn.externals import joblib

num_classes = 10
inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
labels = tf.placeholder(dtype=tf.int32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)

network = TestNet(num_classes, dropout_keep_prob=0.5)
logits = network(inputs, is_training)

acc_op = tfutil.metrics_accuracy(labels, logits)
mapc_op = tfutil.metrics_mean_per_class_accuracy(labels, logits, num_classes)
cm_op = tfutil.metrics_confusion_matrix(labels, logits, num_classes)

model_tensors = tfutil.ModelTensors(inputs, labels, is_training, logits)

model_dir = 'temp/testnet/mnist'
batch_size = 100

data_dir = '/data/testdata/mnist'
data_te = f'{data_dir}/mnist_te.10000_28_28_1.10.tfrec'
_, _, _, _, xte, yte = joblib.load('/data/testdata/mnist/mnist.pkl')

gnte = tfutil.read_tfrec(data_te, batch_size * 2, is_test=True)


model = tfutil.TFModel(model_tensors, model_dir)

sess = tf.InteractiveSession()
model.load_weights(sess)

logits_val, metrics = model.eval([acc_op, mapc_op, cm_op], gnte)
acc = metrics[0]
print(f'Test accuracy directly from model: {acc}')
mapc = metrics[1]
print(f'Mean accuracy per class: {mapc}')
apc = tfutil.metrics_apc_np(metrics[2])
print(f'Corrects per class: {apc}')
print(f'Test accuracy from apc: {np.sum(apc[:, 0]) / np.sum(apc[:,1])}')

logits_val_array, metrics_array = model.eval_from_array([acc_op, mapc_op, cm_op], xte, yte, batch_size)
print(f'Another accuracy: {metrics_array[0]}')
print(f'Another mapc: {metrics_array[1]}')
print(f'Another cpc: {tfutil.metrics_apc_np(metrics_array[2])}')

# Compared with manually calculated metrics

from sklearn.metrics import accuracy_score

ypred_classes = np.argmax(logits_val, axis=1)
acc_np = accuracy_score(yte, ypred_classes)
print(f'Test accuracy from sklearn: {acc_np}')
