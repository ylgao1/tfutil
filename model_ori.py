import tensorflow as tf
import collections
from .listeners import MultiClsTestListerner, BinaryClsTestListerner
from .misc import delete_and_make_dir
from .tfio import read_tfrec_array
from .tftrain import load_ckpt, load_ckpt_path
from progress.bar import Bar
import numpy as np


ModelTensors = collections.namedtuple('ModelTensors',
                                      ('inputs', 'targets',
                                       'is_training', 'logits'))


class TFModel:
    def __init__(self, model_tensors, checkpoint_dir=None):
        self._inputs = model_tensors.inputs
        self._targets = model_tensors.targets
        self._is_training = model_tensors.is_training
        self._logits = model_tensors.logits
        self._checkpoint_dir = checkpoint_dir
        self._sess = None
        self._model_loaded = False

    def load_weights(self, sess, ckpt_path=None):
        self._sess = sess
        if ckpt_path is None:
            load_ckpt(sess, self._checkpoint_dir)
        else:
            load_ckpt_path(sess, ckpt_path)
        self._model_loaded = True

    def train(self, train_op, gntr, num_epochs, gnte=None,
              summ_op=None, training_type=0,
              max_checkpoint_to_keep=10, summ_steps=100,
              from_scratch=True):
        """
        training_type:  0 for multi-class classification
                        1 for binary-class classification
        """
        if from_scratch:
            delete_and_make_dir(self._checkpoint_dir)
        global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
        data_gntr, steps_per_epoch = gntr
        listeners = []
        if gnte is not None:
            test_logdir = f'{self._checkpoint_dir}/test'
            listener_class = None
            if training_type == 0:
                listener_class = MultiClsTestListerner
            elif training_type == 1:
                listener_class = BinaryClsTestListerner
            listener = listener_class(test_logdir, gnte,
                                      self._inputs, self._targets, self._is_training, self._logits)
            listeners.append(listener)
        hooks = [
            tf.train.CheckpointSaverHook(checkpoint_dir=self._checkpoint_dir, save_steps=steps_per_epoch,
                                         saver=tf.train.Saver(max_to_keep=max_checkpoint_to_keep), listeners=listeners)
        ]
        with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=self._checkpoint_dir) as sess:
            global_step_value = sess.run(global_step)
            ckpt_hk = sess._hooks[0]
            summ_writer = ckpt_hk._summary_writer
            epoch = global_step_value // steps_per_epoch
            for _ in range(num_epochs):
                bar = Bar(f'Epoch {epoch+1}', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
                for _ in range(steps_per_epoch):
                    bar.next()
                    xb, yb = sess.run(data_gntr)
                    fd = {self._inputs: xb, self._targets: yb, self._is_training: True}
                    if global_step_value == 0:
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=global_step_value)
                    sess.run(train_op, feed_dict=fd)
                    global_step_value = sess.run(global_step)
                    if global_step_value % summ_steps == 0:
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=global_step_value)
                epoch = global_step_value // steps_per_epoch
                summ_writer.flush()
                bar.finish()

    def predict(self, gnte_pred):
        data_gn, data_gn_init_op, steps_per_epoch = gnte_pred
        logits_lst = []
        if not self._model_loaded:
            raise RuntimeError('Load model first!')
        sess = self._sess
        sess.run(data_gn_init_op)
        bar = Bar('Test predict', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for _ in range(steps_per_epoch):
            bar.next()
            xb = sess.run(data_gn)
            fd = {self._inputs: xb, self._is_training: False}
            logits_val = sess.run(self._logits, feed_dict=fd)
            logits_lst.append(logits_val)
        bar.finish()
        logits = np.concatenate(logits_lst, axis=0)
        return logits

    def predict_from_array(self, x, batch_size=None):
        data_gn, data_gn_init_op, steps_per_epoch = read_tfrec_array(x, batch_size=batch_size, is_test=True)
        logits_lst = []
        if not self._model_loaded:
            raise RuntimeError('Load model first!')
        sess = self._sess
        sess.run(data_gn_init_op)
        bar = Bar('Test predict', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for _ in range(steps_per_epoch):
            bar.next()
            xb = sess.run(data_gn)
            fd = {self._inputs: xb, self._is_training: False}
            logits_val = sess.run(self._logits, feed_dict=fd)
            logits_lst.append(logits_val)
        bar.finish()
        logits = np.concatenate(logits_lst, axis=0)
        return logits

    def eval(self, metrics_ops, gnte):
        data_gn, data_gn_init_op, steps_per_epoch = gnte
        if not isinstance(metrics_ops[0], tuple):
            metrics_ops = [metrics_ops]
        _, update_ops, reset_ops, _ = list(zip(*metrics_ops))
        logits_lst = []
        if not self._model_loaded:
            raise RuntimeError('Load model first!')
        sess = self._sess
        sess.run(data_gn_init_op)
        sess.run(reset_ops)
        metrics = None
        bar = Bar('Test evaluation', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for _ in range(steps_per_epoch):
            bar.next()
            xb, yb = sess.run(data_gn)
            fd = {self._inputs: xb, self._targets: yb, self._is_training: False}
            logits_val, metrics = sess.run([self._logits, update_ops], feed_dict=fd)
            logits_lst.append(logits_val)
        bar.finish()
        logits = np.concatenate(logits_lst, axis=0)
        return logits, metrics

    def eval_from_array(self, metrics_ops, x, y, batch_size=None):
        data_gn, data_gn_init_op, steps_per_epoch = read_tfrec_array((x, y), batch_size=batch_size, is_test=True)
        if not isinstance(metrics_ops[0], tuple):
            metrics_ops = [metrics_ops]
        _, update_ops, reset_ops, _ = list(zip(*metrics_ops))
        logits_lst = []
        if not self._model_loaded:
            raise RuntimeError('Load model first!')
        sess = self._sess
        sess.run(data_gn_init_op)
        sess.run(reset_ops)
        metrics = None
        bar = Bar('Test evaluation', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for _ in range(steps_per_epoch):
            bar.next()
            xb, yb = sess.run(data_gn)
            fd = {self._inputs: xb, self._targets: yb, self._is_training: False}
            logits_val, metrics = sess.run([self._logits, update_ops], feed_dict=fd)
            logits_lst.append(logits_val)
        bar.finish()
        logits = np.concatenate(logits_lst, axis=0)
        return logits, metrics