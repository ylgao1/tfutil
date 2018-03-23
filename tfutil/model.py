import tensorflow as tf
import collections
from .misc import delete_and_make_dir
from .tfio import read_tfrec_array
from .tftrain import load_ckpt, load_ckpt_path, create_init_op
from .visualization import saliency_grad
from progress.bar import Bar
import numpy as np

__all__ = ['ModelTensors', 'TFModel']

ModelTensors = collections.namedtuple('ModelTensors',
                                      ('inputs', 'labels',
                                       'is_training', 'logits'))


class TFModel:
    def __init__(self, model_tensors, checkpoint_dir=None):
        self._inputs = model_tensors.inputs
        self._labels = model_tensors.labels
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

    def train(self, train_op, gntr, num_epochs,
              metric_opdefs, extra_summ_ops=None, listeners=None,
              max_checkpoint_to_keep=10, summ_steps=100,
              graph=None, from_scratch=True):

        metric_ops, update_ops, reset_ops, _, _ = list(zip(*metric_opdefs))
        metric_summ_names = ['train/{0}'.format(m.name.split('/')[-2]) for m in metric_ops]
        metric_summ_ops = [tf.summary.scalar(*tup) for tup in list(zip(metric_summ_names, metric_ops))]
        summ_ops = metric_summ_ops + list(extra_summ_ops) if extra_summ_ops else metric_summ_ops
        summ_op = tf.summary.merge(summ_ops)
        if from_scratch:
            delete_and_make_dir(self._checkpoint_dir)
        global_step = tf.train.get_or_create_global_step()
        data_gntr, steps_per_epoch = gntr

        saver = tf.train.Saver(max_to_keep=max_checkpoint_to_keep)
        summ_writer = tf.summary.FileWriter(f'{self._checkpoint_dir}/train')
        if graph is not None:
            summ_writer.add_graph(graph)
        if listeners:
            for l in listeners:
                l.begin(self._checkpoint_dir, self._inputs, self._labels, self._is_training, self._logits,
                        steps_per_epoch)

        with tf.Session() as sess:
            sess.run(create_init_op())
            if not from_scratch:
                load_ckpt(sess, model_dir=self._checkpoint_dir)
            global_step_value = sess.run(global_step)
            epoch = global_step_value // steps_per_epoch
            for _ in range(num_epochs):
                bar = Bar(f'Epoch {epoch+1}', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
                for _ in range(steps_per_epoch):
                    xb, yb = sess.run(data_gntr)
                    bar.next()
                    fd = {self._inputs: xb, self._labels: yb, self._is_training: True}
                    if global_step_value == 0:
                        sess.run(update_ops, feed_dict=fd)
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=global_step_value)
                        sess.run(reset_ops)
                    sess.run(train_op, feed_dict=fd)
                    global_step_value = sess.run(global_step)
                    if global_step_value % summ_steps == 0:
                        sess.run(update_ops, feed_dict=fd)
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=global_step_value)
                        sess.run(reset_ops)

                epoch = global_step_value // steps_per_epoch
                summ_writer.flush()
                bar.finish()
                saver.save(sess, f'{self._checkpoint_dir}/model.ckpt', global_step_value, write_meta_graph=False)
                if listeners:
                    for l in listeners:
                        l.run(sess, global_step_value)

            if listeners:
                for l in listeners:
                    l.end()

    def train_stepwise(self, train_op, gntr, num_steps, steps_per_checkpoint,
                       metric_opdefs, extra_summ_ops=None, listeners=None,
                       max_checkpoint_to_keep=10, summ_steps=100,
                       graph=None, from_scratch=True):
        metric_ops, update_ops, reset_ops, _, _ = list(zip(*metric_opdefs))
        metric_summ_names = ['train/{0}'.format(m.name.split('/')[-2]) for m in metric_ops]
        metric_summ_ops = [tf.summary.scalar(*tup) for tup in list(zip(metric_summ_names, metric_ops))]
        summ_ops = metric_summ_ops + list(extra_summ_ops) if extra_summ_ops else metric_summ_ops
        summ_op = tf.summary.merge(summ_ops)
        if from_scratch:
            delete_and_make_dir(self._checkpoint_dir)
        global_step = tf.train.get_or_create_global_step()
        data_gntr, _ = gntr
        num_checkpoints = num_steps // steps_per_checkpoint

        saver = tf.train.Saver(max_to_keep=max_checkpoint_to_keep)
        summ_writer = tf.summary.FileWriter(f'{self._checkpoint_dir}/train')
        if graph is not None:
            summ_writer.add_graph(graph)
        if listeners:
            for l in listeners:
                l.begin(self._checkpoint_dir, self._inputs, self._labels, self._is_training, self._logits,
                        steps_per_checkpoint)
        with tf.Session() as sess:
            sess.run(create_init_op())
            if not from_scratch:
                load_ckpt(sess, model_dir=self._checkpoint_dir)
            global_step_value = sess.run(global_step)
            id_checkpoints = global_step_value // steps_per_checkpoint
            for _ in range(num_checkpoints):
                bar = Bar(f'Checkpoint {id_checkpoints+1}', max=steps_per_checkpoint,
                          suffix='%(index)d/%(max)d ETA: %(eta)d s')
                for _ in range(steps_per_checkpoint):
                    xb, yb = sess.run(data_gntr)
                    bar.next()
                    fd = {self._inputs: xb, self._labels: yb, self._is_training: True}
                    if global_step_value == 0:
                        sess.run(update_ops, feed_dict=fd)
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=global_step_value)
                        sess.run(reset_ops)
                    sess.run(train_op, feed_dict=fd)
                    global_step_value = sess.run(global_step)
                    if global_step_value % summ_steps == 0:
                        sess.run(update_ops, feed_dict=fd)
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=global_step_value)
                        sess.run(reset_ops)

                id_checkpoints = global_step_value // steps_per_checkpoint
                summ_writer.flush()
                bar.finish()
                saver.save(sess, f'{self._checkpoint_dir}/model.ckpt', global_step_value, write_meta_graph=False)
                if listeners:
                    for l in listeners:
                        l.run_stepwise(sess, global_step_value)

            if listeners:
                for l in listeners:
                    l.end()

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

    def eval(self, metric_opdefs, gnte):
        data_gn, data_gn_init_op, steps_per_epoch = gnte
        metric_ops, update_ops, reset_ops, _, _ = list(zip(*metric_opdefs))
        logits_lst = []
        if not self._model_loaded:
            raise RuntimeError('Load model first!')
        sess = self._sess
        sess.run(data_gn_init_op)
        sess.run(reset_ops)
        bar = Bar('Test evaluation', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for _ in range(steps_per_epoch):
            bar.next()
            xb, yb = sess.run(data_gn)
            fd = {self._inputs: xb, self._labels: yb, self._is_training: False}
            logits_val, _ = sess.run([self._logits, update_ops], feed_dict=fd)
            logits_lst.append(logits_val)
        bar.finish()
        logits = np.concatenate(logits_lst, axis=0)
        metrics = sess.run(metric_ops)
        return logits, metrics

    def eval_from_array(self, metric_opdefs, x, y, batch_size=None):
        data_gn, data_gn_init_op, steps_per_epoch = read_tfrec_array((x, y), batch_size=batch_size, is_test=True)
        metric_ops, update_ops, reset_ops, _, _ = list(zip(*metric_opdefs))
        logits_lst = []
        if not self._model_loaded:
            raise RuntimeError('Load model first!')
        sess = self._sess
        sess.run(data_gn_init_op)
        sess.run(reset_ops)
        bar = Bar('Test evaluation', max=steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for _ in range(steps_per_epoch):
            bar.next()
            xb, yb = sess.run(data_gn)
            fd = {self._inputs: xb, self._labels: yb, self._is_training: False}
            logits_val, _ = sess.run([self._logits, update_ops], feed_dict=fd)
            logits_lst.append(logits_val)
        bar.finish()
        logits = np.concatenate(logits_lst, axis=0)
        metrics = sess.run(metric_ops)
        return logits, metrics

    def saliency_map_single(self, saliency_class, seed_x=None, niter=100, step=0.1):
        if seed_x is None:
            seed_x = np.zeros(shape=[1, *self._inputs.get_shape().as_list()[1:]], dtype=np.float32)
        sa_grads = saliency_grad(self._inputs, self._logits, saliency_class)
        x = np.copy(seed_x)
        probs = tf.nn.softmax(self._logits)
        sess = self._sess
        for i in range(niter):
            grads_val = sess.run(sa_grads, feed_dict={self._inputs: x, self._is_training: False})
            x += step * grads_val
            logits_value, probs_value = sess.run([self._logits, probs],
                                                 feed_dict={self._inputs: x, self._is_training: False})
            print(f'Iter: {i+1}')
            print(logits_value)
            print(probs_value)
        return x[0]

    def saliency_map_all(self, seed_x=None, niter=100, step=0.1):
        num_classes = self._logits.get_shape().as_list()[-1]
        if seed_x is None:
            seed_x = np.zeros(shape=[num_classes, *self._inputs.get_shape().as_list()[1:]], dtype=np.float32)
        probs = tf.nn.softmax(self._logits)
        sess = self._sess
        for c in range(num_classes):
            sa_grads = saliency_grad(self._inputs, self._logits, c)
            x = np.copy(seed_x[c])[None]
            for i in range(niter):
                grads_val = sess.run(sa_grads, feed_dict={self._inputs: x, self._is_training: False})
                x += step * grads_val
                logits_value, probs_value = sess.run([self._logits, probs],
                                                     feed_dict={self._inputs: x, self._is_training: False})
                print(f'Iter: {i+1}')
                print(logits_value)
                print(probs_value)
            seed_x[c] = x[0]
        logits_value, probs_value = sess.run([self._logits, probs],
                                             feed_dict={self._inputs: seed_x, self._is_training: False})
        print('Final result:')
        print(logits_value)
        print(probs_value)
        return seed_x
