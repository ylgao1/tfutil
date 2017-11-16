import tensorflow as tf
from .metrics import *


class MultiClsTestListerner(tf.train.CheckpointSaverListener):
    def __init__(self, logdir, data_gn, inputs, targets, is_training, logits, steps_per_epoch):
        super(MultiClsTestListerner, self).__init__()
        self._logdir = logdir
        self._inputs = inputs
        self._targets = targets
        self._is_traing = is_training
        self._logits = logits
        self._data_gn, self._data_initializer = data_gn
        self.fw = None
        self.reset_op = []
        self.update_op = []
        self.summ_op = None
        self.acc_pl = tf.placeholder(dtype=tf.float32)
        self._steps_per_epoch = steps_per_epoch

    def begin(self):
        self.fw = tf.summary.FileWriter(self._logdir)
        _, acc_update_op, acc_reset_op, _ = metrics_accuracy(self._targets, tf.argmax(self._logits, axis=1))
        self.reset_op.append(acc_reset_op)
        self.update_op.append(acc_update_op)
        summ_acc_te = tf.summary.scalar('test/accuracy', self.acc_pl)
        self.summ_op = tf.summary.merge([summ_acc_te])

    def before_save(self, session, global_step_value):
        session.run(self.reset_op)
        session.run(self._data_initializer)

    def after_save(self, session, global_step_value):
        metrics = None
        while True:
            try:
                xb, yb = session.run(self._data_gn)
                metrics = session.run(self.update_op,
                                      feed_dict={self._inputs: xb, self._targets: yb, self._is_traing: False})
            except tf.errors.OutOfRangeError:
                break
        step = global_step_value // self._steps_per_epoch
        summ = session.run(self.summ_op, feed_dict={self.acc_pl: metrics[0]})
        print(f'\nEpoch {step}')
        print(f'Accuracy: {metrics[0]}')
        self.fw.add_summary(summ, global_step=step)
        self.fw.flush()

    def end(self, session, global_step_value):
        self.fw.close()


class BinaryClsTestListerner(tf.train.CheckpointSaverListener):
    def __init__(self, logdir, data_gn, inputs, targets, is_training, logits, steps_per_epoch):
        super(BinaryClsTestListerner, self).__init__()
        self._logdir = logdir
        self._inputs = inputs
        self._targets = targets
        self._is_traing = is_training
        self._logits = logits
        self._data_gn, self._data_initializer = data_gn
        self.fw = None
        self.reset_op = []
        self.update_op = []
        self.summ_op = None
        self.acc_pl = tf.placeholder(dtype=tf.float32)
        self.roc_pl = tf.placeholder(dtype=tf.float32)
        self.pr_pl = tf.placeholder(dtype=tf.float32)
        self._steps_per_epoch = steps_per_epoch

    def begin(self):
        self.fw = tf.summary.FileWriter(self._logdir)
        probabilities = tf.nn.softmax(self._logits)
        _, acc_update_op, acc_reset_op, _ = metrics_accuracy(self._targets, tf.argmax(probabilities, axis=1))
        _, roc_update_op, roc_reset_op, _ = metrics_auc(self._targets, probabilities[:, 1], curve='ROC')
        _, pr_update_op, pr_reset_op, _ = metrics_auc(self._targets, probabilities[:, 1], curve='PR')
        self.reset_op.extend([acc_reset_op, roc_reset_op, pr_reset_op])
        self.update_op.extend([acc_update_op, roc_update_op, pr_update_op])
        summ_acc_te = tf.summary.scalar('test/accuracy', self.acc_pl)
        summ_roc_te = tf.summary.scalar('test/roc', self.roc_pl)
        summ_pr_te = tf.summary.scalar('test/pr', self.pr_pl)
        self.summ_op = tf.summary.merge([summ_acc_te, summ_roc_te, summ_pr_te])

    def before_save(self, session, global_step_value):
        session.run(self.reset_op)
        session.run(self._data_initializer)

    def after_save(self, session, global_step_value):
        metrics = None
        while True:
            try:
                xb, yb = session.run(self._data_gn)
                metrics = session.run(self.update_op,
                                      feed_dict={self._inputs: xb, self._targets: yb, self._is_traing: False})
            except tf.errors.OutOfRangeError:
                break
        step = global_step_value // self._steps_per_epoch
        metrics_fd = {k:v for k, v in zip([self.acc_pl, self.roc_pl, self.pr_pl], metrics)}
        summ = session.run(self.summ_op, feed_dict=metrics_fd)
        print(f'\nEpoch {step}')
        print(f'Accuracy: {metrics[0]}')
        print(f'ROC-AUC: {metrics[1]}')
        print(f'PR-AUC: {metrics[2]}')
        self.fw.add_summary(summ, global_step=step)
        self.fw.flush()

    def end(self, session, global_step_value):
        self.fw.close()
