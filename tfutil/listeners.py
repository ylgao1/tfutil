import tensorflow as tf
import numpy as np
from progress.bar import Bar

from .metrics import get_metric_name

__all__ = ['Listener']


class Listener:
    def __init__(self, name, data_gn, metric_opdefs):
        self._name = name
        if len(data_gn) == 3:
            self._data_gn, self._data_initializer, self._steps_per_epoch = data_gn
        elif len(data_gn) == 2:
            self._data_gn, _ = data_gn
        self._metric_ops, self._update_ops, self._reset_ops, self._metric_phs, self._summ_metric_ops = list(
            zip(*metric_opdefs))
        self._checkpoint_dir = None
        self._inputs = None
        self._labels = None
        self._is_training = None
        self._logits = None
        self._training_steps_per_epoch = None
        self._fw = None
        self._summ_names = None
        self._summ_op = None

    def begin(self, checkpoint_dir, inputs, labels, is_training, logits, training_steps_per_epoch):
        self._checkpoint_dir = checkpoint_dir
        self._inputs = inputs
        self._labels = labels
        self._is_training = is_training
        self._logits = logits
        self._training_steps_per_epoch = training_steps_per_epoch
        logdir = f'{self._checkpoint_dir}/{self._name}'
        self._fw = tf.summary.FileWriter(logdir)
        self._summ_names = ['{0}/{1}'.format(self._name, get_metric_name(m)) for m in self._metric_ops]
        self._summ_op = tf.summary.merge(self._summ_metric_ops)

    def run(self, session, global_step_value):
        session.run(self._reset_ops)
        session.run(self._data_initializer)
        epoch = global_step_value // self._training_steps_per_epoch
        bar = Bar(f'{self._name} evaluation {epoch}', max=self._steps_per_epoch,
                  suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for _ in range(self._steps_per_epoch):
            bar.next()
            xb, yb = session.run(self._data_gn)
            fd = {self._inputs: xb, self._labels: yb, self._is_training: False}
            session.run(self._update_ops, feed_dict=fd)
        bar.finish()
        metrics = session.run(self._metric_ops)
        summ = session.run(self._summ_op, feed_dict={m_ph: m for (m_ph, m) in zip(self._metric_phs, metrics)})
        for name, metric in zip(self._summ_names, metrics):
            print(f'{name}: {metric}')
        self._fw.add_summary(summ, global_step=epoch)
        self._fw.flush()

    def run_stepwise(self, session, global_step_value):
        steps_per_checkpoint = self._training_steps_per_epoch
        session.run(self._reset_ops)
        id_checkpoints = global_step_value // steps_per_checkpoint
        bar = Bar(f'{self._name} evaluation {id_checkpoints}', max=steps_per_checkpoint,
                  suffix='%(index)d/%(max)d ETA: %(eta)d s')
        metrics_values = []
        for _ in range(steps_per_checkpoint):
            session.run(self._reset_ops)
            bar.next()
            xb, yb = session.run(self._data_gn)
            fd = {self._inputs: xb, self._labels: yb, self._is_training: False}
            session.run(self._update_ops, feed_dict=fd)
            metrics_values.append(session.run(self._metric_ops))
        bar.finish()
        mean_metrics_values = [np.mean(mv) for mv in zip(*metrics_values)]
        summ = session.run(self._summ_op,
                           feed_dict={m_ph: m for (m_ph, m) in zip(self._metric_phs, mean_metrics_values)})
        for name, metric in zip(self._summ_names, mean_metrics_values):
            print(f'{name}: {metric}')
        self._fw.add_summary(summ, global_step=id_checkpoints)
        self._fw.flush()

    def end(self):
        self._fw.close()
