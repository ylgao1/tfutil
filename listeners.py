import tensorflow as tf
from progress.bar import Bar

__all__ = ['Listener']

class Listener:
    def __init__(self, name, data_gn, metric_opdefs):
        self._name = name
        self._data_gn, self._data_initializer, self._steps_per_epoch = data_gn
        self._metric_ops, self._update_ops, self._reset_ops = list(zip(*metric_opdefs))
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
        self._summ_names = ['{0}/{1}'.format(self._name, m.name.split('/')[-2]) for m in self._metric_ops]
        self._summ_op = tf.summary.merge(
            [tf.summary.scalar(*tup) for tup in list(zip(self._summ_names, self._metric_ops))])

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
        metrics, summ = session.run([self._metric_ops, self._summ_op])
        for name, metric in zip(self._summ_names, metrics):
            print(f'{name}: {metric}')
        self._fw.add_summary(summ, global_step=epoch)
        self._fw.flush()

    def end(self):
        self._fw.close()
