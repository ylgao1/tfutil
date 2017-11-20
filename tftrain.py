import tensorflow as tf
from tensorflow.contrib import slim
from tempfile import NamedTemporaryFile
from tensorflow.python.tools import freeze_graph
import collections
from .listeners import MultiClsTestListerner, BinaryClsTestListerner
from .misc import delete_and_make_dir
from .tfio import read_tfrec_array
from progress.bar import Bar
import numpy as np


def loss(logits, label_pl, is_one_hot=False, scope=None):
    with tf.name_scope(scope):
        if is_one_hot:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_pl, logits=logits))
        else:
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_pl, logits=logits))
        regularization_loss_lst = tf.losses.get_regularization_losses()
        if len(regularization_loss_lst) == 0:
            regularization_loss = 0
        else:
            regularization_loss = tf.add_n(regularization_loss_lst)
        total_loss = cross_entropy + regularization_loss
    return total_loss


def loss_with_aux(logits, aux_logits, label_pl, aux_weight=0.4, is_one_hot=False, scope=None):
    with tf.name_scope(scope):
        if is_one_hot:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_pl, logits=logits))
            aux_cross_entropy = aux_weight * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label_pl, logits=aux_logits))
        else:
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_pl, logits=logits))
            aux_cross_entropy = aux_weight * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_pl, logits=aux_logits))
        regularization_loss_lst = tf.losses.get_regularization_losses()
        if len(regularization_loss_lst) == 0:
            regularization_loss = 0
        else:
            regularization_loss = tf.add_n(regularization_loss_lst)
        total_loss = cross_entropy + aux_cross_entropy + regularization_loss
    return total_loss


def masked_sigmoid_cross_entropy(logits, target, mask, scope=None):
    """Time major"""
    with tf.name_scope(scope):
        xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
        loss_time_batch = tf.reduce_sum(xent, axis=2)
        loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)
        loss = tf.reduce_mean(loss_batch)
    return loss


def seq_sigmoid_cross_entropy(logits, target, scope=None):
    """Time major"""
    with tf.name_scope(scope):
        xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
        loss_time_batch = tf.reduce_sum(xent, axis=2)
        loss_batch = tf.reduce_sum(loss_time_batch, axis=0)
        loss = tf.reduce_mean(loss_batch)
    return loss


def create_init_op():
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return init_op


def create_train_op(total_loss, optimizer, max_grad_norm=None):
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if max_grad_norm is None:
            train_op = optimizer.minimize(total_loss, global_step=global_step)
        else:
            grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tf.trainable_variables()), max_grad_norm)
            train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), global_step=global_step)
    return train_op


def get_all_ckpt(model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    model_path_lst = list(ckpt.all_model_checkpoint_paths)
    return model_path_lst


def load_ckpt_path(sess, model_path, variables_to_restore=None):
    if variables_to_restore is None:
        variables_to_restore = slim.get_variables_to_restore()
    restore_op, restore_fd = slim.assign_from_checkpoint(
        model_path, variables_to_restore)
    sess.run(restore_op, feed_dict=restore_fd)
    print(f'{model_path} loaded')


def load_ckpt(sess, model_dir, variables_to_restore=None):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    model_path = ckpt.model_checkpoint_path
    if variables_to_restore is None:
        variables_to_restore = slim.get_variables_to_restore()
    restore_op, restore_fd = slim.assign_from_checkpoint(
        model_path, variables_to_restore)
    sess.run(restore_op, feed_dict=restore_fd)
    print(f'{model_path} loaded')


def save_model_pb(sess, ckpt_prefix, pb_output_path, output_node_names):
    ckpt_meta_graph_path = f'{ckpt_prefix}.meta'
    with NamedTemporaryFile(mode='w+') as ori_graph_tmp:
        ori_graph_tmp_path = ori_graph_tmp.name
        tf.train.write_graph(sess.graph, '', ori_graph_tmp_path)
        ori_graph_tmp.file.flush()
        freeze_graph.freeze_graph(input_graph=ori_graph_tmp_path,
                                  input_saver=None,
                                  input_binary=False,
                                  input_checkpoint=ckpt_prefix,
                                  output_node_names=output_node_names,
                                  restore_op_name=None,
                                  filename_tensor_name=None,
                                  output_graph=pb_output_path,
                                  clear_devices=None,
                                  initializer_nodes=None,
                                  variable_names_blacklist=ckpt_meta_graph_path)
    return output_node_names


def load_model_pb(pb_path, input_node_name_lst, output_node_name_lst):
    pb_graph_def = tf.GraphDef()
    g = tf.get_default_graph()
    with open(pb_path, 'rb') as f:
        pb_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(pb_graph_def, name='')
    input_nodes = [g.get_tensor_by_name(f'{nm}:0') for nm in input_node_name_lst]
    output_nodes = [g.get_tensor_by_name(f'{nm}:0') for nm in output_node_name_lst]
    return input_nodes, output_nodes


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
