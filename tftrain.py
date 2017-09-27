import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from collections import namedtuple
from tempfile import NamedTemporaryFile
from progress.bar import Bar
from tensorflow.python.tools import freeze_graph


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


def create_init_op():
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    return init_op


def create_train_op(total_loss, optimizer):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss)
    return train_op


def cal_accuracy(logits, label_pl, top_k=1):
    top_k_op = tf.nn.in_top_k(logits, label_pl, top_k)
    correct_num = tf.reduce_sum(tf.cast(top_k_op, tf.int32), name='correct_num')
    accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32), name='accuracy')
    return correct_num, accuracy

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


training_tensors = namedtuple('training_tensors', ('features', 'labels', 'xpl', 'ypl', 'loss',
                                                   'train_op', 'correct_num', 'accuracy'))
testing_tensors = namedtuple('testing_tensors', ('features', 'labels', 'xpl', 'ypl', 'correct_num'))
predicting_tensors = namedtuple('predicting_tensors', ('features', 'xpl', 'logits'))


def predicting_cls(sess, predicting_t):
    logits_lst = []
    while True:
        try:
            xb = sess.run([predicting_t.features])
            logits = sess.run(predicting_t.logits, feed_dict={predicting_t.xpl: xb})
            logits_lst.append(logits)
        except tf.errors.OutOfRangeError:
            break
    return np.concatenate(logits_lst, axis=0)


def training_cls(sess, training_t, num_examples, batch_size, n_epochs, saver=None, model_path=None):
    loss_value_lst = []
    num_steps_per_epoch = num_examples // batch_size
    for epoch in range(n_epochs):
        loss_value = 0
        correct_num_value = 0
        bar = Bar(f'Epoch {epoch+1}', max=num_steps_per_epoch, suffix='%(index)d/%(max)d ETA: %(eta)d s')
        for i in range(num_steps_per_epoch):
            xb, yb = sess.run([training_t.features, training_t.labels])
            feed_dict = {training_t.xpl: xb, training_t.ypl: yb}
            _, loss_value, correct_num_value, accuracy_value = sess.run([training_t.train_op, training_t.loss,
                                                                         training_t.correct_num, training_t.accuracy],
                                                                        feed_dict=feed_dict)
            bar.next()
        bar.finish()
        print(f'Epoch {epoch+1}: {loss_value}; {correct_num_value} / {batch_size}; {accuracy_value}')
        loss_value_lst.append(loss_value)
        if saver is not None:
            saver.save(sess, model_path, global_step=epoch)


def testing_cls(sess, testing_t, num_examples):
    total_correct_num = 0
    bar = Bar(f'Testing: ', max=num_examples, suffix='%(index)d/%(max)d ETA: %(eta)d s')
    while True:
        try:
            xb, yb = sess.run([testing_t.features, testing_t.labels])
            correct_num_value = sess.run(testing_t.correct_num, feed_dict={testing_t.xpl: xb, testing_t.ypl: yb})
            total_correct_num += correct_num_value
            mini_size = xb.shape[0]
            for _ in range(mini_size):
                bar.next()
        except tf.errors.OutOfRangeError:
            bar.finish()
            break
    acc = total_correct_num / num_examples
    print(f'Accuracy: {acc}')
    return acc
