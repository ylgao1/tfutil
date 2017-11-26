import tensorflow as tf
from tensorflow.contrib import slim
from tempfile import NamedTemporaryFile
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
