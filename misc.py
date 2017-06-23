import tensorflow as tf
import numpy as np

def cal_num_parameters():
    return np.sum(list(map(lambda tv: np.prod(tv.get_shape().as_list()), tf.trainable_variables())))