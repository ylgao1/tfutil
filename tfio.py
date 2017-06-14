import tensorflow as tf
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def tfrec_name(prefix, ds_shape, num_class):
    ds_str = '_'.join(map(str, ds_shape))
    return f'{prefix}.{ds_str}.{num_class}.tfrec'


def parse_tfrec_name(fname):
    _, ds_shape_str, y_depth_str, _ = fname.split('.')
    ds_shape = list(map(int, ds_shape_str.split('_')))
    num_class = int(y_depth_str)
    return ds_shape, num_class


def write_tfrec(arr_x, arr_y, prefix, num_class):
    arr_x = arr_x.astype(np.float32)
    arr_y = arr_y.astype(np.int64)
    ds_shape = arr_x.shape
    num_example = ds_shape[0]
    filename = tfrec_name(prefix, ds_shape, num_class)
    writer = tf.python_io.TFRecordWriter(filename)
    for idx in range(num_example):
        x_raw = arr_x[idx].tostring()
        label = arr_y[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'feature': _bytes_feature(x_raw),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrec(fname):
    ds_shape, num_class = parse_tfrec_name(fname)
    fn_q = tf.train.string_input_producer([fname])
    reader = tf.TFRecordReader()
    _, sdata = reader.read(fn_q)
    dataset = tf.parse_single_example(sdata, features={
        'feature': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    })
    feature = tf.decode_raw(dataset['feature'], tf.float32)
    feature = tf.reshape(feature, ds_shape[1:])
    label = tf.cast(dataset['label'], tf.int32)

    return feature, label, ds_shape, num_class


def read_tfrec_batch(fname, batch_size=32, shuffle=True, min_frac_in_q=0.2, num_threads=3):
    feature, label, ds_shape, num_class = read_tfrec(fname)
    num_example = ds_shape[0]
    min_queue_examples = int(num_example * min_frac_in_q)
    capacity = min_queue_examples + 3 * batch_size
    if shuffle:
        features, labels = tf.train.shuffle_batch([feature, label],
                                        batch_size=batch_size,
                                        num_threads=num_threads,
                                        capacity=capacity,
                                        min_after_dequeue=min_queue_examples)
    else:
        features, labels = tf.train.batch([feature, label],
                                batch_size=batch_size,
                                num_threads=num_threads,
                                capacity=capacity)
    return features, labels, ds_shape, num_class












