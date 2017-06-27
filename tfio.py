import tensorflow as tf
import numpy as np
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def tfrec_name(prefix, ds_shape, num_classes):
    ds_str = '_'.join(map(str, ds_shape))
    return f'{prefix}.{ds_str}.{num_classes}.tfrec'


def tfrec_name_var(prefix, size, channels, num_classes):
    return f'{prefix}.{size}_{channels}.{num_classes}.tfrec'


def parse_tfrec_name(fname):
    _, ds_shape_str, num_classes_str, _ = os.path.basename(fname).split('.')
    ds_shape = list(map(int, ds_shape_str.split('_')))
    num_classes = int(num_classes_str)
    return ds_shape, num_classes


def parse_img_tfrec_name(fname):
    _, shape_str, num_classes_str, _ = os.path.basename(fname).split('.')
    size_str, channels_str = shape_str.split('_')
    size = int(size_str)
    channels = int(channels_str)
    num_classes = int(num_classes_str)
    return size, channels, num_classes


def write_tfrec(arr_x, arr_y, prefix, num_classes):
    arr_x = arr_x.astype(np.float32)
    arr_y = arr_y.astype(np.int64)
    ds_shape = arr_x.shape
    num_example = ds_shape[0]
    filename = tfrec_name(prefix, ds_shape, num_classes)
    writer = tf.python_io.TFRecordWriter(filename)
    for idx in range(num_example):
        print(f'{idx+1} / {num_example}')
        x_raw = arr_x[idx].tostring()
        label = arr_y[idx]
        example = tf.train.Example(features=tf.train.Features(feature={
            'feature': _bytes_feature(x_raw),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def write_img_tfrec(x_lst, arr_y, prefix, num_classes):
    arr_y = arr_y.astype(np.int64)
    size = len(x_lst)
    channels = x_lst[0].shape[-1]
    filename = tfrec_name_var(prefix, size, channels, num_classes)
    writer = tf.python_io.TFRecordWriter(filename)
    for idx in range(size):
        print(f'{idx+1} / {size}')
        x = x_lst[idx].astype(np.float32)
        x_raw = x.tostring()
        label = arr_y[idx]
        height, width = x.shape[:2]
        example = tf.train.Example(features=tf.train.Features(feature={
            'feature': _bytes_feature(x_raw),
            'label': _int64_feature(label),
            'height': _int64_feature(height),
            'width': _int64_feature(width)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrec(fname):
    ds_shape, num_classes = parse_tfrec_name(fname)
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

    return feature, label, ds_shape, num_classes


def read_test_tfrec(fname):
    feature, label, ds_shape, _ = read_tfrec(fname)
    size = ds_shape[0]
    feature = feature[None]
    return feature, label, size


def read_img_tfrec(fname):
    size, channels, num_classes = parse_img_tfrec_name(fname)
    fn_q = tf.train.string_input_producer([fname])
    reader = tf.TFRecordReader()
    _, sdata = reader.read(fn_q)
    dataset = tf.parse_single_example(sdata, features={
        'feature': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'height': tf.FixedLenFeature([], dtype=tf.int64),
        'width': tf.FixedLenFeature([], dtype=tf.int64)
    })
    feature = tf.decode_raw(dataset['feature'], tf.float32)
    label = tf.cast(dataset['label'], tf.int32)
    height = tf.cast(dataset['height'], tf.int32)
    width = tf.cast(dataset['width'], tf.int32)
    feature = tf.reshape(feature, [height, width, channels])
    return feature, label, size, channels, num_classes


def read_test_img_tfrec(fname, resize=None, resize_enlarge=True, normalization=True):
    feature, label, size, _, _ = read_img_tfrec(fname)
    if resize is not None:
        if resize_enlarge:
            resize_method = tf.image.ResizeMethod.BICUBIC
        else:
            resize_method = tf.image.ResizeMethod.BILINEAR
        feature = tf.image.resize_images(feature, resize, resize_method)
    if normalization:
        feature = tf.image.per_image_standardization(feature)
    feature = feature[None]
    return feature, label, size


def read_tfrec_batch(fname, batch_size=32, shuffle=True, min_frac_in_q=0.2, num_threads=3):
    feature, label, ds_shape, num_classes = read_tfrec(fname)
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
    return features, labels, ds_shape, num_classes


def read_img_tfrec_batch(fname, batch_size=32, shuffle=True, resize=None, resize_enlarge=True, normalization=True,
                         min_frac_in_q=0.2, num_threads=3):
    feature, label, size, channels, num_classes = read_img_tfrec(fname)
    min_queue_examples = int(size * min_frac_in_q)
    capacity = min_queue_examples + 3 * batch_size
    if resize is not None:
        if resize_enlarge:
            resize_method = tf.image.ResizeMethod.BICUBIC
        else:
            resize_method = tf.image.ResizeMethod.BILINEAR
        feature = tf.image.resize_images(feature, resize, resize_method)
    if normalization:
        feature = tf.image.per_image_standardization(feature)

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
    return features, labels, size, channels, num_classes
