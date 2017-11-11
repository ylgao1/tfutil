import tensorflow as tf
import numpy as np
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _parse_tfrec(shape, is_pred=False, is_reg=False):
    if is_pred:
        def _parse_function(example_proto):
            dataset = tf.parse_single_example(example_proto, features={
                'feature': tf.FixedLenFeature([], dtype=tf.string),
            })
            feature = tf.decode_raw(dataset['feature'], tf.float32)
            feature = tf.reshape(feature, shape)
            return feature
    else:
        if is_reg:
            def _parse_function(example_proto):
                dataset = tf.parse_single_example(example_proto, features={
                    'feature': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([], dtype=tf.float32)
                })
                feature = tf.decode_raw(dataset['feature'], tf.float32)
                feature = tf.reshape(feature, shape)
                label = dataset['label']
                return feature, label
        else:
            def _parse_function(example_proto):
                dataset = tf.parse_single_example(example_proto, features={
                    'feature': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([], dtype=tf.int64)
                })
                feature = tf.decode_raw(dataset['feature'], tf.float32)
                feature = tf.reshape(feature, shape)
                label = tf.cast(dataset['label'], tf.int32)
                return feature, label
    return _parse_function


def _parse_raw_tfimgrec(is_pred=False, is_reg=False):
    if is_pred:
        def _parse_function(example_proto):
            dataset = tf.parse_single_example(example_proto, features={
                'feature': tf.FixedLenFeature([], dtype=tf.string),
                'height': tf.FixedLenFeature([], dtype=tf.int64),
                'width': tf.FixedLenFeature([], dtype=tf.int64)
            })
            feature = tf.decode_raw(dataset['feature'], tf.float32)
            height = tf.cast(dataset['height'], tf.int32)
            width = tf.cast(dataset['width'], tf.int32)
            return feature, height, width
    else:
        if is_reg:
            def _parse_function(example_proto):
                dataset = tf.parse_single_example(example_proto, features={
                    'feature': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([], dtype=tf.float32),
                    'height': tf.FixedLenFeature([], dtype=tf.int64),
                    'width': tf.FixedLenFeature([], dtype=tf.int64)
                })
                feature = tf.decode_raw(dataset['feature'], tf.float32)
                height = tf.cast(dataset['height'], tf.int32)
                width = tf.cast(dataset['width'], tf.int32)
                label = dataset['label']
                return feature, height, width, label
        else:
            def _parse_function(example_proto):
                dataset = tf.parse_single_example(example_proto, features={
                    'feature': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([], dtype=tf.int64),
                    'height': tf.FixedLenFeature([], dtype=tf.int64),
                    'width': tf.FixedLenFeature([], dtype=tf.int64)
                })
                feature = tf.decode_raw(dataset['feature'], tf.float32)
                height = tf.cast(dataset['height'], tf.int32)
                width = tf.cast(dataset['width'], tf.int32)
                label = tf.cast(dataset['label'], tf.int32)
                return feature, height, width, label
    return _parse_function


def _resize_img(feature, height, width, channels, resize, resize_enlarge, normalization):
    feature = tf.reshape(feature, [height, width, channels])
    if resize_enlarge:
        resize_method = tf.image.ResizeMethod.BICUBIC
    else:
        resize_method = tf.image.ResizeMethod.BILINEAR
    feature = tf.image.resize_images(feature, resize, resize_method)
    if normalization:
        feature = tf.image.per_image_standardization(feature)
    return feature


def _parse_tfimgrec(channels, resize, resize_enlarge, normalization, is_pred=False, is_reg=False):
    if is_pred:
        def _parse_function(example_proto):
            dataset = tf.parse_single_example(example_proto, features={
                'feature': tf.FixedLenFeature([], dtype=tf.string),
                'height': tf.FixedLenFeature([], dtype=tf.int64),
                'width': tf.FixedLenFeature([], dtype=tf.int64)
            })
            feature = tf.decode_raw(dataset['feature'], tf.float32)
            height = tf.cast(dataset['height'], tf.int32)
            width = tf.cast(dataset['width'], tf.int32)
            resized_feature = _resize_img(feature, height, width, channels, resize, resize_enlarge, normalization)
            return resized_feature
    else:
        if is_reg:
            def _parse_function(example_proto):
                dataset = tf.parse_single_example(example_proto, features={
                    'feature': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([], dtype=tf.float32),
                    'height': tf.FixedLenFeature([], dtype=tf.int64),
                    'width': tf.FixedLenFeature([], dtype=tf.int64)
                })
                feature = tf.decode_raw(dataset['feature'], tf.float32)
                height = tf.cast(dataset['height'], tf.int32)
                width = tf.cast(dataset['width'], tf.int32)
                label = dataset['label']
                resized_feature = _resize_img(feature, height, width, channels, resize, resize_enlarge, normalization)
                return resized_feature, label
        else:
            def _parse_function(example_proto):
                dataset = tf.parse_single_example(example_proto, features={
                    'feature': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([], dtype=tf.int64),
                    'height': tf.FixedLenFeature([], dtype=tf.int64),
                    'width': tf.FixedLenFeature([], dtype=tf.int64)
                })
                feature = tf.decode_raw(dataset['feature'], tf.float32)
                height = tf.cast(dataset['height'], tf.int32)
                width = tf.cast(dataset['width'], tf.int32)
                label = tf.cast(dataset['label'], tf.int32)
                resized_feature = _resize_img(feature, height, width, channels, resize, resize_enlarge, normalization)
                return resized_feature, label
    return _parse_function


def tfrec_name(prefix, ds_shape, num_classes, index=None):
    ds_str = '_'.join(map(str, ds_shape))
    if index is None:
        return f'{prefix}.{ds_str}.{num_classes}.tfrec'
    else:
        return f'{prefix}.{ds_str}.{num_classes}.tfrec{index}'


def tfrec_pred_name(prefix, ds_shape, index=None):
    ds_str = '_'.join(map(str, ds_shape))
    if index is None:
        return f'{prefix}.{ds_str}.tfrec'
    else:
        return f'{prefix}.{ds_str}.tfrec{index}'


def tfimgrec_name(prefix, num_examples, channels, num_classes, index=None):
    if index is None:
        return f'{prefix}.{num_examples}_{channels}.{num_classes}.tfimgrec'
    else:
        return f'{prefix}.{num_examples}_{channels}.{num_classes}.tfimgrec{index}'


def tfimgrec_pred_name(prefix, num_examples, channels, index=None):
    if index is None:
        return f'{prefix}.{num_examples}_{channels}.tfimgrec'
    else:
        return f'{prefix}.{num_examples}_{channels}.tfimgrec{index}'


def rec_names(raw_name, num_splits):
    return [f'{raw_name}{i}' for i in range(num_splits)]


def parse_tfrec_name(fname):
    name_parts = os.path.basename(fname).split('.')
    with_label = True
    if name_parts[-1][:8] == 'tfimgrec':
        if len(name_parts) == 4:
            _, shape_str, num_classes_str, _ = name_parts
            num_examples_str, channels_str = shape_str.split('_')
            num_examples = int(num_examples_str)
            channels = int(channels_str)
            num_classes = int(num_classes_str)
            return (with_label, num_examples, channels, num_classes)
        else:
            _, shape_str, _ = name_parts
            num_examples_str, channels_str = shape_str.split('_')
            num_examples = int(num_examples_str)
            channels = int(channels_str)
            with_label = False
            return (with_label, num_examples, channels)
    else:
        if len(name_parts) == 4:
            _, ds_shape_str, num_classes_str, _ = name_parts
            ds_shape = list(map(int, ds_shape_str.split('_')))
            num_classes = int(num_classes_str)
            return (with_label, ds_shape, num_classes)
        else:
            _, ds_shape_str, _ = name_parts
            ds_shape = list(map(int, ds_shape_str.split('_')))
            with_label = False
            return (with_label, ds_shape)


def read_tfrec(filenames, batch_size, num_epochs=None, shuffle=True):
    if isinstance(filenames, str):
        filenames = [filenames]
    filename_parsed = parse_tfrec_name(filenames[0])
    ds_shape = filename_parsed[1]
    is_pred = not filename_parsed[0]
    is_reg = False
    if not is_pred and filename_parsed[-1] == 0:
        is_reg = True
    dataset = tf.data.TFRecordDataset(filenames)
    ds = dataset.map(_parse_tfrec(ds_shape[1:], is_pred, is_reg))
    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size * 3)
    ds = ds.batch(batch_size)
    ds = ds.repeat(num_epochs)
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next()


def read_raw_tfimgrec(filenames, num_epochs=None):
    if isinstance(filenames, str):
        filenames = [filenames]
    filename_parsed = parse_tfrec_name(filenames[0])
    is_pred = not filename_parsed[0]
    is_reg = False
    if not is_pred and filename_parsed[-1] == 0:
        is_reg = True
    dataset = tf.data.TFRecordDataset(filenames)
    ds = dataset.map(_parse_raw_tfimgrec(is_pred, is_reg))
    ds = ds.repeat(num_epochs)
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next()


def read_tfimgrec(filenames, shape, batch_size, num_epochs=None, shuffle=True, shape_enlarge=True,
                  normalization=True):
    if isinstance(filenames, str):
        filenames = [filenames]
    filename_parsed = parse_tfrec_name(filenames[0])
    channels = filename_parsed[2]
    is_pred = not filename_parsed[0]
    is_reg = False
    if not is_pred and filename_parsed[-1] == 0:
        is_reg = True
    dataset = tf.data.TFRecordDataset(filenames)
    ds = dataset.map(_parse_tfimgrec(channels, shape, shape_enlarge, normalization, is_pred, is_reg))
    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size * 3)
    ds = ds.batch(batch_size)
    ds = ds.repeat(num_epochs)
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next()


def write_tfrec_from_array(arr_x, arr_y, prefix, num_classes, num_examples_per_file=None):
    arr_x = arr_x.astype(np.float32)
    arr_y = arr_y.astype(np.float32) if num_classes == 0 else arr_y.astype(np.int64)
    ds_shape = arr_x.shape
    num_examples = ds_shape[0]
    if num_examples_per_file is None or num_examples_per_file >= num_examples:
        filename = tfrec_name(prefix, ds_shape, num_classes)
        writer = tf.python_io.TFRecordWriter(filename)
        for idx in range(num_examples):
            print(f'{idx+1} / {num_examples}')
            x_raw = arr_x[idx].tostring()
            label = arr_y[idx]
            if num_classes == 0:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                    'label': _float_feature(label)
                }))
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                    'label': _int64_feature(label)
                }))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        num_splits = num_examples // num_examples_per_file
        res = num_examples % num_examples_per_file
        idx = -1
        for i in range(num_splits):
            filename = tfrec_name(prefix, ds_shape, num_classes, index=i)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(num_examples_per_file):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x_raw = arr_x[idx].tostring()
                label = arr_y[idx]
                if num_classes == 0:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _float_feature(label)
                    }))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _int64_feature(label)
                    }))
                writer.write(example.SerializeToString())
            writer.close()
        if res != 0:
            filename = tfrec_name(prefix, ds_shape, num_classes, index=num_splits)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(res):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x_raw = arr_x[idx].tostring()
                label = arr_y[idx]
                if num_classes == 0:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _float_feature(label)
                    }))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _int64_feature(label)
                    }))
                writer.write(example.SerializeToString())
            writer.close()


def write_tfrec_pred_from_array(arr_x, prefix, num_examples_per_file=None):
    arr_x = arr_x.astype(np.float32)
    ds_shape = arr_x.shape
    num_examples = ds_shape[0]
    if num_examples_per_file is None or num_examples_per_file >= num_examples:
        filename = tfrec_pred_name(prefix, ds_shape)
        writer = tf.python_io.TFRecordWriter(filename)
        for idx in range(num_examples):
            print(f'{idx+1} / {num_examples}')
            x_raw = arr_x[idx].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'feature': _bytes_feature(x_raw),
            }))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        num_splits = num_examples // num_examples_per_file
        res = num_examples % num_examples_per_file
        idx = -1
        for i in range(num_splits):
            filename = tfrec_pred_name(prefix, ds_shape, index=i)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(num_examples_per_file):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x_raw = arr_x[idx].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                }))
                writer.write(example.SerializeToString())
            writer.close()
        if res != 0:
            filename = tfrec_pred_name(prefix, ds_shape, index=num_splits)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(res):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x_raw = arr_x[idx].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                }))
                writer.write(example.SerializeToString())
            writer.close()


def write_tfimgrec_from_lst(x_lst, arr_y, prefix, num_classes, num_examples_per_file=None):
    arr_y = arr_y.astype(np.float32) if num_classes == 0 else arr_y.astype(np.int64)
    num_examples = len(x_lst)
    channels = x_lst[0].shape[-1]
    if num_examples_per_file is None or num_examples_per_file >= num_examples:
        filename = tfimgrec_name(prefix, num_examples, channels, num_classes)
        writer = tf.python_io.TFRecordWriter(filename)
        for idx in range(num_examples):
            print(f'{idx+1} / {num_examples}')
            x = x_lst[idx].astype(np.float32)
            x_raw = x.tostring()
            label = arr_y[idx]
            height, width = x.shape[:2]
            if num_classes == 0:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                    'label': _float_feature(label),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width)
                }))
            else:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                    'label': _int64_feature(label),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width)
                }))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        num_splits = num_examples // num_examples_per_file
        res = num_examples % num_examples_per_file
        idx = -1
        for i in range(num_splits):
            filename = tfimgrec_name(prefix, num_examples, channels, num_classes, index=i)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(num_examples_per_file):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x = x_lst[idx].astype(np.float32)
                x_raw = x.tostring()
                label = arr_y[idx]
                height, width = x.shape[:2]
                if num_classes == 0:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _float_feature(label),
                        'height': _int64_feature(height),
                        'width': _int64_feature(width)
                    }))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _int64_feature(label),
                        'height': _int64_feature(height),
                        'width': _int64_feature(width)
                    }))
                writer.write(example.SerializeToString())
            writer.close()
        if res != 0:
            filename = tfimgrec_name(prefix, num_examples, channels, num_classes, index=num_splits)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(res):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x = x_lst[idx].astype(np.float32)
                x_raw = x.tostring()
                label = arr_y[idx]
                height, width = x.shape[:2]
                if num_classes == 0:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _float_feature(label),
                        'height': _int64_feature(height),
                        'width': _int64_feature(width)
                    }))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _int64_feature(label),
                        'height': _int64_feature(height),
                        'width': _int64_feature(width)
                    }))
                writer.write(example.SerializeToString())
            writer.close()


def write_tfimgrec_pred_from_lst(x_lst, prefix, num_examples_per_file=None):
    num_examples = len(x_lst)
    channels = x_lst[0].shape[-1]
    if num_examples_per_file is None or num_examples_per_file >= num_examples:
        filename = tfimgrec_pred_name(prefix, num_examples, channels)
        writer = tf.python_io.TFRecordWriter(filename)
        for idx in range(num_examples):
            print(f'{idx+1} / {num_examples}')
            x = x_lst[idx].astype(np.float32)
            x_raw = x.tostring()
            height, width = x.shape[:2]
            example = tf.train.Example(features=tf.train.Features(feature={
                'feature': _bytes_feature(x_raw),
                'height': _int64_feature(height),
                'width': _int64_feature(width)
            }))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        num_splits = num_examples // num_examples_per_file
        res = num_examples % num_examples_per_file
        idx = -1
        for i in range(num_splits):
            filename = tfimgrec_pred_name(prefix, num_examples, channels, index=i)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(num_examples_per_file):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x = x_lst[idx].astype(np.float32)
                x_raw = x.tostring()
                height, width = x.shape[:2]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width)
                }))
                writer.write(example.SerializeToString())
            writer.close()
        if res != 0:
            filename = tfimgrec_pred_name(prefix, num_examples, channels, index=num_splits)
            writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(res):
                idx += 1
                print(f'{idx + 1} / {num_examples}')
                x = x_lst[idx].astype(np.float32)
                x_raw = x.tostring()
                height, width = x.shape[:2]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _bytes_feature(x_raw),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width)
                }))
                writer.write(example.SerializeToString())
            writer.close()


def _convert_to_real_name(temp_filename, num_examples):
    return temp_filename.replace('000', f'{num_examples}')


def write_tfrec_from_generator(gn, prefix, num_classes, num_examples_per_file=None):
    if num_examples_per_file is None:
        x, y = next(gn)
        num_examples = 1
        ds_shape = ['000', *x.shape]
        temp_filename = tfrec_name(prefix, ds_shape, num_classes)
        writer = tf.python_io.TFRecordWriter(temp_filename)
        while True:
            try:
                x_raw = x.astype(np.float32)
                label = y.astype(np.float32) if num_classes == 0 else y.astype(np.int64)
                if num_classes == 0:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _float_feature(label)
                    }))
                else:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'feature': _bytes_feature(x_raw),
                        'label': _int64_feature(label)
                    }))
                writer.write(example.SerializeToString())
                x, y = next(gn)
                num_examples += 1
            except StopIteration:
                break
        writer.close()
        filename = _convert_to_real_name(temp_filename, num_examples)
        os.rename(temp_filename, filename)











