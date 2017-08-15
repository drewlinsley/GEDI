import math
import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob
from tensorflow.python.ops import control_flow_ops


def apply_crop(image, target, h_min, w_min, h_max, w_max):
    im_size = image.get_shape()
    if len(im_size) > 2:
        channels = []
        for idx in range(int(im_size[-1])):
            channels.append(
                slice_op(image[:, :, idx], h_min, w_min, h_max, w_max))
        out_im = tf.stack(channels, axis=2)
        out_im.set_shape([target[0], target[1], int(im_size[-1])])
        return out_im
    else:
        out_im = slice_op(image, h_min, w_min, h_max, w_max)
        return out_im.set_shape([target[0], target[1]])


def slice_op(image_slice, h_min, w_min, h_max, w_max):
    return tf.slice(
        image_slice, tf.cast(
            tf.concat(axis=0, values=[h_min, w_min]), tf.int32), tf.cast(
            tf.concat(axis=0, values=[h_max, w_max]), tf.int32))


def get_crop_coors(image_size, target_size):
    h_diff = image_size[0] - target_size[0]
    w_diff = image_size[1] - target_size[1]
    ts = tf.constant(
        target_size[0], shape=[2, 1])
    offset_h = tf.cast(
        tf.round(tf.random_uniform([1], minval=0, maxval=h_diff)), tf.int32)
    offset_w = tf.cast(
        tf.round(tf.random_uniform([1], minval=0, maxval=w_diff)), tf.int32)
    return offset_h, ts[0], offset_w, ts[1]


def get_image_size(config):
    im_size = misc.imread(
      glob(config.train_directory + '*' + config.im_ext)[0]).shape
    if len(im_size) == 2:
        im_size = np.hstack((im_size, 3))
    return im_size


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis=axis, num_or_size_splits=x_shape[axis], value=x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis=axis, values=x_rep)


def read_and_decode_single_example(
        filename,
        im_size,
        model_input_shape,
        train,
        max_value=None,
        min_value=None,
        num_channels=2,
        return_filename=False,
        normalize=False):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if return_filename:
        features = tf.parse_single_example(
            serialized_example,
            features={
                   'label': tf.FixedLenFeature([], tf.int64),
                   'image': tf.FixedLenFeature([], tf.string),
                   'filename': tf.FixedLenFeature([], tf.string)
                }
            )
    else:
        features = tf.parse_single_example(
            serialized_example,
            features={
                   'label': tf.FixedLenFeature([], tf.int64),
                   'image': tf.FixedLenFeature([], tf.string),
                }
            )

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.float32)

    if num_channels == 2:
        res_image = tf.reshape(image, np.asarray(im_size)[:num_channels])
        image = tf.cast(repeat_elements(tf.expand_dims(
            res_image, 2), 3, axis=2), tf.float32)
    else:
        # Need to reconstruct channels first then transpose channels
        res_image = tf.reshape(image, np.asarray(im_size)[[2, 0, 1]])
        image = tf.transpose(res_image, [2, 1, 0])
    # image.set_shape(im_size)

    # Set max_value
    if max_value is None:
        max_value = tf.reduce_max(image, keep_dims=True)
    else:
        if not isinstance(max_value, np.ndarray):
            max_value = np.asarray(max_value)
        max_value = max_value[None, None, None]
    if not isinstance(max_value, tf.Tensor):
        # If we have max and min numpys, normalize to global [0, 1]
        if not isinstance(min_value, np.ndarray):
            min_value = np.asarray(min_value)
        min_value = min_value[None, None, None]
        image = (image - min_value) / (max_value - min_value)
    else:
        # Normalize to the max_value
        image /= max_value

    if normalize:  # If we want to ensure all images are [0, 1]
        image /= tf.reduce_max(image, keep_dims=True)
    image = tf.squeeze(image)

    # Insert augmentation and preprocessing here
    if train is not None:
        if 'left_right' in train:
            image = tf.image.random_flip_left_right(image)
        if 'up_down' in train:
            image = tf.image.random_flip_up_down(image)
        if 'rotate' in train:
            image = tf.image.rot90(image, k=np.random.randint(4))
        if 'random_crop' in train:
            image = tf.random_crop(
                image,
                [model_input_shape[0], model_input_shape[1], im_size[2]])
        if 'random_contrast' in train and 'random_brightness' in train:
            image = random_contrast_brightness(image)
        elif 'random_contrast' in train:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif 'random_brightness' in train:
            image = tf.image.random_brightness(image, max_delta=32./255.)
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, model_input_shape[0], model_input_shape[1])
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_shape[0], model_input_shape[1])

    # Make sure to clip values to [0, 1]
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 1.0)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    if return_filename:
        filename = tf.decode_raw(features['filename'], tf.string)
        return image, label, filename
    else:
        return image, label


def random_contrast_brightness(
        image,
        lower=0.5,
        upper=1.5,
        max_delta=32./255.):
    return tf.cond(
        tf.random_uniform([], minval=0, maxval=1) > 0.5,
        lambda: tf.image.random_contrast(image, lower=lower, upper=upper),
        lambda: tf.image.random_brightness(image, max_delta=max_delta),
    )


def random_crop(image, heatmap, im_size, model_input_shape, return_heatmaps):
    h_min, h_max, w_min, w_max = get_crop_coors(
        image_size=im_size, target_size=model_input_shape)
    im = apply_crop(
        image, model_input_shape, h_min, w_min, h_max, w_max)
    if return_heatmaps:
        hm = apply_crop(
            heatmap, model_input_shape, h_min, w_min, h_max, w_max)
        return im, hm
    else:
        return im


def read_and_decode(
        filename_queue,
        im_size,
        model_input_shape,
        train,
        max_value=None,
        min_value=None,
        num_channels=2,
        return_filename=False,
        return_gedi=False,
        normalize=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    }

    if return_filename:
        features['filename'] = tf.FixedLenFeature([], tf.string)

    if return_gedi:
        features['gedi'] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(
        serialized_example,
        features=features)

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.float32)

    if num_channels == 2:
        res_image = tf.reshape(image, np.asarray(im_size)[:num_channels])
        image = tf.cast(repeat_elements(tf.expand_dims(
            res_image, 2), 3, axis=2), tf.float32)
    else:
        # Need to reconstruct channels first then transpose channels
        res_image = tf.reshape(image, np.asarray(im_size)[[2, 0, 1]])
        image = tf.transpose(res_image, [2, 1, 0])
    # image.set_shape(im_size)

    # Set max_value -- From new REPO but not working...
    # if max_value is None:
    #     max_value = tf.reduce_max(image, keep_dims=True)
    # else:
    #     if not isinstance(max_value, np.ndarray):
    #         max_value = np.asarray(max_value)
    #     max_value = max_value[None, None, None]
    # if not isinstance(max_value, tf.Tensor):
    #     # If we have max and min numpys, normalize to global [0, 1]
    #     if not isinstance(min_value, np.ndarray):
    #         min_value = np.asarray(min_value)
    #     min_value = min_value[None, None, None]
    #     image = (image - min_value) / (max_value - min_value)
    # else:
    #     # Normalize to the max_value
    #     image /= max_value

    # Set max_value -- From old REPO. testing now 6/17/17.
    if max_value is None:
        max_value = tf.reduce_max(image, keep_dims=True)
    else:
        max_value = max_value[None, None, None]
    if not isinstance(max_value, tf.Tensor):
        # If we have max and min numpys, normalize to global [0, 1]
        min_value = min_value[None, None, None]
        image = (image - min_value) / (max_value - min_value)
    else:
        # Normalize to the max_value
        image /= max_value

    if normalize:  # If we want to ensure all images are [0, 1]
        image /= tf.reduce_max(image, keep_dims=True)
    image = tf.squeeze(image)  # had to add this 5/11/17... make sure this is OK

    if return_gedi:
        gedi_image = tf.decode_raw(features['gedi'], tf.float32)
        gedi_image = tf.reshape(gedi_image, np.asarray(im_size)[:num_channels])
        gedi_image /= tf.squeeze(max_value)
        gedi_image = tf.cast(repeat_elements(tf.expand_dims(
            gedi_image, 2), 3, axis=2), tf.float32)

    # Insert augmentation and preprocessing here
    if train is not None:
        if 'left_right' in train:
            image = tf.image.random_flip_left_right(image)
            lorr = tf.less(tf.random_uniform([], minval=0, maxval=1.), .5)
            image = tf.cond(
                lorr,
                lambda: tf.image.flip_left_right(image),
                lambda: image)
            gedi_image = tf.cond(
                lorr,
                lambda: tf.image.flip_left_right(gedi_image),
                lambda: gedi_image)
        if 'up_down' in train:
            image = tf.image.flip_up_down(image)
            lorr = tf.less(tf.random_uniform([], minval=0, maxval=1.), .5)
            image = control_flow_ops.cond(
                lorr,
                lambda: tf.image.flip_up_down(image),
                lambda: image)
            gedi_image = control_flow_ops.cond(
                lorr,
                lambda: tf.image.flip_up_down(gedi_image),
                lambda: gedi_image)
        if 'rotate' in train:
            random_rot = tf.squeeze(tf.multinomial(tf.log([[10., 10., 10., 10., 10.]]), 1))
            rotation = tf.gather([0., 45., 90., 180., 270.], random_rot) * tf.constant(math.pi) / 180.
            image = tf.contrib.image.rotate(image, rotation, interpolation='BILINEAR')
            gedi_image = tf.contrib.image.rotate(gedi_image, rotation, interpolation='BILINEAR')
        if 'random_contrast' in train:
            image = tf.image.random_contrast(image, lower=0.0, upper=0.1)
            image = tf.image.random_contrast(gedi_image, lower=0.0, upper=0.1)
        if 'random_brightness' in train:
            image = tf.image.random_brightness(image, max_delta=0.1)
            gedi_image = tf.image.random_brightness(gedi_image, max_delta=0.1)

    if train is not None and 'random_crop' in train:
        image, gedi_image = random_crop(image, gedi_image, im_size, model_input_shape, return_heatmaps=True)
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_shape[0], model_input_shape[1])
        gedi_image = tf.image.resize_image_with_crop_or_pad(
            gedi_image, model_input_shape[0], model_input_shape[1])

    # Make sure to clip values to [0, 1]
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 1.0)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    if return_filename:
        filename = tf.decode_raw(features['filename'], tf.string)
        return image, label, filename
    elif return_gedi:
        return image, label, gedi_image
    else:
        return image, label


def inputs(
        tfrecord_file,
        batch_size,
        im_size,
        model_input_shape,
        train=None,
        num_epochs=None,
        max_value=None,
        min_value=None,
        return_filename=False,
        return_gedi=False,
        normalize=False):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file],
            num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        if return_filename:
            image, label, files = read_and_decode(
                filename_queue=filename_queue,
                im_size=im_size,
                model_input_shape=model_input_shape,
                train=train,
                max_value=max_value,
                min_value=min_value,
                return_filename=return_filename,
                normalize=normalize)

            images, sparse_labels, filenames = tf.train.shuffle_batch(
                [image, label, files],
                batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=1000)

            return images, sparse_labels, filenames
        elif return_gedi:
            image, label, gedi_images = read_and_decode(
                filename_queue=filename_queue,
                im_size=im_size,
                model_input_shape=model_input_shape,
                train=train,
                max_value=max_value,
                min_value=min_value,
                return_filename=return_filename,
                normalize=normalize,
                return_gedi=return_gedi)

            images, sparse_labels, gedi_images = tf.train.shuffle_batch(
                [image, label, gedi_images],
                batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=1000)

            return images, sparse_labels, gedi_images
        else:
            image, label = read_and_decode(
                filename_queue=filename_queue,
                im_size=im_size,
                model_input_shape=model_input_shape,
                train=train,
                max_value=max_value,
                min_value=min_value,
                return_filename=return_filename,
                normalize=normalize)

            images, sparse_labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=1000)

            return images, sparse_labels
