import math
import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob


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


def random_contrast_brightness(
        image,
        lower=0.5,
        upper=1.5,
        max_delta=32. / 255.):
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
        normalize=False,
        num_panels=3):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = {
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    }

    if return_filename:
        features['filename'] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(
        serialized_example,
        features=features)

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.float32)

    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, np.asarray(im_size))

    # Split the images
    split_image = tf.split(image, num_panels, axis=-1)

    # Insert augmentation and preprocessing here
    if train is not None:
        if 'left_right' in train:
            for idx, im in enumerate(split_image):
                im = tf.image.random_flip_left_right(im)
                split_image[idx] = im
        if 'up_down' in train:
            for idx, im in enumerate(split_image):
                im = tf.image.random_flip_up_down(im)
                split_image[idx] = im
        if 'rotate' in train:
            for idx, im in enumerate(split_image):
                random_rot = tf.squeeze(
                    tf.multinomial(tf.log([[10., 10., 10., 10., 10.]]), 1))
                rotation = tf.gather(
                    [0., 45., 90., 180., 270.], random_rot) * tf.constant(
                    math.pi) / 180.
                im = tf.contrib.image.rotate(
                    im, rotation)
                split_image[idx] = im
        if 'random_contrast' in train:
            for idx, im in enumerate(split_image):
                im = tf.image.random_contrast(
                    im, lower=0.0, upper=0.1)
                split_image[idx] = im
        if 'random_brightness' in train:
            for idx, im in enumerate(split_image):
                im = tf.image.random_brightness(
                    im, max_delta=0.1)
                split_image[idx] = im
        if 'random_crop' in train and 'resize' not in train:
            for idx, im in enumerate(split_image):
                im = tf.random_crop(
                    im, model_input_shape)
                split_image[idx] = im
        if 'resize' in train:
            for idx, im in enumerate(split_image):
                im = tf.image.resize_images(
                    im, model_input_shape[:2])
                split_image[idx] = im
        else:
            # for idx, im in enumerate(split_image):
            #     im = tf.image.resize_image_with_crop_or_pad(
            #         im, model_input_shape[0], model_input_shape[1])
            #     split_image[idx] = im
            for idx, im in enumerate(split_image):
                im = tf.image.resize_images(
                    im, model_input_shape[:2])
                split_image[idx] = im
    else:
        for idx, im in enumerate(split_image):
            im = tf.image.resize_image_with_crop_or_pad(
                im, model_input_shape[0], model_input_shape[1])
            split_image[idx] = im

    # if max_value is None:
    #     max_value = tf.reduce_max(
    #         image,
    #         reduction_indices=[0, 1])
    # else:
    #     if len(max_value) < int(image.get_shape())[-1]:
    #         max_value = max_value.repeat(int(image.get_shape()[-1]))
    #     max_value = np.asarray(max_value)[None, None, None]

    # Make sure to clip values to [0, 1]
    for idx, im in enumerate(split_image):
        if normalize:  # If we want to ensure all images are [0, 1]
            im = im / tf.reduce_max(im, keep_dims=True)
            im = tf.clip_by_value(tf.cast(im, tf.float32), 0.0, 1.0)
            split_image[idx] = im

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    if return_filename:
        # filename = tf.decode_raw(features['filename'], tf.float32)
        # filename = tf.reshape(filename, [])
        return split_image, label, label  # TODO: fix this filename
    else:
        return split_image, label


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
        normalize=True):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file],
            num_epochs=num_epochs)
        assert return_filename, 'For now you must return the filename.'
        # Even when reading in multiple threads, share the filename
        # queue.
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
