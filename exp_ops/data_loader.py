import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob


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


def read_and_decode(
        filename_queue,
        im_size,
        model_input_shape,
        train,
        max_value=None,
        min_value=None,
        num_channels=2,
        return_filename=False,
        normalize=False):
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

    if normalize:  # If we want to ensure all images are [0, 1]
        image /= tf.reduce_max(image, keep_dims=True)
    image = tf.squeeze(image)  # had to add this 5/11/17... make sure this is OK

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
