import os
import time
import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from exp_ops.data_loader_matching import inputs
from exp_ops.tf_fun import make_dir, softmax_cost, fine_tune_prepare_layers, \
    class_accuracy
from gedi_config import GEDIconfig
from models import matching_gedi
from models import baseline_vgg16 as vgg16


def prep_images_for_gedi(
        images,
        res_size=[300, 300],
        crop_size=[224, 224]):
    """Resize images to [300, 300] and then crop to [224, 224]."""
    if int(images.get_shape()[-1]) < 3:
        images = tf.concat(
            [images, images, images],
            axis=-1)
    images = tf.image.resize_images(images, res_size)
    images = tf.map_fn(
        lambda img: tf.image.resize_image_with_crop_or_pad(
            img,
            crop_size[0],
            crop_size[1]),
        images)
    return images


def min_max_norm(x, eps=0.001):
    min_val = tf.reduce_min(x, reduction_indices=[1, 2, 3], keep_dims=True)
    max_val = tf.reduce_max(x, reduction_indices=[1, 2, 3], keep_dims=True)
    return (x - min_val) / (max_val - min_val + eps)


def train_model(train_dir=None, validation_dir=None):
    config = GEDIconfig()
    if train_dir is None:  # Use globals
        train_data = os.path.join(
            config.tfrecord_dir,
            config.tf_record_names['train'])
        meta_data = np.load(
            os.path.join(
                config.tfrecord_dir,
                '%s_%s' % (config.tvt_flags[0], config.max_file)))
    else:
        meta_data = np.load(
            os.path.join(
                train_dir,
                '%s_%s' % (config.tvt_flags[0], config.max_file)))

    # Prepare image normalization values
    if config.max_gedi is None:
        max_value = np.nanmax(meta_data['max_array']).astype(np.float32)
        if max_value == 0:
            max_value = None
            print 'Derived max value is 0'
        else:
            print 'Normalizing with empirical max.'
        if 'min_array' in meta_data.keys():
            min_value = np.min(meta_data['min_array']).astype(np.float32)
            print 'Normalizing with empirical min.'
        else:
            min_value = None
            print 'Not normalizing with a min.'
    else:
        max_value = config.max_gedi
        min_value = config.min_gedi
    ratio = meta_data['ratio']
    print 'Ratio is: %s' % ratio

    if validation_dir is None:  # Use globals
        validation_data = os.path.join(
            config.tfrecord_dir,
            config.tf_record_names['val'])
    elif validation_dir is False:
        pass  # Do not use validation data during training

    # Make output directories if they do not exist
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.which_dataset + '_' + dt_stamp + '/'
    config.train_checkpoint = os.path.join(
        config.train_checkpoint, dt_dataset)  # timestamp this run
    out_dir = os.path.join(config.results, dt_dataset)
    dir_list = [
        config.train_checkpoint, config.train_summaries,
        config.results, out_dir]
    [make_dir(d) for d in dir_list]
    # im_shape = get_image_size(config)
    im_shape = config.gedi_image_size

    print '-' * 60
    print('Training model:' + dt_dataset)
    print '-' * 60

    # Prepare data on CPU
    assert os.path.exists(train_data)
    assert os.path.exists(validation_data)
    assert os.path.exists(config.vgg16_weight_path)
    with tf.device('/cpu:0'):
        train_images_0, train_images_1, train_labels, train_times = inputs(
            train_data,
            config.train_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            train=config.data_augmentations,
            num_epochs=config.epochs,
            normalize=config.normalize,
            return_filename=True)
        val_images_0, val_images_1, val_labels, val_times = inputs(
            validation_data,
            config.validation_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            num_epochs=config.epochs,
            normalize=config.normalize,
            return_filename=True)
        tf.summary.image('train image frame 0', train_images_0)
        tf.summary.image('train image frame 1', train_images_1)
        tf.summary.image('validation image frame 0', val_images_0)
        tf.summary.image('validation image frame 1', val_images_1)

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('gedi'):
            # Build training GEDI model for frame 0
            vgg_train_mode = tf.get_variable(
                name='vgg_training',
                initializer=False)
            gedi_model_0 = vgg16.model_struct(
                vgg16_npy_path=config.gedi_weight_path,
                trainable=False)
            gedi_model_0.build(
                prep_images_for_gedi(train_images_0),
                output_shape=2,
                train_mode=vgg_train_mode)
            gedi_scores_0 = gedi_model_0.fc7

        with tf.variable_scope('match'):
            # Build matching model for frame 0
            model_0 = matching_gedi.model_struct()
            model_0.build(train_images_0)

            # Build frame 0 vector
            frame_0 = tf.concat([gedi_scores_0, model_0.output], axis=-1)

            # Build output layer
            if config.matching_combine == 'concatenate':
                output_shape = [int(frame_0.get_shape()[-1]) * 2, 2]
            elif config.matching_combine == 'subtract':
                output_shape = [int(frame_0.get_shape()[-1]), 2]
            else:
                raise RuntimeError

        # Build GEDI model for frame 1
        with tf.variable_scope('gedi', reuse=True):
            gedi_model_1 = vgg16.model_struct(
                vgg16_npy_path=config.gedi_weight_path,
                trainable=False)
            gedi_model_1.build(
                prep_images_for_gedi(train_images_1),
                output_shape=2,
                train_mode=vgg_train_mode)
            gedi_scores_1 = gedi_model_1.fc7

        with tf.variable_scope('match', reuse=True):
            # Build matching model for frame 1
            model_1 = matching_gedi.model_struct()
            model_1.build(train_images_1)

        # Build frame 0 and frame 1 vectors
        frame_1 = tf.concat([gedi_scores_1, model_1.output], axis=-1)

        with tf.variable_scope('output'):
            # Concatenate or subtract
            if config.matching_combine == 'concatenate':
                output_scores = tf.concat([frame_0, frame_1], axis=-1)
            elif config.matching_combine == 'subtract':
                output_scores = frame_0 - frame_1
            else:
                raise NotImplementedError

            # Build output layer
            output_shape = [int(output_scores.get_shape()[-1]), 2]
            output_weights = tf.get_variable(
                name='output_weights',
                shape=output_shape,
                initializer=tf.contrib.layers.xavier_initializer(
                    uniform=False))
            output_bias = tf.get_variable(
                name='output_bias',
                initializer=tf.truncated_normal([output_shape[-1]], .0, .001))
            decision_logits = tf.nn.bias_add(
                tf.matmul(
                    output_scores,
                    output_weights), output_bias)
            train_soft_decisions = tf.nn.softmax(decision_logits)
            cost = softmax_cost(
                decision_logits,
                train_labels)
            tf.summary.scalar("cce loss", cost)
            cost += tf.nn.l2_loss(output_weights)

            # Weight decay
            if config.wd_layers is not None:
                _, l2_wd_layers = fine_tune_prepare_layers(
                    tf.trainable_variables(), config.wd_layers)
                l2_wd_layers = [
                    x for x in l2_wd_layers if 'biases' not in x.name]
                if len(l2_wd_layers) > 0:
                    cost += (config.wd_penalty * tf.add_n(
                        [tf.nn.l2_loss(x) for x in l2_wd_layers]))

        # Optimize
        train_op = tf.train.AdamOptimizer(config.new_lr).minimize(cost)
        train_accuracy = class_accuracy(
            train_soft_decisions,
            train_labels)  # training accuracy
        tf.summary.scalar("training accuracy", train_accuracy)

        # Setup validation op
        if validation_data is not False:
            with tf.variable_scope('gedi', reuse=tf.AUTO_REUSE):  # FIX THIS
                # Validation graph is the same as training except no batchnorm
                val_gedi_model_0 = vgg16.model_struct(
                    vgg16_npy_path=config.gedi_weight_path)
                val_gedi_model_0.build(
                    prep_images_for_gedi(val_images_0),
                    output_shape=2,
                    train_mode=vgg_train_mode)
                val_gedi_scores_0 = val_gedi_model_0.fc7

                # Build GEDI model for frame 1
                val_gedi_model_1 = vgg16.model_struct(
                    vgg16_npy_path=config.gedi_weight_path)
                val_gedi_model_1.build(
                    prep_images_for_gedi(val_images_1),
                    output_shape=2,
                    train_mode=vgg_train_mode)
                val_gedi_scores_1 = val_gedi_model_1.fc7

            with tf.variable_scope('match', reuse=tf.AUTO_REUSE):
                # Build matching model for frame 0
                val_model_0 = matching_gedi.model_struct()
                val_model_0.build(val_images_0)

                # Build matching model for frame 1
                val_model_1 = matching_gedi.model_struct()
                val_model_1.build(val_images_1)

            # Build frame 0 and frame 1 vectors
            val_frame_0 = tf.concat(
                [val_gedi_scores_0, val_model_0.output], axis=-1)
            val_frame_1 = tf.concat(
                [val_gedi_scores_1, val_model_1.output], axis=-1)

            # Concatenate or subtract
            if config.matching_combine == 'concatenate':
                val_output_scores = tf.concat(
                    [val_frame_0, val_frame_1], axis=-1)
            elif config.matching_combine == 'subtract':
                val_output_scores = val_frame_0 - val_frame_1
            else:
                raise NotImplementedError

            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                # Build output layer
                val_output_weights = tf.get_variable(
                    name='val_output_weights',
                    shape=output_shape,
                    trainable=False,
                    initializer=tf.contrib.layers.xavier_initializer(
                        uniform=False))
                val_output_bias = tf.get_variable(
                    name='output_bias',
                    trainable=False,
                    initializer=tf.truncated_normal(
                        [output_shape[-1]], .0, .001))
                val_decision_logits = tf.nn.bias_add(
                    tf.matmul(
                        val_output_scores,
                        val_output_weights), val_output_bias)
                val_soft_decisions = tf.nn.softmax(val_decision_logits)

            # Calculate validation accuracy
            val_accuracy = class_accuracy(
                val_soft_decisions,
                val_labels)
            tf.summary.scalar("validation accuracy", val_accuracy)

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    summary_dir = os.path.join(
        config.train_summaries, config.which_dataset + '_' + dt_stamp)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(out_dir + 'meta_info', config)
    step, losses = 0, []  # val_max = 0
    try:
        # print response
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value, train_acc, val_acc = sess.run(
                [train_op, cost, train_accuracy, val_accuracy])
            losses += [loss_value]
            duration = time.time() - start_time
            if np.isnan(loss_value).sum():
                assert not np.isnan(loss_value), 'Model loss = NaN'

            if step % config.validation_steps == 0:
                if validation_data is not False:
                    val_acc = sess.run(val_accuracy)
                else:
                    val_acc -= 1  # Store every checkpoint

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | '
                    'logdir = %s')
                print (format_str % (
                    datetime.now(), step, loss_value,
                    config.train_batch / duration, float(duration),
                    train_acc, val_acc, summary_dir))

                # Save the model checkpoint if it's the best yet
                if 1:  # val_acc >= val_max:
                    saver.save(
                        sess, os.path.join(
                            config.train_checkpoint,
                            'model_' + str(step) + '.ckpt'), global_step=step)
                    # Store the new max validation accuracy
                    # val_max = val_acc

            else:
                # Training status
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | Training accuracy = %s | '
                              'Training loss = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.train_batch / duration,
                                     float(duration), loss_value))
            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
    finally:
        coord.request_stop()
        np.save(os.path.join(config.tfrecord_dir, 'training_loss'), losses)
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dir", type=str, dest="train_dir",
        default=None, help="Directory of training data tfrecords bin file.")
    parser.add_argument(
        "--validation_dir", type=str, dest="validation_dir",
        default=None, help="Directory of validation data tfrecords bin file.")
    args = parser.parse_args()
    train_model(**vars(args))
