import os
import time
import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from exp_ops.data_loader_gedi import inputs
from exp_ops.tf_fun import make_dir, softmax_cost, fine_tune_prepare_layers,\
    class_accuracy
from gedi_config import GEDIconfig
from models import baseline_vgg16 as vgg16
import pandas as pd
from sklearn.utils import class_weight


def min_max_norm(x, eps=0.001):
    min_val = tf.reduce_min(x, reduction_indices=[1, 2, 3], keep_dims=True)
    max_val = tf.reduce_max(x, reduction_indices=[1, 2, 3], keep_dims=True)
    return (x - min_val) / (max_val - min_val + eps)


# Train or finetune a vgg16 for the GEDI dataset
def train_vgg16(train_dir=None, validation_dir=None):
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
    if config.encode_time_of_death:
        tod = pd.read_csv(config.encode_time_of_death)
        tod_data = tod['dead_tp'].as_matrix()
        mask = np.isnan(tod_data).astype(int) + (
            tod['plate_well_neuron'] == 'empty').as_matrix().astype(int)
        tod_data = tod_data[mask == 0]
        tod_data = tod_data[tod_data > config.mask_timepoint_value]
        config.output_shape = len(np.unique(tod_data))
        ratio = class_weight.compute_class_weight(
            'balanced',
            np.sort(np.unique(tod_data)),
            tod_data)
        flip_ratio = False
    else:
        flip_ratio = True
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

    print '-'*60
    print('Training model:' + dt_dataset)
    print '-'*60

    # Prepare data on CPU
    assert os.path.exists(train_data)
    assert os.path.exists(validation_data)
    assert os.path.exists(config.vgg16_weight_path)
    with tf.device('/cpu:0'):
        train_images, train_labels, train_gedi_images = inputs(
            train_data,
            config.train_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            train=config.data_augmentations,
            num_epochs=config.epochs,
            normalize=config.normalize,
            return_gedi=config.include_GEDI_in_tfrecords,
            return_extra_gfp=config.extra_image,
            return_GEDI_derivative=True)
        val_images, val_labels, val_gedi_images = inputs(
            validation_data,
            config.validation_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            num_epochs=config.epochs,
            normalize=config.normalize,
            return_gedi=config.include_GEDI_in_tfrecords,
            return_extra_gfp=config.extra_image,
            return_GEDI_derivative=True)
        if config.include_GEDI_in_tfrecords:
            extra_im_name = 'GEDI at current timepoint'
        else:
            extra_im_name = 'next gfp timepoint'
        tf.summary.image('train images', train_images)
        tf.summary.image('validation images', val_images)
        tf.summary.image('train %s' % extra_im_name, train_gedi_images)
        tf.summary.image('validation %s' % extra_im_name, val_gedi_images)

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            if config.ordinal_classification is None:
                vgg_output = 2  # Sign of derivative (inf norm)
                train_labels = tf.cast(tf.sign(train_labels), tf.int32)
                val_labels = tf.cast(tf.sign(val_labels), tf.int32)
            elif config.ordinal_classification == 'regression':
                vgg_output = 1
            else:
                raise RuntimeError(
                    'config.ordinal_classification must be sign or regression.'
                    )
            vgg = vgg16.model_struct()
            train_mode = tf.get_variable(name='training', initializer=True)

            # Mask NAN images from loss
            image_nan = tf.reduce_sum(
                tf.cast(tf.is_nan(train_images), tf.float32),
                reduction_indices=[1, 2, 3])
            gedi_nan = tf.reduce_sum(
                tf.cast(tf.is_nan(train_gedi_images), tf.float32),
                reduction_indices=[1, 2, 3],
                keep_dims=True)
            image_mask = tf.cast(tf.equal(image_nan, 0.), tf.float32)
            gedi_nan = tf.cast(tf.equal(gedi_nan, 0.), tf.float32)
            train_images = tf.where(
                tf.is_nan(train_images),
                tf.zeros_like(train_images),
                train_images)
            train_gedi_images = tf.where(
                tf.is_nan(train_gedi_images),
                tf.zeros_like(train_gedi_images),
                train_gedi_images)
            train_images = tf.concat([train_images, train_images, train_images], axis=3)
            val_images = tf.concat([val_images, val_images, val_images], axis=3)
            vgg.build(
                train_images, output_shape=vgg_output,
                train_mode=train_mode, batchnorm=config.batchnorm_layers)
            # Prepare the cost function
            if config.ordinal_classification is None:
                # Encode y w/ k-hot and yhat w/ sigmoid ce. units capture dist.
                cost = softmax_cost(
                    vgg.fc8,
                    train_labels,
                    mask=image_mask)
            elif config.ordinal_classification == 'regression':
                cost = tf.nn.l2_loss(tf.squeeze(vgg.fc8) - train_labels)

            class_loss = cost
            tf.summary.scalar("cce cost", cost)

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
            if config.ordinal_classification is None:
                train_accuracy = class_accuracy(
                    vgg.prob, train_labels)  # training accuracy
            elif config.ordinal_classification == 'regression':
                train_accuracy = tf.nn.l2_loss(
                    tf.squeeze(vgg.fc8) - train_labels)
            tf.summary.scalar("training accuracy", train_accuracy)

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                # Validation graph is the same as training except no batchnorm
                val_vgg = vgg16.model_struct(
                    fine_tune_layers=config.fine_tune_layers)
                val_vgg.build(val_images, output_shape=vgg_output)
                # Calculate validation accuracy
                if config.ordinal_classification is None:
                    val_accuracy = class_accuracy(val_vgg.prob, val_labels)
                elif config.ordinal_classification == 'regression':
                    val_accuracy = tf.nn.l2_loss(tf.squeeze(val_vgg.fc8) - val_labels)
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

    # Restore model if requested
    if config.restore_path is not None:
        print '-' * 60
        print 'Restoring from a previous model: %s' % config.restore_path
        print '-' * 60
        saver.restore(sess, config.restore_path)

    # Start training loop
    np.save(out_dir + 'meta_info', config)
    step, losses = 0, []  # val_max = 0

    try:
        # print response
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value, train_acc = sess.run(
                [train_op, cost, train_accuracy])
            losses.append(loss_value)
            duration = time.time() - start_time
            if np.isnan(loss_value).sum():
                import ipdb;ipdb.set_trace()
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % config.validation_steps == 0:
                if validation_data is not False:
                    _, val_acc = sess.run([train_op, val_accuracy])
                else:
                    val_acc -= 1  # Store every checkpoint

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Training %s = %s | Training class loss = %s | '
                    'Validation accuracy = %s | Validation %s = %s | '
                    'logdir = %s')
                print (format_str % (
                    datetime.now(), step, loss_value,
                    config.train_batch / duration, float(duration),
                    train_acc, extra_im_name, 0.,
                    0., val_acc, extra_im_name,
                    0., summary_dir))

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
                              'Training %s = %s | Training class loss = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.train_batch / duration,
                                     float(duration), train_acc,
                                     extra_im_name, 0.,
                                     0.))
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
    train_vgg16(**vars(args))
