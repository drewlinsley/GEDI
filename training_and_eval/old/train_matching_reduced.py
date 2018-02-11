import os
import time
import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from exp_ops.data_loader_matching import inputs
from exp_ops.tf_fun import make_dir, fine_tune_prepare_layers
from gedi_config import GEDIconfig
from models import reduced_matching_gedi_big as matching_gedi
# from models import baseline_vgg16 as vgg16


def l2_dist(x, y, axis=None):
    """L2 distance in tensorflow."""
    if axis is None:
        return tf.norm(x - y, ord='euclidean')
    else:
        return tf.norm(x - y, ord='euclidean', axis=axis)


def train_model(train_dir=None, validation_dir=None, debug=True):
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
            # min_value = np.min(meta_data['min_array']).astype(np.float32)
            print 'Normalizing with empirical min.'
        else:
            # min_value = None
            print 'Not normalizing with a min.'
    else:
        max_value = config.max_gedi
        # min_value = config.min_gedi
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
        train_images, train_labels, train_times = inputs(
            train_data,
            config.train_batch,
            im_shape,
            config.model_image_size,
            # max_value=max_value,
            # min_value=min_value,
            train=config.data_augmentations,
            num_epochs=config.epochs,
            normalize=config.normalize,
            return_filename=True)
        val_images, val_labels, val_times = inputs(
            validation_data,
            config.validation_batch,
            im_shape,
            config.model_image_size,
            # max_value=max_value,
            # min_value=min_value,
            num_epochs=config.epochs,
            normalize=config.normalize,
            return_filename=True)
        train_image_list, val_image_list = [], []
        for idx in range(int(train_images.get_shape()[1])):
            train_image_list += [tf.gather(train_images, idx, axis=1)]
            val_image_list += [tf.gather(val_images, idx, axis=1)]
            tf.summary.image(
                'train_image_frame_%s' % idx, train_image_list[idx])
            tf.summary.image(
                'validation_image_frame_%s' % idx, val_image_list[idx])

    # Prepare model on GPU
    with tf.device('/gpu:0'):

        with tf.variable_scope('match'):
            # Build matching model for frame 0
            model_0 = matching_gedi.model_struct()
            frame_activity = []
            frame_activity += [tf.nn.l2_normalize(
                model_0.build(train_image_list[0]), 0, 1e-12)]

        with tf.variable_scope('match', reuse=tf.AUTO_REUSE):
            # Build matching model for other frames
            for idx in range(1, len(train_image_list)):
                frame_activity += [tf.nn.l2_normalize(
                    model_0.build(train_image_list[idx]), 0, 1e-12)]

        pos = l2_dist(frame_activity[0], frame_activity[1], axis=1)
        neg = l2_dist(frame_activity[0], frame_activity[2], axis=1)
        # loss = tf.reduce_mean(tf.pow(pos - neg + 0.2, 2))
        loss = tf.reduce_mean(tf.maximum(pos - neg + 0.01, 0.))
        # loss = tf.reduce_mean(tf.nn.relu(1 - (neg / (pos + 0.2)))) * 100
        tf.summary.scalar('Triplet_loss', loss)

        # Weight decay
        if config.wd_layers is not None:
            _, l2_wd_layers = fine_tune_prepare_layers(
                tf.trainable_variables(), config.wd_layers)
            l2_wd_layers = [
                x for x in l2_wd_layers if 'biases' not in x.name]
            if len(l2_wd_layers) > 0:
                loss += (config.wd_penalty * tf.add_n(
                    [tf.nn.l2_loss(x) for x in l2_wd_layers]))

        # Optimize
        train_op = tf.train.AdamOptimizer(config.new_lr).minimize(loss)
        train_accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.nn.relu(tf.sign(neg - pos)),  # 1 if pos < neg
                    tf.cast(tf.ones_like(train_labels), tf.float32)),
                tf.float32))
        tf.summary.scalar('training_accuracy', train_accuracy)

        # Setup validation op
        if validation_data is not False:
            with tf.variable_scope('match', tf.AUTO_REUSE) as match:
                # Build matching model for frame 0
                match.reuse_variables()
                val_model_0 = matching_gedi.model_struct()
                val_frame_activity = []
                val_frame_activity += [tf.nn.l2_normalize(
                    val_model_0.build(val_image_list[0]), 1, 1e-12) ]

                # Build matching model for other frames
                for idx in range(1, len(train_image_list)):
                    val_frame_activity += [
                        tf.nn.l2_normalize(
                            val_model_0.build(
                                val_image_list[idx]), 1, 1e-12)]

            val_pos = l2_dist(
                val_frame_activity[0], val_frame_activity[1], axis=1)
            val_neg = l2_dist(
                val_frame_activity[0], val_frame_activity[2], axis=1)
            # val_loss = tf.reduce_mean(tf.pow(val_pos - val_neg + 0.2, 2))
            val_loss = tf.reduce_mean(tf.maximum(val_pos - val_neg + 0.01, 0.))
            # val_loss = tf.reduce_mean(tf.nn.relu(1 - (val_neg / (val_pos + 0.2)))) * 100
            tf.summary.scalar('Validation_triplet_loss', val_loss)

            # Calculate validation accuracy
        val_accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.nn.relu(tf.sign(val_neg - val_pos)), # 1 if pos < neg
                    tf.cast(tf.ones_like(val_labels), tf.float32)),
                tf.float32))
        tf.summary.scalar('val_accuracy', val_accuracy)

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

    # Train operations
    train_dict = {
        'train_op': train_op,
        'loss': loss,
        'pos': pos,
        'neg': neg,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
    }
    val_dict = {
        'val_accuracy': val_accuracy
    }
    if debug:
        for idx in range(len(train_image_list)):
            train_dict['train_im_%s' % idx] = train_image_list[idx]
        for idx in range(len(val_image_list)):
            val_dict['val_im_%s' % idx] = val_image_list[idx]

    # Start training loop
    np.save(out_dir + 'meta_info', config)
    step, losses = 0, []  # val_max = 0
    try:
        # print response
        while not coord.should_stop():
            start_time = time.time()
            train_values = sess.run(train_dict.values())
            it_train_dict = {k: v for k, v in zip(
                train_dict.keys(), train_values)}
            losses += [it_train_dict['loss']]
            duration = time.time() - start_time
            if np.isnan(it_train_dict['loss']).sum():
                assert not np.isnan(it_train_dict['loss']),\
                    'Model loss = NaN'

            if step % config.validation_steps == 0:
                if validation_data is not False:
                    val_values = sess.run(val_dict.values())
                    it_val_dict = {k: v for k, v in zip(
                        val_dict.keys(), val_values)}
                    val_acc = it_val_dict['val_accuracy']
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
                    datetime.now(),
                    step,
                    it_train_dict['loss'],
                    config.train_batch / duration,
                    float(duration),
                    it_train_dict['train_accuracy'],
                    it_train_dict['val_accuracy'],
                    summary_dir))

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
                              '%.3f sec/batch) | Training accuracy = %s | ')
                print (format_str % (
                    datetime.now(),
                    step,
                    it_train_dict['loss'],
                    config.train_batch / duration,
                    float(duration),
                    it_train_dict['train_accuracy']))
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
