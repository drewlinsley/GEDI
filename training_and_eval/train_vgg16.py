import os
import time
import re
import tensorflow as tf
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from exp_ops.data_loader import inputs
from exp_ops.tf_fun import make_dir, softmax_cost, fine_tune_prepare_layers, \
    ft_non_optimized, class_accuracy
from gedi_config import GEDIconfig
# from models import GEDI_vgg16_trainable_batchnorm_shared as vgg16
from models import baseline_vgg16 as vgg16
# from models import small_cnn as vgg16
import pandas as pd
from sklearn.utils import class_weight


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
        mask = np.isnan(tod_data).astype(int) + (tod['plate_well_neuron'] == 'empty').as_matrix().astype(int)
        tod_data = tod_data[mask == 0]
        tod_data = tod_data[tod_data < 10]  # throw away values have a high number
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
        train_images, train_labels = inputs(
            train_data,
            config.train_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            train=config.data_augmentations,
            num_epochs=config.epochs,
            normalize=config.normalize)
        val_images, val_labels = inputs(
            validation_data,
            config.validation_batch,
            im_shape,
            config.model_image_size[:2],
            max_value=max_value,
            min_value=min_value,
            num_epochs=config.epochs,
            normalize=config.normalize)
        tf.summary.image('train images', train_images)
        tf.summary.image('validation images', val_images)

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            if config.ordinal_classification == 'ordinal':  # config.output_shape > 2:  # Hardcoded fix for timecourse pred
                vgg_output = config.output_shape * 2
            elif config.ordinal_classification == 'regression':
                vgg_output = 1
            elif config.ordinal_classification is None:
                vgg_output = config.output_shape
            vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            train_mode = tf.get_variable(name='training', initializer=True)
            vgg.build(
                train_images, output_shape=vgg_output,
                train_mode=train_mode, batchnorm=config.batchnorm_layers)

            # Prepare the cost function
            if config.ordinal_classification == 'ordinal':
                # Encode y w/ k-hot and yhat w/ sigmoid ce. units capture dist.
                enc = tf.concat(
                    [tf.reshape(
                        tf.range(
                            0, config.output_shape), [1, -1]) for x in range(
                        config.train_batch)], axis=0)
                enc_train_labels = tf.cast(tf.greater_equal(
                    enc,
                    tf.expand_dims(train_labels, axis=1)), tf.float32)
                split_labs = tf.split(enc_train_labels, config.output_shape, axis=1)
                res_output = tf.reshape(vgg.fc8, [config.train_batch, 2, config.output_shape])
                split_logs = tf.split(res_output, config.output_shape, axis=2)
                if config.balance_cost:
                    cost = tf.add_n([tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.one_hot(tf.cast(tf.squeeze(s), tf.int32), 2),
                        logits=tf.squeeze(l)) * r for s, l, r in zip(split_labs, split_logs, ratio)])
                else:
                    cost = tf.add_n([tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.one_hot(tf.cast(tf.squeeze(s), tf.int32), 2),
                        logits=tf.squeeze(l)) for s, l in zip(split_labs, split_logs)])
                cost = tf.reduce_mean(cost)
            elif config.ordinal_classification == 'regression':
                if config.balance_cost:
                    weight_vec = tf.gather(train_labels, ratio)
                    cost = tf.reduce_mean(tf.pow((vgg.fc8) - tf.cast(train_labels, tf.float32), 2) * weight_vec)
                else:
                    cost = tf.nn.l2_loss((vgg.fc8) - tf.cast(train_labels, tf.float32))
            else:
                if config.balance_cost:
                    cost = softmax_cost(
                        vgg.fc8,
                        train_labels,
                        ratio=ratio,
                        flip_ratio=flip_ratio)
                else:
                    cost = softmax_cost(vgg.fc8, train_labels)
            tf.summary.scalar("cost", cost)

            # Finetune the learning rates
            if config.wd_layers is not None:
                _, l2_wd_layers = fine_tune_prepare_layers(
                    tf.trainable_variables(), config.wd_layers)
                l2_wd_layers = [
                    x for x in l2_wd_layers if 'biases' not in x.name]
                cost += (config.wd_penalty * tf.add_n(
                        [tf.nn.l2_loss(x) for x in l2_wd_layers]))

            # for all variables in trainable variables
            # print name if there's duplicates you fucked up
            other_opt_vars, ft_opt_vars = fine_tune_prepare_layers(
                tf.trainable_variables(), config.fine_tune_layers)
            if config.optimizer == 'adam':
                train_op = ft_non_optimized(
                    cost, other_opt_vars, ft_opt_vars,
                    tf.train.AdamOptimizer, config.hold_lr, config.new_lr)
            elif config.optimizer == 'sgd':
                train_op = ft_non_optimized(
                    cost, other_opt_vars, ft_opt_vars,
                    tf.train.GradientDescentOptimizer,
                    config.hold_lr, config.new_lr)

            if config.ordinal_classification == 'ordinal':
                arg_guesses = tf.cast(tf.reduce_sum(tf.squeeze(tf.argmax(res_output, axis=1)), reduction_indices=[1]), tf.int32)
                train_accuracy = tf.reduce_mean(tf.cast(tf.equal(arg_guesses, train_labels), tf.float32)) 
            elif config.ordinal_classification == 'regression':
                train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(vgg.prob), tf.int32), train_labels), tf.float32))
            else:
                train_accuracy = class_accuracy(
                    vgg.prob, train_labels)  # training accuracy
            tf.summary.scalar("training accuracy", train_accuracy)

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                # Validation graph is the same as training except no batchnorm
                val_vgg = vgg16.Vgg16(
                    vgg16_npy_path=config.vgg16_weight_path,
                    fine_tune_layers=config.fine_tune_layers)
                val_vgg.build(val_images, output_shape=vgg_output)
                # Calculate validation accuracy
                if config.ordinal_classification == 'ordinal':
                    val_res_output = tf.reshape(val_vgg.fc8, [config.validation_batch, 2, config.output_shape]) 
                    val_arg_guesses = tf.cast(tf.reduce_sum(tf.squeeze(tf.argmax(val_res_output, axis=1)), reduction_indices=[1]), tf.int32)
                    val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_arg_guesses, val_labels), tf.float32))
                elif config.ordinal_classification == 'regression':
                    val_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(val_vgg.prob), tf.int32), val_labels), tf.float32))
                else:
                    val_accuracy = class_accuracy(val_vgg.prob, val_labels)
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
            _, loss_value, train_acc = sess.run(
                [train_op, cost, train_accuracy])
            losses.append(loss_value)
            duration = time.time() - start_time
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
                    'Validation accuracy = %s | logdir = %s')
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
                              '%.3f sec/batch) | Training accuracy = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.train_batch / duration,
                                     float(duration), train_acc))
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
