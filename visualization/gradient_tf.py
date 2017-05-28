import os
import sys
import time
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from exp_ops.data_loader import inputs
from exp_ops.tf_fun import make_dir, find_ckpts
from exp_ops.plotting_fun import plot_accuracies, plot_std, plot_cms, plot_pr, plot_cost, plot_hms
from gedi_config import GEDIconfig
from models import GEDI_vgg16_trainable_batchnorm_shared as vgg16
from tqdm import tqdm
from matplotlib import pyplot as plt


def hm_normalize(x):
    return np.abs(x).sum(axis=-1).squeeze()


def loop_plot(ims, hms, label, pointer, blur):
    for im, hm in zip(ims, hms):
        plot_hms(im, hm, label, pointer, fsize=blur)
    print 'Saved images to %s' % pointer


def test_vgg16(validation_data, model_dir, selected_ckpts=-1):
    config = GEDIconfig()
    if validation_data is None:  # Use globals
        validation_data = config.tfrecord_dir + 'val.tfrecords'
        meta_data = np.load(
            os.path.join(
                config.tfrecord_dir, 'val_' +
                config.max_file))
    else:
        meta_data = np.load(validation_data.split('.tfrecords')[0] + '_maximum_value.npz')
    label_list = os.path.join(
        config.processed_image_patch_dir, 'list_of_' + '_'.join(
            x for x in config.image_prefixes) + '_labels.txt')
    with open(label_list) as f:
        file_pointers = [l.rstrip('\n') for l in f.readlines()]

    # Prepare image normalization values
    try:
        max_value = np.max(meta_data['max_array']).astype(np.float32)
    except:
        max_value = np.asarray([config.max_gedi])
    try:
        min_value = np.max(meta_data['min_array']).astype(np.float32)
    except:
        min_value = np.asarray([config.min_gedi])


    # Find model checkpoints
    ckpts, ckpt_names = find_ckpts(config, model_dir)
    ds_dt_stamp = re.split('/', ckpts[0])[-2]
    out_dir = os.path.join(config.results, ds_dt_stamp + '/')
    try:
        config = np.load(os.path.join(out_dir, 'meta_info.npy')).item()
        # Make sure this is always at 1
        config.validation_batch = 1
        print '-'*60
        print 'Loading config meta data for:%s' % out_dir
        print '-'*60
    except:
        print '-'*60
        print 'Using config from gedi_config.py for model:%s' % out_dir
        print '-'*60


    # Make output directories if they do not exist
    im_shape = config.gedi_image_size

    # Prepare data on CPU
    with tf.device('/cpu:0'):
            val_images, val_labels = inputs(
                validation_data,
                1,
                im_shape,
                config.model_image_size[:2],
                max_value=max_value,
                min_value=min_value,
                num_epochs=1,
                normalize=config.normalize)

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            vgg.build(
                val_images, output_shape=config.output_shape)

        # Setup validation op
        preds = tf.argmax(vgg.prob, 1)
        targets = tf.cast(val_labels, dtype=tf.int64)
        grad_labels = tf.one_hot(
            val_labels,
            config.output_shape,
            dtype=tf.float32)
        heatmap_op = tf.gradients(
            vgg.fc8 * grad_labels,
            val_images)[0]

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    if selected_ckpts is not None:
        # Select a specific ckpt
        ckpts = [ckpts[selected_ckpts]]
    else:
        ckpts = ckpts[-1]

    # Loop through each checkpoint then test the entire validation set
    print '-'*60
    print 'Beginning evaluation on ckpt: %s' % ckpts
    print '-'*60
    yhat, y, tn_hms, tp_hms, fn_hms, fp_hms, tn_ims, tp_ims, fn_ims, fp_ims = [], [], [], [], [], [], [], [], [], []
    for idx, c in tqdm(enumerate(ckpts), desc='Running checkpoints'):
        try:
            # Initialize the graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(
                tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()))

            # Set up exemplar threading
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            saver.restore(sess, c)
            start_time = time.time()
            while not coord.should_stop():
                tyh, ty, thm, tim = sess.run([preds, targets, heatmap_op, val_images])
                tyh = tyh[0]
                ty = ty[0]
                tim = tim / tim.max()
                yhat += [tyh]
                y += [ty]
                if tyh == ty and not tyh:  # True negative
                    tn_hms += [hm_normalize(thm)]
                    tn_ims += [tim]
                elif tyh == ty and tyh:  # True positive
                    tp_hms += [hm_normalize(thm)]
                    tp_ims += [tim]
                elif tyh != ty and not tyh:  # False negative
                    fn_hms += [hm_normalize(thm)]
                    fn_ims += [tim]
                elif tyh != ty and tyh:  # False positive
                    fp_hms += [hm_normalize(thm)]
                    fp_ims += [tim]
        except tf.errors.OutOfRangeError:
            print 'Batch %d took %.1f seconds' % (
                idx, time.time() - start_time)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

    # Plot images
    dir_pointer = os.path.join(config.heatmap_source_images, ds_dt_stamp) 
    stem_dirs = ['tn', 'tp', 'fn', 'fp']
    dir_list = [dir_pointer]
    dir_list += [os.path.join(dir_pointer, x) for x in stem_dirs]
    [make_dir(d) for d in dir_list]
    loop_plot(tn_ims, tn_hms, 'True negative', os.path.join(dir_pointer, 'tn'), blur=config.blur)
    loop_plot(tp_ims, tp_hms, 'True positive', os.path.join(dir_pointer, 'tp'), blur=config.blur)
    loop_plot(fn_ims, fn_hms, 'False negative', os.path.join(dir_pointer, 'fn'), blur=config.blur)
    loop_plot(fp_ims, fp_hms, 'False positive', os.path.join(dir_pointer, 'fp'), blur=config.blur)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--validation_data", type=str, dest="validation_data",
        default=None, help="Validation data tfrecords bin file.")
    parser.add_argument(
        "--model_dir", type=str, dest="model_dir",
        default=None, help="Feed in a specific model for validation.")
    parser.add_argument(
        "--selected_ckpts", type=int, dest="selected_ckpts",
        default=None, help="Which checkpoint?")
    args = parser.parse_args()
    test_vgg16(**vars(args))
