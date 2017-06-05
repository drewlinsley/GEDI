#!/usr/bin/env python
import os, sys, re, shutil
import numpy as np
import tensorflow as tf
import glob
sys.path.append('../../') #puts model_depo on the path
sys.path.insert(0,re.split(__file__,os.path.realpath(__file__))[0]) #puts this experiment into path
from scipy.misc import imresize, imsave
from exp_ops.gedi_config import GEDIconfig
from exp_ops.helper_functions import make_dir
from exp_ops.mosaic_visualizations import maxabs, zscore_channels
from exp_ops import lrp
from model_depo import vgg16_trainable_lrp as vgg16
from ops import utils
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import OneHotEncoder as oe


# Evaluate your trained model on GEDI images
def test_vgg16(validation_data, model_dir, selected_ckpts=-1):
    config = GEDIconfig()
    if validation_data is None:  # Use globals
        validation_data = config.tf_record_names['val']
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
    dir_list = [config.results, out_dir]
    [make_dir(d) for d in dir_list]
    # im_shape = get_image_size(config)
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
        scores = vgg.prob
        preds = tf.argmax(vgg.prob, 1)
        targets = tf.cast(val_labels, dtype=tf.int64)
        oh_targets = tf.one_hot(val_labels, config.output_shape)

        # Masked LRP op
        heatmap = lrp.lrp(logits*oh_targets, -123.68, 255 - 123.68)
        # heatmap = lrp.get_lrp_im(sess, F, images, y, img, lab)[0]

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Get class indices for all files
    if use_true_label:
        label_key = np.asarray(config.label_directories)
        class_indices = [np.where(config.which_dataset + '_' + fn == label_key) for fn in labels]
    else:
        class_indices = [None] * len(image_filenames)


    # Loop through each checkpoint then test the entire validation set
    ckpt_yhat, ckpt_y, ckpt_scores = [], [], []
    print '-'*60
    print 'Beginning visualization'
    print '-'*60

    if selected_ckpts is not None:
        # Select a specific ckpt
        if selected_ckpts < 0:
            ckpts = ckpts[selected_ckpts:]
        else:
            ckpts = ckpts[:selected_ckpts]

    dec_scores, yhat, y, yoh, ims, hms = [], [], [], [], [], []
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
                sc, tyh, ty, tyoh, imb, ihm = sess.run([scores, preds, targets, oh_targets, val_images, heatmap])
                dec_scores += [sc]
                yhat += [tyh]
                y += [ty]
                yoh += [tyoh]
                ims += [imb]
                hms += [ihm]
        except tf.errors.OutOfRangeError:
            print 'Batch %d took %.1f seconds' % (
                idx, time.time() - start_time)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


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

