import os
import time
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from exp_ops.tf_fun import make_dir, find_ckpts
from exp_ops.plotting_fun import plot_accuracies, plot_std, plot_cms, plot_pr,\
    plot_cost
from exp_ops.preprocessing_GEDI_images import produce_patch
from gedi_config import GEDIconfig
from models import GEDI_vgg16_trainable_batchnorm_shared as vgg16
from tqdm import tqdm


def crop_center(img, crop_size):
    x, y = img.shape
    cx, cy = crop_size
    startx = x//2-(cx//2)
    starty = y//2-(cy//2)
    return img[starty:starty+cy, startx:startx+cx]


def renormalize(img, max_value, min_value):
    return (img - min_value) / (max_value - min_value)


def image_batcher(
        start,
        num_batches,
        images,
        config):
    for b in range(num_batches):
        print start, len(images)
        next_image_batch = images[start:start + config.validation_batch]
        image_stack = [renormalize(
            crop_center(
                produce_patch(
                    f,
                    config.channel,
                    config.panel,
                    divide_panel=config.divide_panel),
                config.model_image_size[:2]),
            max_value=config.max_gedi,
            min_value=config.min_gedi   
            ) for f in next_image_batch]
        # Add dimensions and concatenate
        yield np.concatenate(
            [x[None, :, :, None] for x in image_stack], axis=0).repeat(
                3, axis=-1), next_image_batch


def randomization_test(y, yhat, iterations=10000):
    true_score = np.mean(y == yhat)
    perm_scores = np.zeros((iterations))
    lab_len = len(y)
    for it in range(iterations):
        perm_scores[it] = np.mean(
            yhat == np.copy(y)[np.random.permutation(lab_len)])
    p_value = (np.sum(true_score < perm_scores) + 1) / float(iterations + 1)
    return p_value


# Evaluate your trained model on GEDI images
def test_vgg16(image_dir, model_file):
    config = GEDIconfig()

    if image_dir is None:
        raise RuntimeError(
            'You need to supply a directory path to the images.')

    combined_files = np.asarray(glob(os.path.join(image_dir, '*%s' % config.raw_im_ext)))

    # Find model checkpoints
    ds_dt_stamp = re.split('/', model_file)[-2]
    out_dir = os.path.join(config.results, ds_dt_stamp)
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
        max_value = np.asarray(config.max_gedi).astype(np.float32)
        min_value = np.asarray(config.min_gedi).astype(np.float32)

    # Make output directories if they do not exist
    dir_list = [config.results, out_dir]
    [make_dir(d) for d in dir_list]

    # Prepare data on CPU
    images = tf.placeholder(
        tf.float32,
        shape=[None] + config.model_image_size,
        name='images')

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            vgg.build(
                images, output_shape=config.output_shape)

        # Setup validation op
        scores = vgg.prob
        preds = tf.argmax(vgg.prob, 1)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Loop through each checkpoint then test the entire validation set
    ckpts = [model_file]
    ckpt_yhat, ckpt_y, ckpt_scores, ckpt_file_array = [], [], [], []
    print '-'*60
    print 'Beginning evaluation'
    print '-'*60

    for idx, c in tqdm(enumerate(ckpts), desc='Running checkpoints'):
        dec_scores, yhat, file_array = [], [], []
        # Initialize the graph
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(
            tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()))

        # Set up exemplar threading
        saver.restore(sess, c)
        start_time = time.time()
        num_batches = len(combined_files) // config.validation_batch
        for image_batch, file_batch in tqdm(
                image_batcher(
                    start=0,
                    num_batches=num_batches,
                    images=combined_files,
                    config=config),
                total=num_batches):
            feed_dict = {
                images: image_batch
            }
            sc, tyh = sess.run(
                [scores, preds],
                feed_dict=feed_dict)
            dec_scores = np.append(dec_scores, sc)
            yhat = np.append(yhat, tyh)
            file_array = np.append(file_array, file_batch)
        ckpt_yhat.append(yhat)
        ckpt_scores.append(dec_scores)
        ckpt_file_array.append(file_array)
        print 'Batch %d took %.1f seconds' % (
            idx, time.time() - start_time)
    sess.close()

    # Save everything
    np.savez(
        os.path.join(out_dir, 'validation_accuracies'),
        ckpt_yhat=ckpt_yhat,
        ckpt_scores=ckpt_scores,
        ckpt_names=ckpts,
        combined_files=ckpt_file_array,
        )

    # Also save a csv with item/guess pairs
    try:
        dec_scores = np.asarray(dec_scores)
        yhat = np.asarray(yhat)
        df = pd.DataFrame(
            np.hstack((
                np.asarray(ckpt_file_array).reshape(-1, 1),
                yhat.reshape(-1, 1),
                dec_scores.reshape(dec_scores.shape[0]//2, 2))),
            columns=['files', 'guesses', 'classifier score live', 'classifier score dead'])
        df.to_csv(os.path.join(out_dir, 'prediction_file.csv'))
        print 'Saved csv to: %s' % os.path.join(out_dir, 'prediction_file.csv')
    except:
        print 'X'*60
        print 'Could not save a spreadsheet of file info'
        print 'X'*60

    # Plot everything
    try:
        plot_accuracies(
            ckpt_y, ckpt_yhat, config, ckpts,
            os.path.join(out_dir, 'validation_accuracies.png'))
        plot_std(
            ckpt_y, ckpt_yhat, ckpts, os.path.join(
                out_dir, 'validation_stds.png'))
        plot_cms(
            ckpt_y, ckpt_yhat, config, os.path.join(
                out_dir, 'confusion_matrix.png'))
        plot_pr(
            ckpt_y, ckpt_yhat, ckpt_scores, os.path.join(
                out_dir, 'precision_recall.png'))
        plot_cost(
            os.path.join(out_dir, 'training_loss.npy'), ckpts,
            os.path.join(out_dir, 'training_costs.png'))
    except:
        print 'X'*60
        print 'Could not locate the loss numpy'
        print 'X'*60


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        dest="image_dir",
        default='/Users/drewlinsley/Documents/GEDI_images/human_bs',
        help="Directory containing your .tiff images.")
    parser.add_argument(
        "--model_file",
        type=str,
        dest="model_file",
        default='/Users/drewlinsley/Desktop/trained_gedi_model/model_58600.ckpt-58600',
        help="Folder containing your trained CNN's checkpoint files.")
    args = parser.parse_args()
    test_vgg16(**vars(args))
