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


def image_batcher(
        start,
        images,
        labels,
        config):
    if start + config.validation_batch > len(images):
        return
    next_image_batch = images[start:start + config.validation_batch]
    next_label_batch = labels[start:start + config.validation_batch]
    image_stack = np.concatenate(
        [crop_center(
            produce_patch(
                f,
                config.channel,
                config.panel,
                divide_panel=config.divide_panel,
                max_value=config.max_gedi,
                min_value=config.min_gedi).astype(
                    np.float32),
            config.config.model_image_size[:2]) for f in next_image_batch])
    yield image_stack, next_label_batch


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
def test_vgg16(live_data, dead_data, model_dir, selected_ckpts=None):
    config = GEDIconfig()

    if live_data is None:
        raise RuntimeError(
            'You need to supply a directory path to the validation_data.')
    if dead_data is None:
        print 'No dead file path detected. Running \'blinded\' analysis.'
    if selected_ckpts is None:
        raise RuntimeError(
            'Supply the name of your ckpt file.')

    live_files = glob(os.path.join(live_data, '*%s' % config.raw_im_ext))
    dead_files = glob(os.path.join(dead_data, '*%s' % config.raw_im_ext))
    combined_files = np.asarray(live_files + dead_files)
    combined_labels = np.concatenate(
        np.zeros((len(live_files)), np.ones(dead_files)))

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

    # Prepare data on CPU
    images = tf.placeholder(
        tf.float32,
        shape=[None] + config.model_image_size[:2],
        name='images')
    labels = tf.placeholder(
        tf.int64,
        shape=None,
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
        targets = tf.cast(labels, dtype=tf.int64)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Loop through each checkpoint then test the entire validation set
    ckpt_yhat, ckpt_y, ckpt_scores = [], [], []
    print '-'*60
    print 'Beginning evaluation'
    print '-'*60

    if selected_ckpts is not None:
        # Select a specific ckpt
        if selected_ckpts < 0:
            ckpts = ckpts[selected_ckpts:]
        else:
            ckpts = ckpts[:selected_ckpts]

    for idx, c in tqdm(enumerate(ckpts), desc='Running checkpoints'):
        dec_scores, yhat, y = [], [], []
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
            for image_batch, label_batch in image_batcher(
                    start=0,
                    images=combined_files,
                    labels=combined_labels,
                    config=config):
                feed_dict = {
                    images: image_batch,
                    labels: label_batch,
                }
                sc, tyh, ty = sess.run(
                    [scores, preds, targets],
                    feed_dict=feed_dict)
                dec_scores = np.append(dec_scores, sc)
                yhat = np.append(yhat, tyh)
                y = np.append(y, ty)
        except tf.errors.OutOfRangeError:
            ckpt_yhat.append(yhat)
            ckpt_y.append(y)
            ckpt_scores.append(dec_scores)
            print 'Iteration accuracy: %s' % np.mean(yhat == y)
            print 'Iteration pvalue: %.5f' % randomization_test(y=y, yhat=yhat)
            print 'Batch %d took %.1f seconds' % (
                idx, time.time() - start_time)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

    # Save everything
    np.savez(
        os.path.join(out_dir, 'validation_accuracies'),
        ckpt_yhat=ckpt_yhat,
        ckpt_y=ckpt_y,
        ckpt_scores=ckpt_scores,
        ckpt_names=ckpt_names,
        live_files=live_files,
        dead_files=dead_files,
        )

    # Also save a csv with item/guess pairs
    try:
        trimmed_files = [re.split('/', x)[-1] for x in combined_files]
        trimmed_files = np.asarray(trimmed_files)
        dec_scores = np.asarray(dec_scores)
        yhat = np.asarray(yhat)
        df = pd.DataFrame(
            np.hstack((
                trimmed_files.reshape(-1, 1),
                yhat.reshape(-1, 1),
                dec_scores.reshape(dec_scores.shape[0]//2, 2))),
            columns=['files', 'guesses', 'score dead', 'score live'])
        df.to_csv(os.path.join(out_dir, 'prediction_file.csv'))
        print 'Saved csv to: %s' % out_dir
    except:
        print 'X'*60
        print 'Could not save a spreadsheet of file info'
        print 'X'*60

    # Plot everything
    try:
        plot_accuracies(
            ckpt_y, ckpt_yhat, config, ckpt_names,
            os.path.join(out_dir, 'validation_accuracies.png'))
        plot_std(
            ckpt_y, ckpt_yhat, ckpt_names, os.path.join(
                out_dir, 'validation_stds.png'))
        plot_cms(
            ckpt_y, ckpt_yhat, config, os.path.join(
                out_dir, 'confusion_matrix.png'))
        plot_pr(
            ckpt_y, ckpt_yhat, ckpt_scores, os.path.join(
                out_dir, 'precision_recall.png'))
        plot_cost(
            os.path.join(out_dir, 'training_loss.npy'), ckpt_names,
            os.path.join(out_dir, 'training_costs.png'))
    except:
        print 'X'*60
        print 'Could not locate the loss numpy'
        print 'X'*60


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--live_data",
        type=str,
        dest="live_data",
        default=None,
        help="Folder containing your live images.")
    parser.add_argument(
        "--dead_data",
        type=str,
        dest="dead_data",
        default=None,
        help="Folder containing your dead images.")
    parser.add_argument(
        "--selected_ckpts",
        type=str,
        dest="selected_ckpts",
        default=None,
        help="Which checkpoint?")
    args = parser.parse_args()
    test_vgg16(**vars(args))
