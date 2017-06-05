import os
import time
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from exp_ops.data_loader import inputs
from exp_ops.tf_fun import make_dir, find_ckpts
from exp_ops.plotting_fun import plot_accuracies, plot_std, plot_cms, plot_pr, plot_cost
from gedi_config import GEDIconfig
from models import GEDI_vgg16_trainable_batchnorm_shared as vgg16
from tqdm import tqdm


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
def test_vgg16(validation_data, model_dir, which_set, selected_ckpts=-1):
    config = GEDIconfig()
    if validation_data is None:  # Use globals
        validation_data = os.path.join(
            config.tfrecord_dir,
            config.tf_record_names[which_set])
        meta_data = np.load(
            os.path.join(
                config.tfrecord_dir, 'val_%s' % config.max_file))
    else:
        meta_data = np.load(
            '%s_maximum_value.npz' % validation_data.split('.tfrecords')[0])
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
            while not coord.should_stop():
                sc, tyh, ty = sess.run([scores, preds, targets])
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
        os.path.join(out_dir, 'validation_accuracies'), ckpt_yhat=ckpt_yhat,
        ckpt_y=ckpt_y, ckpt_scores=ckpt_scores, ckpt_names=ckpt_names,
        file_pointers=file_pointers)

    # Also save a csv with item/guess pairs
    try:
        trimmed_files = [re.split('/', x)[-1] for x in file_pointers]
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
        "--validation_data",
        type=str,
        dest="validation_data",
        default=None,
        help="Validation data tfrecords bin file.")
    parser.add_argument(
        "--model_dir",
        type=str,
        dest="model_dir",
        default=None,
        help="Feed in a specific model for validation.")
    parser.add_argument(
        "--selected_ckpts",
        type=int,
        dest="selected_ckpts",
        default=None,
        help="Which checkpoint?")
    parser.add_argument(
        "--which_set",
        type=str,
        dest="which_set",
        default='val',
        help="Which set (e.g. 'val', 'test', or 'train')?")
    args = parser.parse_args()
    test_vgg16(**vars(args))
