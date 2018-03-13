import os
import time
import re
import cPickle
import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from exp_ops.tf_fun import make_dir
from exp_ops.preprocessing_GEDI_images import produce_patch
from gedi_config import GEDIconfig
from models import baseline_vgg16 as vgg16
from tqdm import tqdm
from datetime import datetime


def crop_center(img, crop_size):
    """Center crop images."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def renormalize(img, max_value, min_value):
    """Normalize images to [0, 1]."""
    return (img - min_value) / (max_value - min_value)


def image_batcher(
        start,
        num_batches,
        images,
        labels,
        config,
        training_max,
        training_min):
    """Placeholder image/label batch loader."""
    for b in range(num_batches):
        next_image_batch = images[start:start + config.validation_batch]
        image_stack = []
        if labels is None:
            label_stack = None
        else:
            label_stack = labels[start:start + config.validation_batch]
        for f in next_image_batch:
            # 1. Load image patch
            patch = produce_patch(
                f,
                config.channel,
                config.panel,
                divide_panel=config.divide_panel,
                max_value=config.max_gedi,
                min_value=config.min_gedi).astype(np.float32)
            # 2. Repeat to 3 channel (RGB) image
            patch = np.repeat(patch[:, :, None], 3, axis=-1)
            # 3. Renormalize based on the training set intensities
            patch = renormalize(
                patch,
                max_value=training_max,
                min_value=training_min)
            # 4. Crop the center
            patch = crop_center(patch, config.model_image_size[:2])
            # 5. Clip to [0, 1] just in case
            patch[patch > 1.] = 1.
            patch[patch < 0.] = 0.
            # 6. Add to list
            image_stack += [patch[None, :, :, :]]
        # Add dimensions and concatenate
        start += config.validation_batch
        yield np.concatenate(image_stack, axis=0), label_stack, next_image_batch


def randomization_test(y, yhat, iterations=10000):
    """Randomization test of difference of predicted accuracy from chance."""
    true_score = np.mean(y == yhat)
    perm_scores = np.zeros((iterations))
    lab_len = len(y)
    for it in range(iterations):
        perm_scores[it] = np.mean(
            yhat == np.copy(y)[np.random.permutation(lab_len)])
    p_value = (np.sum(true_score < perm_scores) + 1) / float(iterations + 1)
    return p_value


def test_vgg16(
        model_file,
        trained_svm,
        ims,
        dead_ims=None,
        output_csv='prediction_file',
        C=1e-3,
        k_folds=10):
    """Test an SVM you've trained on a new dataset."""
    config = GEDIconfig()
    if ims is None:
        raise RuntimeError(
            'You need to supply a directory path to the images.')
    if dead_ims is None:
        print 'Assuming all of your images are in the ims folder' + \
            '-- will not derive labels to calculate accuracy.'
    # if not os.path.exists(trained_svm):
    #     raise RuntimeError(
    #         'Cannot find the trained svm model. Check the path you passed.')
    try:
        clf = cPickle.load(open(trained_svm, 'rb'))
    except:
        raise RuntimeError('Cannot find SVM file: %s' % trained_svm)

    if dead_ims is not None:
        live_files = glob(os.path.join(ims, '*%s' % config.raw_im_ext))
        dead_files = glob(os.path.join(dead_ims, '*%s' % config.raw_im_ext))
        combined_labels = np.concatenate((
            np.zeros(len(live_files)),
            np.ones(len(dead_files))))
        combined_files = np.concatenate((live_files, dead_files))
    else:
        live_files = glob(os.path.join(ims, '*%s' % config.raw_im_ext))
        combined_labels = None
        combined_files = np.asarray(live_files)
    if len(combined_files) == 0:
        raise RuntimeError('Could not find any files. Check your image path.')

    config = GEDIconfig()
    model_file_path = os.path.sep.join(model_file.split(os.path.sep)[:-1])
    meta_file_pointer = os.path.join(
        model_file_path,
        'train_maximum_value.npz')
    if not os.path.exists(meta_file_pointer):
        raise RuntimeError(
            'Cannot find the training data meta file: train_maximum_value.npz'
            'Closest I could find from directory %s was %s.'
            'Download this from the link described in the README.md.'
            % (model_file_path, glob(os.path.join(model_file_path, '*.npz'))))
    meta_data = np.load(meta_file_pointer)

    # Prepare image normalization values
    training_max = np.max(meta_data['max_array']).astype(np.float32)
    training_min = np.min(meta_data['min_array']).astype(np.float32)

    # Find model checkpoints
    ds_dt_stamp = re.split('/', model_file)[-2]
    out_dir = os.path.join(config.results, ds_dt_stamp)

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
            vgg = vgg16.model_struct(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            vgg.build(
                images,
                output_shape=config.output_shape)

        # Setup validation op
        scores = vgg.fc7
        preds = tf.argmax(vgg.prob, 1)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Loop through each checkpoint then test the entire validation set
    ckpts = [model_file]
    ckpt_yhat, ckpt_y, ckpt_scores, ckpt_file_array = [], [], [], []
    print '-' * 60
    print 'Beginning evaluation'
    print '-' * 60

    if config.validation_batch > len(combined_files):
        print 'Trimming validation_batch size to %s (same as # of files).' % len(
            combined_files)
        config.validation_batch = len(combined_files)

    for idx, c in tqdm(enumerate(ckpts), desc='Running checkpoints'):
        dec_scores, yhat, y, file_array = [], [], [], []
        # Initialize the graph
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(
            tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()))

        # Set up exemplar threading
        saver.restore(sess, c)
        start_time = time.time()
        num_batches = np.floor(
            len(combined_files) / float(
                config.validation_batch)).astype(int)
        for image_batch, label_batch, file_batch in tqdm(
                image_batcher(
                    start=0,
                    num_batches=num_batches,
                    images=combined_files,
                    labels=combined_labels,
                    config=config,
                    training_max=training_max,
                    training_min=training_min),
                total=num_batches):
            feed_dict = {
                images: image_batch
            }
            sc, tyh = sess.run(
                [scores, preds],
                feed_dict=feed_dict)
            dec_scores += [sc]
            yhat = np.append(yhat, tyh)
            y = np.append(y, label_batch)
            file_array = np.append(file_array, file_batch)
        ckpt_yhat.append(yhat)
        ckpt_y.append(y)
        ckpt_scores.append(dec_scores)
        ckpt_file_array.append(file_array)
        print 'Batch %d took %.1f seconds' % (
            idx, time.time() - start_time)
    sess.close()

    # Save everything
    new_dt_string = re.split('\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    np.savez(
        os.path.join(out_dir, '%s_validation_accuracies' % new_dt_string),
        ckpt_yhat=ckpt_yhat,
        ckpt_y=ckpt_y,
        ckpt_scores=ckpt_scores,
        ckpt_names=ckpts,
        combined_files=ckpt_file_array)

    # Run SVM
    predictions = clf.predict(np.concatenate(dec_scores))
    if combined_labels is not None:
        mean_acc = np.mean(predictions == y)
        p_value = randomization_test(y=y, yhat=predictions)
        print 'SVM performance: mean accuracy = %s%%, p = %.5f' % (
            mean_acc,
            p_value)
        df_col_label = 'true label'
    else:
        mean_acc, p_value = None, None
        y = np.copy(yhat)
        df_col_label = 'Dummy column (no labels supplied)'
    np.savez(
        os.path.join(out_dir, '%s_svm_test_data' % new_dt_string),
        yhat=yhat,
        y=y,
        scores=dec_scores,
        ckpts=ckpts,
        p_value=p_value)

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
                y.reshape(-1, 1))),
            columns=['files', 'guesses', df_col_label])
        df.to_csv(os.path.join(out_dir, 'prediction_file.csv'))
        print 'Saved csv to: %s' % out_dir
    except:
        print 'X' * 60
        print 'Could not save a spreadsheet of file info'
        print 'X' * 60


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--ims",
        type=str,
        dest="ims",
        default='/Users/drewlinsley/Documents/GEDI_images/human_bs',
        help="Directory containing your .tiff images.")
    parser.add_argument(
        "--dead_ims",
        type=str,
        dest="dead_ims",
        default='/Users/drewlinsley/Documents/GEDI_images/human_bs',
        help="Directory containing your .tiff images.")
    parser.add_argument(
        "--model_file",
        type=str,
        dest="model_file",
        default='/Users/drewlinsley/Desktop/trained_gedi_model/model_58600.ckpt-58600',
        help="Folder containing your trained CNN's checkpoint files.")
    parser.add_argument(
        "--output_csv",
        type=str,
        dest="output_csv",
        default='prediction_file',
        help="Name of your prediction csv file.")
    parser.add_argument(
        "--trained_svm",
        type=str,
        dest="trained_svm",
        default='trained_svm',
        help="Directory pointer to your trained svm model.")
    parser.add_argument(
        "--C",
        type=float,
        dest="C",
        default=1e-3,
        help="C parameter for your SVM.")
    args = parser.parse_args()
    test_vgg16(**vars(args))
