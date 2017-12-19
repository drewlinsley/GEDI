"""Visualize a specified VGG16 and label according to disease line."""

import os
import time
import re
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
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA


# TODO: Do a Live/Dead autopsy

def crop_center(img, crop_size):
    """Center crop an image."""
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def renormalize(img, max_value, min_value):
    """Normalize image to [0, 1]."""
    return (img - min_value) / (max_value - min_value)


def image_batcher(
        start,
        num_batches,
        images,
        config,
        training_max,
        training_min):
    """Yield processed image batches for training."""
    for b in range(num_batches):
        next_image_batch = images[start:start + config.validation_batch]
        image_stack = []
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
        yield np.concatenate(image_stack, axis=0), next_image_batch


def randomization_test(y, yhat, iterations=10000):
    """Randomization hypothesis testing."""
    true_score = np.mean(y == yhat)
    perm_scores = np.zeros((iterations))
    lab_len = len(y)
    for it in range(iterations):
        perm_scores[it] = np.mean(
            yhat == np.copy(y)[np.random.permutation(lab_len)])
    p_value = (np.sum(true_score < perm_scores) + 1) / float(iterations + 1)
    return p_value


# Evaluate your trained model on GEDI images
def test_vgg16(
        image_dir,
        model_file,
        autopsy_csv=None,
        autopsy_path=None,
        output_csv='prediction_file',
        target_layer='fc7',
        save_npy=False,
        shuffle_images=True,
        embedding_type='PCA'):
    """Testing function for pretrained vgg16."""
    assert autopsy_csv is not None, 'You must pass an autopsy file name.'
    assert autopsy_path is not None, 'You must pass an autopsy path.'

    # Load autopsy information
    autopsy_data = pd.read_csv(
        os.path.join(
            autopsy_path,
            autopsy_csv))

    # Load config and begin preparing data
    config = GEDIconfig()
    if image_dir is None:
        raise RuntimeError(
            'You need to supply a directory path to the images.')

    combined_files = np.asarray(
        glob(os.path.join(image_dir, '*%s' % config.raw_im_ext)))
    if shuffle_images:
        combined_files = combined_files[np.random.permutation(
            len(combined_files))]
    if len(combined_files) == 0:
        raise RuntimeError('Could not find any files. Check your image path.')

    config = GEDIconfig()
    meta_file_pointer = os.path.join(
        model_file.split('/model')[0], 'train_maximum_value.npz')
    if not os.path.exists(meta_file_pointer):
        raise RuntimeError(
            'Cannot find the training data meta file.'
            'Download this from the link described in the README.md.')
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
            vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            vgg.build(
                images,
                output_shape=config.output_shape)

        # Setup validation op
        scores = vgg[target_layer]
        preds = tf.argmax(vgg.prob, 1)

    # Derive pathologies from file names
    pathologies = []
    for f in combined_files:
        sf = f.split('/')[-1].split('_')
        sf = '_'.join(sf[1:4])
        it_path = autopsy_data[
            autopsy_data['plate_well_neuron'] == sf]['disease']
        if not len(it_path):
            it_path = 'Absent'
        else:
            it_path = it_path.as_matrix()[0]
        pathologies += [it_path]
    pathologies = np.asarray(pathologies)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Loop through each checkpoint then test the entire validation set
    ckpts = [model_file]
    ckpt_yhat, ckpt_scores, ckpt_file_array = [], [], []
    print '-' * 60
    print 'Beginning evaluation'
    print '-' * 60

    if config.validation_batch > len(combined_files):
        print 'Trimming validation_batch size to %s.' % len(combined_files)
        config.validation_batch = len(combined_files)

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
        num_batches = np.floor(
            len(combined_files) / float(
                config.validation_batch)).astype(int)
        for image_batch, file_batch in tqdm(
                image_batcher(
                    start=0,
                    num_batches=num_batches,
                    images=combined_files,
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
            yhat += [tyh]
            file_array += [file_batch]
        ckpt_yhat.append(yhat)
        ckpt_scores.append(dec_scores)
        ckpt_file_array.append(file_array)
        print 'Batch %d took %.1f seconds' % (
            idx, time.time() - start_time)
    sess.close()

    # Create and plot an embedding
    im_path_map = pathologies[:num_batches * config.validation_batch]
    dec_scores = np.concatenate(dec_scores)
    mu = dec_scores.mean(0)[None, :]
    sd = dec_scores.std(0)[None, :]
    dec_scores = (dec_scores - mu) / sd
    yhat = np.concatenate(yhat)
    file_array = np.concatenate(file_array)

    if embedding_type == 'TSNE' or embedding_type == 'tsne':
        emb = manifold.TSNE(n_components=2, init='pca', random_state=0)
    elif embedding_type == 'PCA' or embedding_type == 'pca':
        emb = PCA(n_components=2, svd_solver='randomized', random_state=0)
    elif embedding_type == 'spectral':
        emb = manifold.SpectralEmbedding(n_components=2, random_state=0)
    y = emb.fit_transform(dec_scores)

    # Ouput csv
    df = pd.DataFrame(
        np.hstack((y, im_path_map.reshape(-1, 1), file_array.reshape(-1, 1))),
        columns=['D1', 'D2', 'pathology', 'filename'])
    out_name = os.path.join(out_dir, 'embedding.csv')
    df.to_csv(out_name)
    print 'Saved csv to: %s' % out_name

    # Create plot
    f, ax = plt.subplots()
    unique_cats = np.unique(im_path_map)
    h = []
    for idx, cat in enumerate(unique_cats):
        h += [plt.scatter(
            y[im_path_map == cat, 0],
            y[im_path_map == cat, 1],
            c=plt.cm.Spectral(idx * 1000))]
    plt.legend(h, unique_cats)
    plt.axis('tight')
    plt.show()
    plt.savefig('embedding.png')
    plt.close(f)

    # Save everything
    if save_npy:
        np.savez(
            os.path.join(out_dir, 'validation_accuracies'),
            ckpt_yhat=ckpt_yhat,
            ckpt_scores=ckpt_scores,
            ckpt_names=ckpts,
            combined_files=ckpt_file_array)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        dest="image_dir",
        default='/Users/drewlinsley/Documents/GEDI_images/cache_of_RGEDIMachineLearning26',
        help="Directory containing your .tiff images.")
    parser.add_argument(
        "--autopsy_path",
        type=str,
        dest="autopsy_path",
        default='timecourse_processing',
        help="Directory containing your processed autopsy file.")
    parser.add_argument(
        "--autopsy_csv",
        type=str,
        dest="autopsy_csv",
        default='processed_autopsy_info.csv',
        help="CSV file with your autopsy info.")
    parser.add_argument(
        "--model_file",
        type=str,
        dest="model_file",
        default='/Users/drewlinsley/Desktop/trained_gedi_model/model_58600.ckpt-58600',
        help="Folder containing your trained CNN's checkpoint files.")
    parser.add_argument(
        "--target_layer",
        type=str,
        dest="target_layer",
        default='fc7',
        help="Folder containing your trained CNN's checkpoint files.")
    parser.add_argument(
        "--save_npy",
        dest="save_npy",
        action='store_true',
        help='Save a numpy of the data.')
    args = parser.parse_args()
    test_vgg16(**vars(args))
