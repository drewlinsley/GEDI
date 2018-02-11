import os
import time
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from glob import glob
from exp_ops import tf_fun
from gedi_config import GEDIconfig
from skimage import io
from tqdm import tqdm
from sklearn import manifold
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from models import matching_vgg16 as vgg16


def process_image(
        filename,
        model_input_shape,
        num_panels=1,
        normalize=True):
    """Process images for the matching model with numpy."""
    image = io.imread(filename)
    im_shape = image.shape
    image = image.reshape(im_shape[0], im_shape[0], num_panels)

    # Split the images
    split_image = np.split(image, num_panels, axis=-1)

    # Insert augmentation and preprocessing here
    ims, filenames = [], []
    for idx, im in enumerate(split_image):
        ims += [tf_fun.crop_center(im, model_input_shape)]
        filenames += ['%s %s' % (filename, idx)]

    ims = np.asarray(ims)
    if normalize:
        ims /= ims.max(0, keepdims=True).max(1, keepdims=True)
        ims = np.minimum(np.maximum(ims, 1), 0)
    return ims, np.asarray(filenames)


def image_batcher(
        start,
        num_batches,
        images,
        config,
        n_images):
    for b in range(num_batches):
        next_image_batch = images[start:start + config.validation_batch]
        image_stack = []
        for f in next_image_batch:
            # 1. Load image patch
            patch = process_image(
                filename=f,
                model_input_shape=config.model_image_size,
                num_panels=n_images,
                normalize=config.normalize).astype(np.float32)
            # 2. Repeat to 3 channel (RGB) image
            patch = np.repeat(patch[:, :, None], 3, axis=-1)
            # 3. Add to list
            image_stack += [patch[None, :, :, :]]
        # Add dimensions and concatenate
        start += config.validation_batch
        yield np.concatenate(image_stack, axis=0), next_image_batch


def test_placeholder(
        image_path=None,
        model_file=None,
        n_images=1,
        debug=True,
        margin=.1,
        autopsy_path=None,
        autopsy_csv=None,
        embedding_type='PCA'):
    config = GEDIconfig()
    assert margin is not None, 'Need a margin for the loss.'
    assert image_path is not None, 'Provide a path to an image directory.'
    assert model_file is not None, 'Provide a path to the model file.'

    try:
        # Load the model's config
        config = glob(os.path.join(model_file, '*.npy'))
    except IOError:
        'Could not load model config, falling back to default config.'
    try:
        # Load autopsy information
        autopsy_data = pd.read_csv(
            os.path.join(
                autopsy_path,
                autopsy_csv))
    except IOError:
        print 'Unable to load autopsy file.'
    combined_files = np.asarray(
        glob(os.path.join(image_path, '*%s' % config.raw_im_ext)))
    if len(combined_files) == 0:
        raise RuntimeError('Could not find any files. Check your image path.')

    # Make output directories if they do not exist
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.which_dataset + '_' + dt_stamp + '/'
    config.train_checkpoint = os.path.join(
        config.train_checkpoint, dt_dataset)  # timestamp this run
    out_dir = os.path.join(config.results, dt_dataset)
    dir_list = [out_dir]
    [tf_fun.make_dir(d) for d in dir_list]

    print '-' * 60
    print('Training model:' + dt_dataset)
    print '-' * 60

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        images = []
        for idx in range(n_images):
            images += [tf.placeholder(
                tf.float32,
                shape=[None] + config.model_image_size,
                name='images %s' % idx)]

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('match'):
            # Build matching model for frame 0
            model_0 = vgg16.model_struct(
                vgg16_npy_path=config.gedi_weight_path)  # ,
            frame_activity = []
            model_activity = model_0.build(
                images[0],
                output_shape=config.output_shape,
                include_GEDI=config.include_GEDI)
            if config.l2_norm:
                model_activity = [model_activity]
            frame_activity += [model_activity]
        if n_images > 1:
            with tf.variable_scope('match', reuse=tf.AUTO_REUSE):
                # Build matching model for other frames
                for idx in range(1, len(images)):
                    model_activity = model_0.build(
                        images[idx],
                        output_shape=config.output_shape,
                        include_GEDI=config.include_GEDI)
                    if config.l2_norm:
                        model_activity = tf_fun.l2_normalize(model_activity)
                    frame_activity += [model_activity]
            if config.dist_fun == 'l2':
                pos = tf_fun.l2_dist(
                    frame_activity[0],
                    frame_activity[1], axis=1)
                neg = tf_fun.l2_dist(
                    frame_activity[0],
                    frame_activity[2], axis=1)
            elif config.dist_fun == 'pearson':
                pos = tf_fun.pearson_dist(
                    frame_activity[0],
                    frame_activity[1],
                    axis=1)
                neg = tf_fun.pearson_dist(
                    frame_activity[0],
                    frame_activity[2],
                    axis=1)
            model_activity = pos - neg  # Store the difference in distances

    if config.validation_batch > len(combined_files):
        print (
            'Trimming validation_batch size to %s '
            '(same as # of files).' % len(combined_files))
        config.validation_batch = len(combined_files)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))

    # Set up exemplar threading
    saver.restore(sess, model_file)
    start_time = time.time()
    num_batches = np.floor(
        len(combined_files) / float(
            config.validation_batch)).astype(int)
    score_array, file_array = [], []
    for image_batch, file_batch in tqdm(
            image_batcher(
                start=0,
                num_batches=num_batches,
                images=combined_files,
                config=config,
                n_images=n_images),
            total=num_batches):
        feed_dict = {
            images: image_batch
        }
        activity = sess.run(
            model_activity,
            feed_dict=feed_dict)
        score_array += [activity]
        file_array = np.append(file_array, file_batch)
    print 'Image processing %d took %.1f seconds' % (
        idx, time.time() - start_time)
    sess.close()
    score_array = np.asarray(score_array)
    file_array = np.asarray(file_array)

    # Save everything
    np.savez(
        os.path.join(out_dir, 'validation_accuracies'),
        score_array=score_array,
        file_array=file_array)

    if n_images == 1:
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
        pathologies = np.asarray(pathologies)[:len(score_array)]

        if embedding_type == 'TSNE' or embedding_type == 'tsne':
            emb = manifold.TSNE(n_components=2, init='pca', random_state=0)
        elif embedding_type == 'PCA' or embedding_type == 'pca':
            emb = PCA(n_components=2, svd_solver='randomized', random_state=0)
        elif embedding_type == 'spectral':
            emb = manifold.SpectralEmbedding(n_components=2, random_state=0)
        y = emb.fit_transform(score_array)

        # Ouput csv
        df = pd.DataFrame(
            np.hstack((
                y,
                pathologies.reshape(-1, 1),
                file_array.reshape(-1, 1))),
            columns=['dim1', 'dim2', 'pathology', 'filename'])
        out_name = os.path.join(out_dir, 'embedding.csv')
        df.to_csv(out_name)
        print 'Saved csv to: %s' % out_name

        # Create plot
        f, ax = plt.subplots()
        unique_cats = np.unique(pathologies)
        h = []
        for idx, cat in enumerate(unique_cats):
            h += [plt.scatter(
                y[pathologies == cat, 0],
                y[pathologies == cat, 1],
                c=plt.cm.Spectral(idx * 1000))]
        plt.legend(h, unique_cats)
        plt.axis('tight')
        plt.show()
        plt.savefig('embedding.png')
        plt.close(f)
    else:
        # Do a classification (sign of the score)
        decisions = np.sign(score_array)
        df = pd.DataFrame(
            np.hstack(decisions, score_array),
            columns=['Decisions', 'Scores'])
        df.to_csv(
            os.path.join(
                out_dir, 'tracking_model_scores.csv'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--image_path', type=str, dest='image_path',
        default=None, help='Directory with tiff images.')
    parser.add_argument(
        '--model_file', type=str, dest='model_file',
        default=None, help='Path to the model files.')
    parser.add_argument(
        '--n_images', type=str, dest='n_images',
        default=None, help='Number of images in each exemplar.')
    parser.add_argument(
        '--autopsy_path',
        type=str,
        dest='autopsy_path',
        default='timecourse_processing',
        help='Directory containing your processed autopsy file.')
    parser.add_argument(
        '--autopsy_csv',
        type=str,
        dest='autopsy_csv',
        default='processed_autopsy_info.csv',
        help='CSV file with your autopsy info.')
    args = parser.parse_args()
    test_placeholder(**vars(args))