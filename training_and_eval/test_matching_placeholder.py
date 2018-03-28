import os
import time
import re
import tensorflow as tf
import numpy as np
import seaborn as sns
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


def create_figs(emb, out_dir, out_name, embedding_type, embedding_name):
    if embedding_type == 'PCA':
        plt.plot(emb.explained_variance_ratio_)
        plt.title('Percentage of variance explained by the PCs.')
        plt.savefig(
            os.path.join(
                out_dir,
                'eigen_%s.png' % embedding_name))

    # Create plot
    plt_df = pd.read_csv(out_name)
    sns.lmplot(
        x='dim1',
        y='dim2',
        data=plt_df,
        fit_reg=False,
        hue='pathology',
        legend=False)
    plt.legend(loc='lower right')
    plt.savefig('%s.png' % embedding_name)
    sns.plt.show()


def process_image(
        filename,
        model_input_shape,
        channel_number=0,
        num_panels=3,
        first_n_images=1,
        normalize=True):
    """Process images for the matching model with numpy."""
    image = io.imread(filename).astype(np.float32)
    im_shape = image.shape
    if len(im_shape) == 3:
        # Multi-timestep image
        image = image[channel_number]
    elif len(im_shape) == 2:
        pass
    else:
        raise RuntimeError('Cannot understand the dimensions of your image.')
    if 3 * im_shape[0] == im_shape[1]:
        split_image = np.split(image, 3, axis=-1)

        # Insert augmentation and preprocessing here
        ims, filenames = [], []
        for idx in range(first_n_images):
            ims += [tf_fun.crop_center(
                split_image[idx], model_input_shape[:2])]
            filenames += ['%s %s' % (filename, idx)]
        ims = np.asarray(ims).astype(np.float32)
        np.asarray(filenames)
        if normalize:
            ims /= ims.max(1, keepdims=True).max(2, keepdims=True)
            ims = np.minimum(np.maximum(ims, 1), 0)
    else:
        ims = np.asarray(image).astype(np.float32)
        ims = tf_fun.crop_center(
            ims, model_input_shape[:2])
        filenames = np.asarray(filename)
        if normalize:
            ims /= (ims.max() + 1e-12)
            ims = np.maximum(np.minimum(ims, 1), 0)
    return ims, np.asarray(filenames)


def image_batcher(
        start,
        num_batches,
        images,
        config,
        first_n_images,
        n_images):
    for b in range(num_batches):
        next_image_batch = images[start:start + config.validation_batch]
        image_stack = []
        image_filenames = []
        for f in next_image_batch:
            # 1. Load image patch
            patches, filenames = process_image(
                filename=f,
                channel_number=config.channel,
                model_input_shape=config.model_image_size,
                num_panels=n_images,
                first_n_images=first_n_images,
                normalize=config.normalize)
            # 2. Repeat to 3 channel (RGB) image
            pshape = patches.shape
            # import ipdb;ipdb.set_trace()
            # if pshape[0] > 1:
            #     raise NotImplementedError(
            #    'Unfinished multi-image processing.')
            image_filenames += [filenames]
            # 3. Add to list
            if len(pshape) < 4:
                patches = np.expand_dims(patches, axis=-1)
            image_stack += [patches]
        # Add dimensions and concatenate
        start += config.validation_batch
        yield np.asarray(image_stack), np.asarray(image_filenames)


def test_placeholder(
        image_path,
        model_file,
        model_meta,
        out_dir,
        n_images=3,
        first_n_images=1,
        debug=True,
        margin=.1,
        autopsy_csv=None,
        embedding_type='tsne',
        autopsy_model='match'):
    config = GEDIconfig()
    assert margin is not None, 'Need a margin for the loss.'
    assert image_path is not None, 'Provide a path to an image directory.'
    assert model_file is not None, 'Provide a path to the model file.'

    try:
        # Load the model's config
        config = np.load(model_meta).item()
    except:
        print 'Could not load model config, falling back to default config.'
    config.model_image_size[-1] = 1
    try:
        # Load autopsy information
        autopsy_data = pd.read_csv(autopsy_csv)
    except IOError:
        print 'Unable to load autopsy file.'
    if not hasattr(config, 'include_GEDI'):
        raise RuntimeError('You need to pass the correct meta file.')
        config.include_GEDI = True
        config.l2_norm = False
        config.dist_fun = 'pearson'
        config.per_batch = False
        config.output_shape = 32
        config.margin = 0.1
    if os.path.isdir(image_path):
        combined_files = np.asarray(
            glob(os.path.join(image_path, '*%s' % config.raw_im_ext)))
    else:
        combined_files = [image_path]
    if len(combined_files) == 0:
        raise RuntimeError('Could not find any files. Check your image path.')

    # Make output directories if they do not exist
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.which_dataset + '_' + dt_stamp + '/'
    config.train_checkpoint = os.path.join(
        config.train_checkpoint, dt_dataset)  # timestamp this run
    out_dir = os.path.join(out_dir, dt_dataset)
    dir_list = [out_dir]
    [tf_fun.make_dir(d) for d in dir_list]

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        images = []
        for idx in range(first_n_images):
            images += [tf.placeholder(
                tf.float32,
                shape=[None] + config.model_image_size,
                name='images_%s' % idx)]

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        if autopsy_model == 'match':
            from models import matching_vgg16 as model_type
            with tf.variable_scope('match'):
                # Build matching model for frame 0
                model_0 = model_type.model_struct(
                    vgg16_npy_path=config.gedi_weight_path)  # ,
                frame_activity = []
                model_activity = model_0.build(
                    images[0],
                    output_shape=config.output_shape,
                    include_GEDI=config.include_GEDI)
                if config.l2_norm:
                    model_activity = [model_activity]
                frame_activity += [model_activity]
            if first_n_images > 1:
                with tf.variable_scope('match', reuse=tf.AUTO_REUSE):
                    # Build matching model for other frames
                    for idx in range(1, len(images)):
                        model_activity = model_0.build(
                            images[idx],
                            output_shape=config.output_shape,
                            include_GEDI=config.include_GEDI)
                        if config.l2_norm:
                            model_activity = tf_fun.l2_normalize(
                                model_activity)
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
        elif autopsy_model == 'GEDI' or autopsy_model == 'gedi':
            from models import baseline_vgg16 as model_type
            model = model_type.model_struct(
                vgg16_npy_path=config.gedi_weight_path)  # ,
            model.build(
                images[0],
                output_shape=config.output_shape)
            model_activity = model.fc7
        else:
            raise NotImplementedError(autopsy_model)

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
    if autopsy_model == 'match':
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
                first_n_images=first_n_images,
                n_images=n_images),
            total=num_batches):
        for im_head in images:
            feed_dict = {
                im_head: image_batch
            }
            activity = sess.run(
                model_activity,
                feed_dict=feed_dict)
            score_array += [activity]
        file_array += [file_batch]
    print 'Image processing %d took %.1f seconds' % (
        idx, time.time() - start_time)
    sess.close()
    score_array = np.concatenate(score_array, axis=0)
    score_array = score_array.reshape(-1, score_array.shape[-1])
    file_array = np.concatenate(file_array, axis=0)

    # Save everything
    np.savez(
        os.path.join(out_dir, 'validation_accuracies'),
        score_array=score_array,
        file_array=file_array)

    if first_n_images == 1:
        # Derive pathologies from file names
        pathologies = []
        for f in combined_files:
            sf = f.split(os.path.sep)[-1].split('_')
            line = sf[1]
            # time_col = sf[2]
            well = sf[4]
            disease = autopsy_data[
                np.logical_and(
                    autopsy_data['line'] == line,
                    autopsy_data['wells'] == well)]['type']
            try:
                disease = disease.as_matrix()[0]
            except:
                disease = 'Not_found'
            pathologies += [disease]
        pathologies = np.asarray(pathologies)[:len(score_array)]

        mu = score_array.mean(0)
        sd = score_array.std(0)
        z_score_array = (score_array - mu) / (sd + 1e-4)
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
        out_name = os.path.join(out_dir, 'raw_embedding.csv')
        df.to_csv(out_name)
        print 'Saved csv to: %s' % out_name

        create_figs(
            emb=emb,
            out_dir=out_dir,
            out_name=out_name,
            embedding_type=embedding_type,
            embedding_name='raw_embedding')

        # Now work on zscored data
        y = emb.fit_transform(z_score_array)

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
        create_figs(
            emb=emb,
            out_dir=out_dir,
            out_name=out_name,
            embedding_type=embedding_type,
            embedding_name='normalized_embedding')

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
        '--image_path',
        type=str,
        dest='image_path',
        default='/home/drew/tissue/GEDI/combined',  # None,
        help='Directory with tiff images.')
    parser.add_argument(
        '--model_file',
        type=str,
        dest='model_file',
        default='/media/data/GEDI/drew_images/project_files/train_checkpoint/gfp_2018_03_21_19_24_54/model_225500.ckpt-225500',  # None,
        help='Path to the model checkpoint file.')
    parser.add_argument(
        '--model_meta',
        type=str,
        dest='model_meta',
        default='/media/data/GEDI/drew_images/project_files/results/gfp_2018_03_21_19_24_54/meta_info.npy',  # None,
        help='Path to the model meta file.')
    parser.add_argument(
        '--n_images',
        type=int,
        dest='n_images',
        default=3,
        help='Number of images in each exemplar.')
    parser.add_argument(
        '--first_n_images',
        type=int,
        dest='n_images',
        default=1,
        help='Analyze the first n images.')
    parser.add_argument(
        '--autopsy_csv',
        type=str,
        dest='autopsy_csv',
        default='timecourse_processing/autopsy_huntington_parkinson_3_14_18.csv',
        help='Full path to the CSV file with your autopsy info.')
    parser.add_argument(
        '--model_type',
        type=str,
        dest='autopsy_model',
        default='match',
        help='Type of autopsy model [match/gedi].')
    parser.add_argument(
        '--embedding',
        type=str,
        dest='embedding_type',
        default='tsne',
        help='Type of embedding [tsne/pca/spectral]')
    parser.add_argument(
        '--out_dir',
        type=str,
        dest='out_dir',
        default='autopsy_results',
        help='Output directory.')
    args = parser.parse_args()
    test_placeholder(**vars(args))

