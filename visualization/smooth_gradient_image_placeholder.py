import os
import time
import re
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from glob import glob
from exp_ops.tf_fun import make_dir
from exp_ops.preprocessing_GEDI_images import produce_patch
from gedi_config import GEDIconfig
import ipdb;ipdb.set_trace()
from models import baseline_vgg16 as vgg16
from matplotlib import pyplot as plt
from tqdm import tqdm


def flatten_list(l):
    """Flatten a list of lists."""
    return [val for sublist in l for val in sublist]


def save_images(
        y,
        yhat,
        viz,
        files,
        output_folder,
        target,
        label_dict,
        ext='.png'):
    """Save TP/FP/TN/FN images in separate folders."""
    quality = ['true', 'false']
    folders = [[os.path.join(
        output_folder, '%s_%s' % (
            k, quality[0])), os.path.join(
        output_folder, '%s_%s' % (
            k, quality[1]))] for k in label_dict.keys()]
    flat_folders = flatten_list(folders)
    [make_dir(f) for f in flat_folders]
    for iy, iyhat, iviz, ifiles in zip(y, yhat, viz, files):
        correct = iy == iyhat
        target_label = iy == target
        f = plt.figure()
        plt.imshow(iviz)
        it_f = ifiles.split('/')[-1].split('\.')[0]
        if correct and target_label:
            # TP
            it_folder = folders[0][0]
        elif correct and not target_label:
            # TN
            it_folder = folders[0][1]
        elif not correct and target_label:
            # FP
            it_folder = folders[1][0]
        elif not correct and not target_label:
            # FN
            it_folder = folders[1][1]
        plt.title('Predicted label=%s, true label=%s' % (iyhat, iy))
        plt.savefig(
            os.path.join(
                it_folder,
                '%s%s' % (it_f, ext)))
        plt.close(f)


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def visualization_function(images, viz):
    """Wrapper for summarizing visualizations across channels."""
    if viz == 'sum_abs':
        return np.sum(np.abs(images), axis=-1)
    elif viz == 'sum_p':
        return np.sum(np.pow(images, 2), axis=-1)
    else:
        raise RuntimeError('Visualization method not implemented.')


def add_noise(image_batch, loc=0, scale=0.15 / 255):
    """Add gaussian noise to the input for smoothing visualizations."""
    return np.copy(image_batch) + np.random.normal(
        size=image_batch.shape, loc=loc, scale=scale)


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


# Evaluate your trained model on GEDI images
def visualize_model(
        live_ims,
        dead_ims,
        model_file,
        output_folder,
        smooth_iterations=50,
        untargeted=False,
        viz='sum_abs'):
    """Train an SVM for your dataset on GEDI-model encodings."""
    config = GEDIconfig()
    if live_ims is None:
        raise RuntimeError(
            'You need to supply a directory path to the live images.')
    if dead_ims is None:
        raise RuntimeError(
            'You need to supply a directory path to the dead images.')

    live_files = glob(os.path.join(live_ims, '*%s' % config.raw_im_ext))
    dead_files = glob(os.path.join(dead_ims, '*%s' % config.raw_im_ext))
    combined_labels = np.concatenate((
        np.zeros(len(live_files)),
        np.ones(len(dead_files))))
    combined_files = np.concatenate((live_files, dead_files))
    if len(combined_files) == 0:
        raise RuntimeError('Could not find any files. Check your image path.')

    config = GEDIconfig()
    meta_file_pointer = os.path.join(
        model_file.split('/model')[0], 'train_maximum_value.npz')
    if not os.path.exists(meta_file_pointer):
        raise RuntimeError(
            'Cannot find the training data meta file. Download this from the link described in the README.md.')
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
    labels = tf.placeholder(
        tf.int64,
        shape=[None],
        name='labels')

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
        scores = vgg.fc7
        preds = tf.argmax(vgg.prob, 1)
        activity_pattern = vgg.fc8
        if not untargeted:
            oh_labels = tf.one_hot(labels, config.output_shape)
            activity_pattern *= oh_labels
        grad_image = tf.gradients(activity_pattern, images)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Loop through each checkpoint then test the entire validation set
    ckpts = [model_file]
    ckpt_yhat, ckpt_y, ckpt_scores, ckpt_file_array, ckpt_viz_images = [], [], [], [], []
    print '-' * 60
    print 'Beginning evaluation'
    print '-' * 60

    if config.validation_batch > len(combined_files):
        print 'Trimming validation_batch size to %s (same as # of files).' % len(
            combined_files)
        config.validation_batch = len(combined_files)

    for idx, c in tqdm(enumerate(ckpts), desc='Running checkpoints'):
        dec_scores, yhat, y, file_array, viz_images = [], [], [], [], []
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
                images: image_batch,
                labels: label_batch
            }
            it_grads = np.zeros((image_batch.shape))
            sc, tyh = sess.run(
                [scores, preds],
                feed_dict=feed_dict)
            for idx in range(smooth_iterations):
                feed_dict = {
                    images: add_noise(image_batch),
                    labels: label_batch
                }
                it_grad = sess.run(
                    grad_image,
                    feed_dict=feed_dict)
                it_grads += it_grad[0]
            import ipdb;ipdb.set_trace()
            it_grads /= smooth_iterations  # Mean across iterations
            it_grads = visualization_function(it_grads, viz)
            dec_scores += [sc]
            yhat = np.append(yhat, tyh)
            y = np.append(y, label_batch)
            file_array = np.append(file_array, file_batch)
            viz_images += [it_grads]
        ckpt_yhat.append(yhat)
        ckpt_y.append(y)
        ckpt_scores.append(dec_scores)
        ckpt_file_array.append(file_array)
        ckpt_viz_images.append(viz_images)
        print 'Batch %d took %.1f seconds' % (
            idx, time.time() - start_time)
    sess.close()

    # Save everything
    np.savez(
        os.path.join(out_dir, 'validation_accuracies'),
        ckpt_yhat=ckpt_yhat,
        ckpt_y=ckpt_y,
        ckpt_scores=ckpt_scores,
        ckpt_names=ckpts,
        combined_files=ckpt_file_array,
        ckpt_viz_images=ckpt_viz_images)

    # Save images
    save_images(
        y=ckpt_y,
        yhat=ckpt_yhat,
        viz=ckpt_viz_images,
        files=ckpt_file_array,
        output_folder=output_folder,
        target='dead',
        label_dict={
            'live': 0,
            'dead': 1
        })


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--live_ims",
        type=str,
        dest="live_ims",
        default='/Users/drewlinsley/Documents/GEDI_images/Live_bs_rat',
        help="Directory containing your Live .tiff images.")
    parser.add_argument(
        "--dead_ims",
        type=str,
        dest="dead_ims",
        default='/Users/drewlinsley/Documents/GEDI_images/Dead_bs_rat',
        help="Directory containing your Dead .tiff images.")
    parser.add_argument(
        "--model_file",
        type=str,
        dest="model_file",
        default='/Users/drewlinsley/Desktop/trained_gedi_model/model_58600.ckpt-58600',
        help="Folder containing your trained CNN's checkpoint files.")
    parser.add_argument(
        "--untargeted",
        dest="untargeted",
        action='store_true',
        help='Visualize overall saliency instead of features related to the most likely category.')
    parser.add_argument(
        "--smooth_iterations",
        type=int,
        dest="model_file",
        default=10,
        help='Number of iterations of smoothing for visualizations.')
    parser.add_argument(
        "--visualization",
        type=str,
        dest="viz",
        default='sum_abs',
        help='Number of iterations of smoothing for visualizations.')
    parser.add_argument(
        "--output_folder",
        type=str,
        dest="output_folder",
        default='gradient_images',
        help='Folder to save the visualizations.')

    args = parser.parse_args()
    visualize_model(**vars(args))
