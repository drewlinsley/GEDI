#!/usr/bin/env python
import os, sys, re, shutil
import numpy as np
import tensorflow as tf
import glob
sys.path.append('../../') #puts model_depo on the path
sys.path.insert(0,re.split(__file__,os.path.realpath(__file__))[0]) #puts this experiment into path
from scipy.misc import imresize, imsave
from clicktionary_bubbles_config import GEDIconfig
from exp_ops.helper_functions import make_dir
from model_depo import baseline_vgg16 as vgg16
from ops import utils
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter, convolve


def gauss_filter(size, sigma, ndims=1):
    # make sure size is odd
    x = int(size)
    if x % 2 == 0:
        x = x + 1
    x_zeros = np.zeros((size, size))

    center = int(np.floor(x / 2.))

    x_zeros[center, center] = 1
    y = gaussian_filter(
        x_zeros, sigma=sigma)[:, :, None, None]
    y = np.repeat(y, ndims, axis=2)
    return y

def blur(image, kernel=3, sigma=7):
    k = tf.get_variable(
        name='blur', initializer=gauss_filter(
            kernel, sigma, ndims=int(image.get_shape()[-1])).astype(
            np.float32), dtype=tf.float32, trainable=False)
    return tf.nn.conv2d(image, k, [1, 1, 1, 1], padding='SAME')

def init_session():
    return tf.Session(config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.95))))

def load_model_vgg16(config):
    #Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            images = tf.placeholder("float", (config.heatmap_batch,) + tuple(config.model_image_size))
            vgg = vgg16.Vgg16(vgg16_npy_path=config.vgg16_weight_path)
            vgg.build(images)
    return vgg, images

def get_bubbles_heatmap(sess, model, images, input_image, class_index=None, block_size=10, block_stride=12, variant='neg'):
    # Compute bubbles heatmap of class_index for given image on model
    assert variant in ['pos', 'neg']
    # Get shape information from model
    input_shape = images.get_shape().as_list()
    batch_size = input_shape[0]
    if len(input_image.shape) == 2:
        # Grayscale to RGB
        input_image = np.dstack((input_image,)*3)
    if list(input_image.shape) != input_shape[1:]:
        # Resize to fit model
        print '  Reshaping image from %s to %s.' % (str(input_image.shape), str(input_shape[1:]))
        input_image = imresize(input_image, input_shape[1:])
    print '  Normalizing to [0, 1]'
    input_image /= np.max(input_image)

    # Prepare batch information
    coords = [None] * batch_size
    batch = np.zeros(input_shape)
    feed_dict = {images: batch}

    # Get baseline prob
    batch[0, ...] = input_image
    prob = sess.run(model.prob, feed_dict=feed_dict)[0].squeeze()
    # Get class index
    if class_index is None:
        class_index = np.argmax(prob)
    base_prob = prob[class_index]
    print '  Using class index %d (prob %.3f)' % (class_index, base_prob)

    # Prepare output (zoomed down for block_stride>1)
    output_size = [c / block_stride for c in input_shape[1:3]]
    heatmap = np.zeros(output_size)

    # Get output from one batch and put it into the heatmap
    def process_batch(n=batch_size):
        prob = sess.run(model.prob, feed_dict=feed_dict).squeeze()
        for i, c in enumerate(coords[:n]):
            heatmap[c[0], c[1]] = prob[i, class_index] - base_prob

    # Accumulate image regions into batch and process them
    i_batch = 0
    print ('  Processing %s...\n  ' % str(output_size)),
    for iy in xrange(output_size[0]):
        print str(iy),
        y = iy * block_stride
        for ix in xrange(output_size[1]):
            x = ix * block_stride
            y0 = max(0, y - block_size / 2)
            y1 = min(input_shape[1], y + (block_size + 1) / 2)
            x0 = max(0, x - block_size / 2)
            x1 = min(input_shape[2], x + (block_size + 1) / 2)
            if variant == 'pos':
                batch[i_batch, ...] = 0
                batch[i_batch, y0:y1, x0:x1, :] = input_image[y0:y1, x0:x1, :]
            else:
                batch[i_batch, ...] = input_image
                batch[i_batch, y0:y1, x0:x1, :] = 0
            coords[i_batch] = [iy, ix]
            i_batch += 1
            if i_batch == batch_size:
                print ".",
                process_batch()
                i_batch = 0
        if not (iy % 10): print '\n  ',
    # Process remainder
    if i_batch:
        process_batch(i_batch)
    print '  Heatmap done.'

    # Undo zoom
    heatmap = zoom(heatmap, block_stride)

    # Reverse signal of blanked-out
    if variant == 'neg':
        heatmap = -heatmap

    return heatmap, base_prob

def get_heatmap_filename(output_path, image_filename):
    # Derive filename to save heatmap in from model + image
    return os.path.join(output_path, re.split('\.', re.split('/', image_filename)[-1])[0])

def generate_heatmaps_for_images(config, ckpt, image_filenames, labels, model_name, method_name, variant, block_size=10, block_stride=1, generate_plots=False, use_true_label=True):
    # Generate all heatmaps for images in list
    if generate_plots:
        import matplotlib.pyplot as plt
    # Get class indices for all files
    label_key = np.asarray(config.label_directories)
    class_indices = labels  # [np.where(config.which_dataset + '_' + fn == label_key) for fn in labels]
    print class_indices
    # Process all files
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [1])

    with tf.variable_scope('cnn'):
        vgg = vgg16.Vgg16(
            vgg16_npy_path=config.vgg16_weight_path)
        train_mode = tf.get_variable(name='training', initializer=False)
        vgg.build(
            images,
            output_shape=1000,
            train_mode=train_mode)

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))

    if config.restore_model is not None:
        saver.restore(sess, config.restore_model)
    attention_labels = tf.one_hot(
        labels, 1000, dtype=tf.float32)
    heatmap_op = tf.gradients(
        vgg.fc8 * attention_labels,
        images)[0]

    filt = gauss_filter(20, 7, ndims=1)
    print 'Working on %s indices and %s filenames' % (len(class_indices), len(image_filenames))
    for class_index, image_filename in zip(class_indices, image_filenames):
        heatmap_filename = get_heatmap_filename(output_path=config.visualization_output, image_filename=image_filename)
        print 'Heatmap for %s...' % os.path.basename(heatmap_filename)
        img = utils.load_image(image_filename)
        if len(img.shape) == 2:
            img = np.repeat(utils.load_image(image_filename)[:, :, None], 3, axis=-1)
        img = img / np.max(img)
        img = img[None, :, :, :]
        feed_dict = {images: img, labels: [class_index]}
        heatmap = sess.run(heatmap_op, feed_dict=feed_dict)# [0, :, :, 0]
        import ipdb;ipdb.set_trace()
        print ' Saving heatmap to %s...' % heatmap_filename
        heatmap = np.squeeze(convolve(heatmap, np.squeeze(filt)))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) 
        np.save(heatmap_filename, heatmap)
        if generate_plots:
            f, axarr = plt.subplots(2, 1)
            axarr[0].imshow(np.squeeze(img), cmap='gray')
            m = axarr[1].matshow(heatmap)
            f.colorbar(m)
            plt.savefig(heatmap_filename + '.png')
            plt.close()

def match_filenames_labels(images,label_file):
    labels = []
    with open(label_file,'r') as f:
        for line in f:
            labels.append(line.split('\n')[0])
    label_lables = np.asarray([re.split('_',re.split('/',x)[-2])[-1] for x in labels])
    label_images = [re.split('/',x)[-1] for x in labels]
    image_images = [re.split('/',x)[-1] for x in images]
    label_idx = []
    for idx in image_images:
        label_idx = np.append(label_idx,label_lables[np.where(np.asarray([idx in x for x in label_images]))[0][0]])
    return label_idx

def random_sample_images(labels, in_dict, out_dir, im_ext, amount):
    image_filenames = glob.glob(labels + '/*')
    labels = []
    for im in image_filenames:
        for k, v in in_dict.iteritems():
            label_name = re.split('\d+', re.split('/', im)[-1])[0]
            if label_name == k:
                labels += [v]
    return image_filenames, labels

if __name__ == "__main__":
    config = GEDIconfig()

    #Make directories if they haven't been made yet
    dir_list = [config.heatmap_source_images,config.heatmap_dataset_images]
    [make_dir(d) for d in dir_list]
    image_filenames,image_labels = random_sample_images(config.heatmap_image_labels,config.heatmap_image_dict, config.heatmap_dataset_images, config.im_ext, config.heatmap_image_amount)
    print len(image_filenames)
    #Run bubbles
    image_labels = [int(x.strip('\n')) for x in image_labels]
    generate_heatmaps_for_images(config, None, image_filenames, image_labels, 'vgg16', 'bubbles', 'neg', block_size=config.block_size, block_stride=config.block_stride, generate_plots=config.generate_plots, use_true_label=config.use_true_label)
