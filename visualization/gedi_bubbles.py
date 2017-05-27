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
from model_depo import vgg16_trainable_batchnorm as vgg16
from ops import utils
from scipy.ndimage.interpolation import zoom

def init_session():
    return tf.Session(config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.95))))

def load_model_vgg16(config):
    #Prepare model on GPU
    with tf.device('/gpu:0'):
        images = tf.placeholder("float", (config.heatmap_batch,) + tuple(config.model_image_size))
        vgg = vgg16.Vgg16(vgg16_npy_path=config.vgg16_weight_path,fine_tune_layers=config.fine_tune_layers)
        validation_mode = tf.Variable(False, name='training')
        vgg.build(images,output_shape=config.output_shape,train_mode=validation_mode)
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

def get_heatmap_filename(config, model_name, method_name, variant_name, class_index, image_filename):
    # Derive filename to save heatmap in from model + image
    path = os.path.join(config.visualization_output, method_name, variant_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    return os.path.join(path, '%s_%s_%s.npy' % (model_name, str(class_index), os.path.basename(image_filename)))

def generate_heatmaps_for_images(config, ckpt, image_filenames, labels, model_name, method_name, variant, block_size=10, block_stride=1, generate_plots=False, use_true_label=False):
    # Generate all heatmaps for images in list
    if generate_plots:
        import matplotlib.pyplot as plt
    # Get class indices for all files
    if use_true_label:
        label_key = np.asarray(config.label_directories)
        class_indices = [np.where(config.which_dataset + '_' + fn == label_key) for fn in labels]
    else:
        class_indices = [None] * len(image_filenames)
    # Process all files
    variant_name = '%s_%d_%d' % (variant, block_size, block_stride)
    vgg,images = load_model_vgg16(config)
    saver = tf.train.Saver(tf.global_variables())
    with init_session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())) #need to initialize both if supplying num_epochs to inputs
        # saver.restore(sess, ckpt)
        for class_index, image_filename in zip(class_indices, image_filenames):
            heatmap_filename = get_heatmap_filename(config, model_name=model_name, method_name=method_name, variant_name=variant_name, class_index=class_index, image_filename=image_filename)
            print 'Heatmap for %s...' % os.path.basename(heatmap_filename)
            if os.path.isfile(heatmap_filename):
                print ' Skipping existing heatmap at %s' % heatmap_filename
            else:
                img = utils.load_image(image_filename)
                heatmap, base_prob = get_bubbles_heatmap(sess, vgg, images, img, class_index, variant=variant, block_size=block_size, block_stride=block_stride)
                split_file_name = re.split('vgg16',heatmap_filename)
                heatmap_filename = split_file_name[0] + str(base_prob) + '_vgg16' + split_file_name[1]
                print ' Saving heatmap to %s...' % heatmap_filename
                np.save(heatmap_filename, heatmap)
                if generate_plots:
                    f, axarr = plt.subplots(2, 1)
                    axarr[0].imshow(img, cmap='gray')
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

def random_sample_images(labels, in_dir, out_dir, im_ext, amount):
    image_filenames = np.asarray(glob.glob(os.path.join(in_dir, '*' + im_ext)))
    np.random.shuffle(image_filenames)
    if amount < 1:
        amount = np.round(amount * len(image_filenames))
    image_filenames = image_filenames[:amount]
    labels = match_filenames_labels(image_filenames,labels)
    new_filenames = [os.path.join(out_dir,labels[idx] + '_' + re.split('/',x)[-1]) for idx, x in enumerate(image_filenames)]
    [shutil.copy(image_filenames[x],new_filenames[x]) for x in range(len(image_filenames))]
    return new_filenames, labels

if __name__ == "__main__":
    config = GEDIconfig()

    #Make directories if they haven't been made yet
    dir_list = [config.heatmap_source_images,config.heatmap_dataset_images]
    [make_dir(d) for d in dir_list]
    image_filenames,image_labels = random_sample_images(config.heatmap_image_labels,config.validation_directory, config.heatmap_dataset_images, config.im_ext, config.heatmap_image_amount)

    #Run bubbles
    generate_heatmaps_for_images(config, config.bubbles_ckpt, image_filenames, image_labels, 'vgg16', 'bubbles', 'neg', block_size=config.block_size, block_stride=config.block_stride, generate_plots=config.generate_plots, use_true_label=config.use_true_label)
