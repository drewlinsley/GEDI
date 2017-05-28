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
from exp_ops.mosaic_visualizations import maxabs, zscore_channels
from exp_ops import lrp
from model_depo import vgg16_trainable_lrp as vgg16
from ops import utils
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import OneHotEncoder as oe

def init_session():
    return tf.Session(config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.95))))

def load_model_vgg16(config):
    #Prepare model on GPU
    with tf.device('/gpu:0'):
        images = tf.placeholder("float", (1,) + tuple(config.model_image_size),'images')
        vgg = vgg16.Vgg16(vgg16_npy_path=config.vgg16_weight_path,fine_tune_layers=config.fine_tune_layers)
        validation_mode = tf.Variable(False, name='training')
        vgg.build(images,output_shape=config.output_shape,train_mode=validation_mode)
        y = tf.placeholder(tf.float32, (None,len(config.raw_im_dirs)), name='y')
    return vgg, images, y

def get_heatmap_filename(config, model_name, method_name, variant_name, class_index, image_filename):
    # Derive filename to save heatmap in from model + image
    path = os.path.join(config.visualization_output, method_name, variant_name)
    dir_list = [path,path + '/+_+/',path,path + '/+_-/',path,path + '/-_+/',path,path + '/-_-/']
    [make_dir(d) for d in dir_list]

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
    vgg,images,y = load_model_vgg16(config)
    saver = tf.train.Saver(tf.global_variables())
    with init_session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())) #need to initialize both if supplying num_epochs to inputs
        saver.restore(sess, ckpt)
        for class_index, image_filename, lab in zip(class_indices, image_filenames, labels):
            heatmap_filename = get_heatmap_filename(config, model_name=model_name, method_name=method_name, variant_name=variant_name, class_index=class_index, image_filename=image_filename)
            print 'Heatmap for %s...' % os.path.basename(heatmap_filename)
            if os.path.isfile(heatmap_filename):
                print ' Skipping existing heatmap at %s' % heatmap_filename
            else:
                img = np.repeat(utils.load_image(image_filename)[:,:,None],3,axis=-1)[None,:,:,:]
                logits,base_prob = sess.run([vgg.fc8,vgg.prob],feed_dict={images:img})
                F = lrp.lrp(logits*y,-123.68, 255 - 123.68)
                heatmap = lrp.get_lrp_im(sess, F, images, y, img, lab)[0]
                norm_heatmap = maxabs(zscore_channels(heatmap),axis=-1)
                split_file_name = re.split('vgg16',heatmap_filename)
                conf_dir = get_confusion(base_prob,lab)
                heatmap_filename = split_file_name[0] + conf_dir + str(base_prob) + '_vgg16' + split_file_name[1]
                print ' Saving heatmap to %s...' % heatmap_filename
                np.save(heatmap_filename, heatmap)
                if generate_plots:
                    f, axarr = plt.subplots(2, 1)
                    axarr[0].imshow(img[0], cmap='Greys')
                    plt.grid(False)
                    m = axarr[1].imshow(norm_heatmap, cmap='Reds')
                    plt.grid(False)
                    f.colorbar(m)
                    plt.savefig(heatmap_filename + '.png')
                    plt.close()

def get_confusion(base_prob,lab):
    yhat = np.argmax(base_prob)
    lab_idx = np.where(np.asarray(lab.astype(int).tolist()[0]) == 1)[0][0] #you have got to be kidding me
    if yhat == 0 and lab_idx == 0:
        out_dir = '/+_+/'

    elif yhat == 0 and lab_idx == 1:
        out_dir = '/+_-/'

    elif yhat == 1 and lab_idx == 0:
        out_dir = '/-_+/'

    elif yhat == 1 and lab_idx == 1:
        out_dir = '/-_-/'
    return out_dir

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
    enc = oe()
    oh_labels = enc.fit_transform(np.asarray([(x == 'Live') for x in image_labels]).astype(np.float32).reshape(-1,1)).todense()

    #Run lrp
    generate_heatmaps_for_images(config, config.bubbles_ckpt, image_filenames, oh_labels, 'vgg16', 'lrp', 'neg', block_size=5, block_stride=1, generate_plots=config.generate_plots, use_true_label=config.use_true_label)
