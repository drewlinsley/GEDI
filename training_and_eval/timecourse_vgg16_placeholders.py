#!/usr/bin/env python
import os, sys, re, shutil
import numpy as np
import tensorflow as tf
from glob import glob
sys.path.append('../../') #puts model_depo on the path
sys.path.insert(0,re.split(__file__,os.path.realpath(__file__))[0]) #puts this experiment into path
from scipy.misc import imresize, imsave
from gedi_config import GEDIconfig
from exp_ops.helper_functions import make_dir
from exp_ops import lrp
from model_depo import vgg16_trainable_batchnorm as vgg16
from ops import utils
from tifffile import TiffFile
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import OneHotEncoder as oe
from tqdm import tqdm

def init_session():
    return tf.Session(config=tf.ConfigProto(allow_soft_placement = True, gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.95))))

def load_model_vgg16(config):
    #Prepare model on GPU
    with tf.device('/gpu:0'):
        images = tf.placeholder("float", (1,) + tuple(config.model_image_size),'images')
        vgg = vgg16.Vgg16(vgg16_npy_path=config.vgg16_weight_path,fine_tune_layers=config.fine_tune_layers)
        validation_mode = tf.Variable(False, name='training')
        vgg.build(images,output_shape=config.output_shape,train_mode=validation_mode)
        y = tf.placeholder(tf.float32, (1,len(config.raw_im_dirs)), name='y')
    return vgg, images, y

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

def produce_patches(home_dir,p,label,channel,panel,in_im_ext,out_im_ext,max_value,divide_panel=None):
    im_name = re.split(in_im_ext,re.split('/',p)[-1])[0] + out_im_ext
    tn = home_dir + 'original_images/' + label + '/' + im_name
    with TiffFile(tn) as tif:
        im = tif.asarray()[channel]
    patch = get_patch(im,panel).astype(np.float32) / max_value
    if divide_panel != None:
        patch = patch.astype(np.float32) / (get_patch(im,divide_panel).astype(np.float32) + 0.01)
    return np.repeat(center_crop(patch,[224,224])[:,:,None],3,axis=-1), label + '/' + im_name

def center_crop(image,size):
    x_off = (image.shape[0] - size[0]) // 2
    y_off = (image.shape[1] - size[1]) // 2
    return image[x_off:-x_off,y_off:-y_off]    

def get_patch(im,panel):
        if panel == 0:
                im = im[:,:300]
        elif panel == 1:
                im = im[:,300:600]
        elif panel == 2:
                im = im[:,600:]
        return im

def process_timeseries(config,paths,labels,oh_labels,ckpt,output):
    vgg,images,y = load_model_vgg16(config)
    saver = tf.train.Saver(tf.global_variables())
    num_timepoints = 3
    with init_session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())) #need to initialize both if supplying num_epochs to inputs
        saver.restore(sess, ckpt)
        decisions = np.zeros((len(labels),num_timepoints))
        tiff_names = []
        out = open(output,'w')
        for idx in tqdm(range(len(labels))):
            #Accumulate timecourse from tifs
            _,it_label = produce_patches(config.home_dir,paths[idx],labels[idx],0,config.panel,config.im_ext,config.raw_im_ext,config.max_gedi,divide_panel=config.divide_panel)
            out.write('%s,' % it_label)
            for channel in range(num_timepoints):
                #tc = np.zeros((images.get_shape()))
                tc,it_label = produce_patches(config.home_dir,paths[idx],labels[idx],channel,config.panel,config.im_ext,config.raw_im_ext,config.max_gedi,divide_panel=config.divide_panel) #really should be doing this through tfrecords instead of placeholders... 
                probs = sess.run([vgg.prob],feed_dict={images:tc[None,:,:,:]})[0]
                decisions[idx,channel] = np.argmax(probs)
                out.write('%d,' % decisions[idx,channel])
            out.write('\r')
            tiff_names.append(it_label)
    import ipdb;ipdb.set_trace()
    
    #Finish csv and save a numpy 
    out.close()
    np.savez('tcs',decisions=decisions,paths=paths)

if __name__ == "__main__":
    #0. Get ground truth for each validation image

    #1. find tifs for validation images in /media/data/GEDI/drew_images/original_images/
    #config.home_dir + config.raw_im_dirs[0]
    #config.home_dir + config.raw_im_dirs[0]

    #2. For each tif, extract timepoints 0-5, put in a batch

    #3. Pass placeholder to cnn

    #4. record argmax and export to csv
    config = GEDIconfig()
    validation_images = glob(config.validation_directory + '*.png')
    validation_labels = match_filenames_labels(validation_images,config.heatmap_image_labels)
    enc = oe()
    oh_labels = enc.fit_transform(np.asarray([(x == 'Live') for x in validation_labels]).astype(np.float32).reshape(-1,1)).todense()
    process_timeseries(config,validation_images,validation_labels,oh_labels,config.bubbles_ckpt,'survivals.csv')    



