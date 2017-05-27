import re, os, sys
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
from scipy import misc, stats
sys.path.append('../')
import gedi_config

def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that are furthest away
    from 0 along axis"""
    maxa = a.max(axis=axis)
    mina = a.min(axis=axis)
    p = abs(maxa) > abs(mina) # bool, or indices where +ve values win
    n = abs(mina) > abs(maxa) # bool, or indices where -ve values win
    if axis == None:
        if p: return maxa
        else: return mina
    shape = list(a.shape)
    shape.pop(axis)
    out = np.zeros(shape, dtype=a.dtype)
    out[p] = maxa[p]
    out[n] = mina[n]
    return out

def normalize_raw(z):
    mu_z = np.mean(z,axis=2)
    maxval = np.max(np.abs(z)) #z[:,:,:2]
    out_z = mu_z / maxval
    return np.repeat(out_z[:,:,None],3,axis=2) #convert back to a 3-channel image

def zscore_whole(z):
    mu = np.mean(z)
    sd = np.std(z)
    for layer in range(z.shape[-1]):
        z[:,:,layer] -= mu
        z[:,:,layer] /= sd
    return z

def zscore_channels(z):
    for layer in range(z.shape[-1]):
        z[:,:,layer] -= np.mean(z[:,:,layer])
        z[:,:,layer] /= (np.std(z[:,:,layer]) + 1e-3)
    return z

def normalize_channels(z):
    for layer in range(z.shape[-1]):
        z[:,:,layer] -= np.mean(z[:,:,layer])
        z[:,:,layer] /= (np.max(np.abs(z[:,:,layer])))
    return z

def load_images(files):
    dum_shape = np.load(files[0]).shape[:2]
    images = np.zeros((len(files),dum_shape[0],dum_shape[1]))
    for idx,f in enumerate(files):
        images[idx,:,:] = maxabs(zscore_channels(np.load(f)),axis=-1)[None,:,:]
    return images

def make_mosaic(images,remove_images=None):
    if remove_images != None:
        rem_idx = np.ones(len(images))
        rem_idx[remove_images] = 0
        images = images[rem_idx == 1,:,:]

    im_dim = images.shape
    num_cols = np.sqrt(im_dim[0]).astype(int)
    num_cols = np.min([num_cols,5])
    num_rows = num_cols# + (im_dim[0] - num_cols)
    canvas = np.zeros((im_dim[1] * num_rows, im_dim[2] * num_cols))
    count = 0
    row_anchor = 0
    col_anchor = 0
    for x in range(num_rows):
        for y in range(num_cols):
            it_image = images[count] * -1
            it_image += (np.sign(np.min(it_image)) * np.min(it_image))
            it_image /= (np.max(it_image) + 1e-3)
            canvas[row_anchor:row_anchor + im_dim[1],col_anchor:col_anchor + im_dim[2]] += np.sqrt(it_image)
            col_anchor += im_dim[2]
            count += 1
        col_anchor = 0
        row_anchor += im_dim[1]
    return canvas

def plot_filters(layer_weights, title=None, show=False):
    mosaic = make_mosaic(layer_weights)
    plt.imshow(mosaic, interpolation='nearest')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def plot_mosaic(images,title,output_file,save=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24,24))
    im = ax.imshow(images, cmap=plt.cm.Reds, vmin=0, vmax=1)
    plt.title(title)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    if save:
        plt.savefig(output_file)

def print_overlay(gfp,gedi_im,heatmap,output,alpha=0.5):
    cmap = pl.cm.Reds
    extent = (0, gfp.shape[0], 0, gfp.shape[1])
    fig = plt.figure(figsize=(24,24), frameon=False)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(gfp, extent=extent, cmap='gray', vmin=0, vmax=1)
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    ax1.title.set_text('GFP')
    ax2.imshow(gedi_im, extent=extent, cmap='gray', vmin=0, vmax=1)
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])
    ax2.title.set_text('GEDI')
    im = ax3.imshow(heatmap, extent=extent, cmap=cmap, vmin=0, vmax=1)
    ax3.set_xticks([], [])
    ax3.set_yticks([], [])
    ax3.title.set_text('CNN explanation')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.4, 0.02, 0.2])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(output)
    plt.close('all')

def overlay_with_gedi(config,files,images,out_name,remove_images=None):
    if remove_images != None:
        rem_idx = np.ones(len(images))
        rem_idx[remove_images] = 0
        images = images[rem_idx == 1,:,:]
        files = files[rem_idx == 1]

    corrs = np.zeros((len(files),3))
    for idx,fi in enumerate(files):
        im_name = 'cache_' + re.split('.npy',re.split('cache_',fi)[-1])[0]
        it_im = images[idx] * -1
        it_im += (np.sign(np.min(it_im)) * np.min(it_im))
        it_im = np.sqrt(it_im)
        it_im /= (np.max(it_im) + 1e-3)

        gfp_fp = config.GEDI_path + 'validation/gfp/' + im_name
        gfp_im = misc.imread(gfp_fp)
        try: 
            gedi_fp = config.GEDI_path + 'gedi_Live/' + im_name
            gedi_im = misc.imread(gedi_fp)
        except:
            gedi_fp = config.GEDI_path + 'gedi_Dead/' + im_name
            gedi_im = misc.imread(gedi_fp)
        gfp_im = (misc.imresize(gfp_im,it_im.shape).astype(np.float32))/255
        gedi_im = (misc.imresize(gedi_im,it_im.shape).astype(np.float32))/255
        print_overlay(gfp_im,gedi_im,it_im,'../overlays/' + out_name + '/' + im_name)
        mv = np.min(it_im)
        it_im[:10,:] = mv
        it_im[:,:10] = mv
        it_im[-10:,:] = mv
        it_im[:,-10:] = mv 
        corrs[idx,0] = stats.spearmanr(gfp_im.ravel(),it_im.ravel()).correlation
        corrs[idx,1] = stats.spearmanr(gedi_im.ravel(),it_im.ravel()).correlation
        corrs[idx,2] = stats.spearmanr(gfp_im.ravel(),gedi_im.ravel()).correlation
    return corrs

def main():
    dead_dir = '/media/data/GEDI/drew_images/patches/visualizations/gfp/lrp/neg_5_1/+_+/'
    live_dir = '/media/data/GEDI/drew_images/patches/visualizations/gfp/lrp/neg_5_1/-_-/'
    fp_dir = '/media/data/GEDI/drew_images/patches/visualizations/gfp/lrp/neg_5_1/-_+/'
    fn_dir = '/media/data/GEDI/drew_images/patches/visualizations/gfp/lrp/neg_5_1/+_-/'
    dead_files = glob(dead_dir + '*.npy')
    live_files = glob(live_dir + '*.npy')
    fp_files = glob(fp_dir + '*.npy')
    fn_files = glob(fn_dir + '*.npy')
    dead_images = load_images(dead_files)
    live_images = load_images(live_files)
    fp_images = load_images(fp_files)
    fn_images = load_images(fn_files)

    dead_mosaic = make_mosaic(dead_images,remove_images=[8,21])
    live_mosaic = make_mosaic(live_images,remove_images=None)
    fp_mosaic = make_mosaic(fp_images)
    fn_mosaic = make_mosaic(fn_images)

    plot_mosaic(dead_mosaic,'Correctly classified dead neurons','../dead_mosaic.png',True)
    plot_mosaic(live_mosaic,'Correctly classified live neurons','../live_mosaic.png',True)
    plot_mosaic(fp_mosaic,'Dead neurons classified as live','../fp_mosaic.png',True)
    plot_mosaic(fn_mosaic,'Live neurons classified as dead','../fn_mosaic.png',True)

    config = gedi_config.GEDIconfig()
    dead_corrs = overlay_with_gedi(config,np.asarray(dead_files),dead_images,'dead',remove_images=None)
    live_corrs = overlay_with_gedi(config,np.asarray(live_files),live_images,'live',remove_images=None)




