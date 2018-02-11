import re
import os
import sys
import traceback
import numpy as np
from scipy import misc
from tifffile import TiffFile
from multiprocessing import Pool
from tqdm import tqdm


# # # for extract frames from GEDI
def produce_patches_parallel(config, im_lists, number_of_threads=8):
    pool = Pool(number_of_threads)
    try:
        for od, il in zip(config.output_dirs, im_lists):
            engine = Engine(
                channel=config.channel, panel=config.panel,
                output_dir=os.path.join(config.home_dir, od),
                raw_im_ext=config.raw_im_ext,
                out_im_ext=config.im_ext,
                max_gedi=config.max_gedi,
                divide_panel=config.divide_panel)
            print('Extracting patches to %s' % od)
            pool.map(engine, il, chunksize=128)
    except IOError:
        print '-' * 60
        traceback.print_exc(file=sys.stdout)
        print '-' * 60
    finally:
        pool.close()
        pool.join()


def produce_patches(
        paths, channel, panel, output_dir, raw_im_ext,
        out_im_ext, max_value, divide_panel=None):

    """For single thread, serial"""
    for p in tqdm(paths, total=len(paths)):
        im_name = re.split(raw_im_ext, re.split('/', p)[-1])[0]
        with TiffFile(p) as tif:
            im = tif.asarray()[channel]
        patch = get_patch(im, panel).astype(np.float32) / max_value
        if divide_panel is not None:
            patch = patch.astype(
                np.float32) / (
                get_patch(im, divide_panel).astype(np.float32) + 0.01)
        misc.imsave(
            os.path.join(
                output_dir, im_name + out_im_ext), np.repeat(
                patch[:, :, None], 3, axis=-1))


def get_patch(im, panel):
    if panel == 0:
        im = im[:, :300]  # GFP raw
    elif panel == 1:
        im = im[:, 300:600]  # GFP masked
    elif panel == 2:
        im = im[:, 600:]  # GEDI signal raw
    return im


def rescale_patch(patch, min_value, max_value):
    return (patch - min_value) / (max_value - min_value)


def produce_patch(
        p,
        channel,
        panel,
        max_value=None,
        min_value=None,
        divide_panel=None,
        raw_im_ext=None,
        output_dir=None,
        out_im_ext=None,
        return_raw=False,
        matching=False,
        debug=False):
    """For multithread"""
    with TiffFile(p) as tif:
        im = tif.asarray()
        if return_raw:
            return im[:, :, :300]
        if matching:
            return im
        if len(im.shape) > 2:
            im = im[channel]
        else:
            im = im.squeeze()
            if debug:
                print(
                    'Warning: Image has no slices. '
                    'Image size is %s.' % str(im.shape))
    patch = get_patch(im, panel).astype(np.float32)
    if max_value is None:
        max_value = np.max(patch)
    if min_value is None:
        min_value = np.min(patch)
    # patch /= max_value
    patch = rescale_patch(
        patch,
        min_value=min_value,
        max_value=max_value)

    if divide_panel is not None:
        patch /= ((
            get_patch(im, divide_panel).astype(np.float32) + 0.01) / max_value)

    if raw_im_ext is not None:
        im_name = re.split(raw_im_ext, re.split('/', p)[-1])[0]
        misc.imsave(os.path.join(output_dir, im_name + out_im_ext), patch)
    else:
        return patch


def derive_timepoints(p):
    """Figure out the number of timepoints in the slide."""
    with TiffFile(p) as tif:
        im = tif.asarray()
        tp = im.shape[0]
    return tp


class Engine(object):
    def __init__(
        self, channel, panel, output_dir, raw_im_ext,
            out_im_ext, max_gedi, divide_panel):
        self.channel = channel
        self.panel = panel
        self.output_dir = output_dir
        self.raw_im_ext = raw_im_ext
        self.out_im_ext = out_im_ext
        self.max_gedi = max_gedi
        self.divide_panel = divide_panel

    def __call__(self, filename):
        produce_patch(
            filename,
            channel=self.channel, panel=self.panel, output_dir=self.output_dir,
            raw_im_ext=self.raw_im_ext, out_im_ext=self.out_im_ext,
            max_value=self.max_gedi, divide_panel=self.divide_panel)
