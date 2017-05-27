import re
import os
import sys
sys.path.insert(0, re.split(__file__, os.path.realpath(__file__))[0])  # puts this experiment into path
from gedi_config import GEDIconfig
from glob import glob
from exp_ops.tf_fun import make_dir
from exp_ops.preprocessing_GEDI_images import produce_patches_parallel, produce_patches

# Grab the global config
config = GEDIconfig()
USE_PARALLEL = False

# Process images
[make_dir(os.path.join(config.home_dir, d)) for d in config.output_dirs]  # Make directories for processed images
im_lists = [glob(os.path.join(config.home_dir, r, '*' + config.raw_im_ext))  # gather file names of images to process
    for r in config.raw_im_dirs]

if USE_PARALLEL:
    produce_patches_parallel(config, im_lists)  # multithread processing
else:
    [produce_patches(p, config.channel, config.panel,
        config.output_dirs[i], config.raw_im_ext,
        config.im_ext, max_value=config.max_gedi, divide_panel=config.divide_panel)
        for i, p in enumerate(im_lists)]  # single thread processing