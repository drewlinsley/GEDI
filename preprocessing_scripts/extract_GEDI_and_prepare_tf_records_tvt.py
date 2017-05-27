import os
import re
import sys
sys.path.insert(0, re.split(__file__, os.path.realpath(__file__))[0])
from gedi_config import GEDIconfig
from glob import glob
from exp_ops.tf_fun import make_dir
from exp_ops.preprocessing_tfrecords import write_label_list, sample_files, \
    extract_to_tf_records, write_label_file, flatten_list, find_label


def get_image_dict(config):
    # gather file names of images to process
    im_lists = dict()
    for fl in config.tvt_flags:
        it_files = [x for x in config.raw_im_dirs if fl in x]
        im_lists[fl] = flatten_list(
            [glob(os.path.join(config.home_dir, r, '*' + config.raw_im_ext))
                for r in it_files])
    return im_lists


def write_labels(flag, im_lists, config):
    # Write labels list
    label_list = os.path.join(
        config.processed_image_patch_dir, 'list_of_' + '_'.join(
            x for x in config.image_prefixes) + '_labels.txt')
    write_label_list(im_lists[flag], label_list)

    # Finally, write the labels file:
    labels_to_class_names = dict(
        zip(range(len(config.label_directories)), config.label_directories))
    write_label_file(labels_to_class_names, config.tfrecord_dir)
    return label_list


def extract_tf_records_from_GEDI_tiffs():
    """Extracts data directly from GEDI tiffs and
    inserts them into tf records. This allows us to
    offload the normalization procedure to either
    right before training (via sampling) or during
    training (via normalization with a batch's max)"""

    # Grab the global config
    config = GEDIconfig()

    # Make dirs if they do not exist
    dir_list = [
        config.train_directory, config.validation_directory,
        config.tfrecord_dir, config.train_checkpoint]
    if 'test' in config.tvt_flags:
        dir_list += [config.test_directory]
        config.raw_im_dirs = [x + '_train' for x in config.raw_im_dirs]
        config.raw_im_dirs += [x.split(
            '_train')[0] + '_test' for x in config.raw_im_dirs]
    [make_dir(d) for d in dir_list]
    im_lists = get_image_dict(config)

    # Sample from training for validation images
    if 'val' in config.tvt_flags:
        im_lists['val'], im_lists['train'] = sample_files(
            im_lists['train'], config.train_proportion, config.tvt_flags)
    im_labels = {k: find_label(v) for k, v in im_lists.iteritems()}

    if type(config.tvt_flags) is str:
        tvt_flags = [config.tvt_flags]
    else:
        tvt_flags = config.tvt_flags
    label_list = [write_labels(
        flag=x, im_lists=im_lists, config=config) for x in tvt_flags]

    if type(config.tvt_flags) is str:
            files = im_lists[config.tvt_flags]
            label_list = im_labels[config.tvt_flags]
            output_pointer = os.path.join(
                config.tfrecord_dir, config.tvt_flags + '.tfrecords')
            extract_to_tf_records(
                files, label_list, output_pointer, config,
                config.tvt_flags)
    else:
        for k in config.tvt_flags:
            files = im_lists[k]
            label_list = im_labels[k]
            output_pointer = os.path.join(
                config.tfrecord_dir, k + '.tfrecords')
            extract_to_tf_records(files, label_list, output_pointer, config, k)


if __name__ == '__main__':
    extract_tf_records_from_GEDI_tiffs()
