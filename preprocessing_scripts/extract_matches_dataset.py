import os
import csv
import numpy as np
import pandas as pd
from gedi_config import GEDIconfig
from glob import glob
from glob2 import glob as glob2
from exp_ops.tf_fun import make_dir
from exp_ops.preprocessing_tfrecords import write_label_list, sample_files, \
    write_label_file, find_label, find_timepoint
from exp_ops.preprocessing_tfrecords_matching_tri import extract_to_tf_records
from tqdm import tqdm


def get_image_dict(config, splits=[.9, 1]):
    # gather file names of images to process
    im_lists = dict()
    if config.multi_dirs is None:
        for fl in config.tvt_flags:
            it_dir = '%s_%s' % (config.raw_im_dirs[0], fl)
            im_lists[fl] = glob(
                os.path.join(it_dir, '*%s' % config.raw_im_ext))
    else:
        dfiles = []
        for d in tqdm(config.raw_im_dirs, total=len(config.raw_im_dirs)):
            dfiles += [glob2(os.path.join(d, '**', '*%s' % config.raw_im_ext))]
        dfiles = np.asarray([item for sublist in dfiles for item in sublist])
        if len(config.tvt_flags) == 3 and len(splits) == 2:
            diff = splits[-1] - splits[0]
            splits[-1] = splits[0] + diff / 2
            splits += [splits[-1] + diff / 2]
        cvs = []
        for idx in range(len(splits)):
            cvs += [np.round(splits[idx] * len(dfiles)).astype(int)]
        start = 0
        for idx, fl in enumerate(config.tvt_flags):
            im_lists[fl] = dfiles[start:cvs[idx]]
            start = cvs[idx]
    return im_lists


def write_labels(flag, im_lists, config):
    # Write labels list
    make_dir(config.processed_image_patch_dir)
    label_list = os.path.join(
        config.processed_image_patch_dir, 'list_of_' + '_'.join(
            x for x in config.image_prefixes) + '_labels.txt')
    write_label_list(im_lists[flag], label_list)

    # Finally, write the labels file:
    labels_to_class_names = dict(
        zip(range(len(config.label_directories)), config.label_directories))
    write_label_file(labels_to_class_names, config.tfrecord_dir)
    return label_list


def read_csv(x):
    with open(x, 'rb') as csvfile:
        reader = [ix for ix in csv.reader(csvfile, delimiter='\n')]
    columns = reader[0][0].split(',')
    columns[0] = 'id'
    columns = [ix.strip('"') for ix in columns]
    data = []
    for r in range(1, len(reader)):
        data += [[c.strip('"') for c in reader[r][0].split(',')]]
    return pd.DataFrame(data, columns=columns)


def extract_tf_records_from_GEDI_tiffs():
    """Extracts data directly from GEDI tiffs and
    inserts them into tf records. This allows us to
    offload the normalization procedure to either
    right before training (via sampling) or during
    training (via normalization with a batch's max)"""

    # Grab the global config
    config = GEDIconfig()

    # If requested load in the ratio file
    ratio_file = os.path.join(
        config.home_dir,
        config.original_image_dir,
        config.ratio_stem,
        '%s%s.csv' % (
            config.ratio_prefix,
            config.experiment_image_set))
    if config.ratio_prefix is not None and os.path.exists(ratio_file):
        ratio_list = read_csv(
            os.path.join(ratio_file))
    else:
        ratio_list = None

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
    if config.encode_time_of_death is not None:
        death_timepoints = pd.read_csv(
            config.encode_time_of_death)[['plate_well_neuron', 'dead_tp']]
        keep_experiments = pd.read_csv(config.time_of_death_experiments)
        im_labels = {}
        for k, v in im_lists.iteritems():
            if k is not 'test':
                proc_ims, proc_labels = find_timepoint(
                    images=v,
                    data=death_timepoints,
                    keep_experiments=keep_experiments,
                    remove_thresh=config.mask_timepoint_value)
                im_labels[k] = proc_labels
                im_lists[k] = proc_ims
                df = pd.DataFrame(
                    np.vstack((proc_ims, proc_labels)).transpose(),
                    columns=['image', 'timepoint'])
                df.to_csv('%s.csv' % k)
            else:
                im_labels[k] = find_label(v)
    else:
        im_labels = {k: find_label(v) for k, v in im_lists.iteritems()}

    if type(config.tvt_flags) is str:
        tvt_flags = [config.tvt_flags]
    else:
        tvt_flags = config.tvt_flags
    assert len(np.concatenate(im_lists.values())), 'Could not find any files.'
    label_list = [write_labels(
        flag=x, im_lists=im_lists, config=config) for x in tvt_flags]

    if config.include_GEDI_in_tfrecords > 0:
        tf_flag = '_%sgedi' % config.include_GEDI_in_tfrecords
    else:
        tf_flag = ''
    if config.extra_image:
        tf_flag = '_1image'
    else:
        tf_flag = ''

    if type(config.tvt_flags) is str:
        files = im_lists[config.tvt_flags]
        label_list = im_labels[config.tvt_flags]
        output_pointer = os.path.join(
            config.tfrecord_dir, '%s%s' % (
                config.tvt_flags, tf_flag))
        extract_to_tf_records(
            files=files,
            label_list=label_list,
            output_pointer=output_pointer,
            ratio_list=ratio_list,
            config=config,
            k=config.tvt_flags)
    else:
        for k in config.tvt_flags:
            files = im_lists[k]
            label_list = im_labels[k]
            output_pointer = os.path.join(
                config.tfrecord_dir, '%s%s.tfrecords' % (
                    tf_flag, config.tf_record_names[k]))
            extract_to_tf_records(
                files=files,
                label_list=label_list,
                output_pointer=output_pointer,
                ratio_list=ratio_list,
                config=config,
                k=k)


if __name__ == '__main__':
    extract_tf_records_from_GEDI_tiffs()
