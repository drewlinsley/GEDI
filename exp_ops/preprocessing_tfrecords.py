import re
import os
import sys
import traceback
import shutil
import numpy as np
import tensorflow as tf
from scipy import misc
from glob import glob
from tqdm import tqdm
from exp_ops.preprocessing_GEDI_images import produce_patch
import pandas as pd


# # # For prepare tf records
def flatten_list(l):
    return [item for sublist in l for item in sublist]


def get_file_list(GEDI_path, label_directories, im_ext):
    files = []
    for idx in label_directories:
        print 'Getting files from: %s' % (os.path.join(
            GEDI_path, idx, '*' + im_ext))
        dir_files = glob(os.path.join(GEDI_path, idx, '*' + im_ext))
        files += dir_files
        print 'Found %s files' % len(dir_files)
    return files


def write_label_list(files, label_list):
    with open(label_list, "w") as f:
        f.writelines([ln + '\n' for ln in files])


def split_files(files, train_proportion, tvt_flags):
    num_files = len(files)
    all_labels = find_label(files)
    files = np.asarray(files)
    rand_order = np.random.permutation(num_files)
    split_int = int(np.round(num_files * train_proportion))
    train_inds = rand_order[:split_int]
    if type(tvt_flags) is str:
        new_files = {
            tvt_flags: files,
            tvt_flags + '_labels': all_labels,
        }
    else:
        new_files = {
            'train': files[train_inds],
            'train_labels': all_labels[train_inds],
        }
        if 'test' in tvt_flags and 'val' in tvt_flags:
            hint = int(np.round(num_files * (1-train_proportion)))
            val_inds = rand_order[split_int:split_int + hint]
            test_inds = rand_order[split_int + hint:]
            new_files['val'] = files[val_inds]
            new_files['val_labels'] = all_labels[val_inds]
            new_files['test'] = files[test_inds]
            new_files['test_labels'] = all_labels[test_inds]
        elif 'test' in tvt_flags and 'val' not in tvt_flags:
            val_inds = rand_order[split_int:]
            new_files['test'] = files[test_inds]
            new_files['test_labels'] = all_labels[test_inds]
        elif 'val' in tvt_flags and 'test' not in tvt_flags:
            val_inds = rand_order[split_int:]
            new_files['val'] = files[val_inds]
            new_files['val_labels'] = all_labels[val_inds]
    return new_files


def sample_files(files, train_proportion, tvt_flags):
    num_files = len(files)
    files = np.asarray(files)
    rand_order = np.random.permutation(num_files)
    split_int = int(np.round(num_files * train_proportion))
    val_files = files[rand_order[split_int:]]
    train_files = files[rand_order[:split_int]]
    return val_files, train_files


def move_files(files, target_dir):
    for idx in files:
        shutil.copyfile(idx, target_dir + re.split('/', idx)[-1])


def load_im_batch(files, hw, normalize):
    labels = []
    images = []
    if len(hw) == 2:
        rep_channel = True
    else:
        rep_channel = False
    for idx in files:
        # the parent directory is the label
        labels.append(re.split('/', idx)[-2])
        if rep_channel:
            images.append(np.repeat(misc.imread(idx)[:, :, None], 3, axis=-1))
        else:
            images.append(misc.imread(idx))
    if normalize is not None:
        images = [im.astype(np.float32)/255 for im in images]
    # transpose images to batch,ch,h,w
    return np.asarray(images).transpose(0, 3, 1, 2), np.asarray(labels)


def find_label(files):
    _, c = np.unique(
        [re.split('/', l)[-2] for l in files], return_inverse=True)
    return c


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def floats_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_converter(im_ext):
    if im_ext == '.jpg' or im_ext == '.jpeg' or im_ext == '.JPEG':
        out_fun = tf.image.encode_jpeg
    elif im_ext == '.png':
        out_fun = tf.image.encode_png
    else:
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60
    return out_fun


def get_image_ratio(
        f,
        ratio_list,
        timepoints,
        id_column,
        regex_match
        ):
    '''Loop through the ratio list until you find the
    id_column row that matches f'''
    if f is None or ratio_list is None:
        return None
    else:
        f_re = re.search(regex_match, f).group()
        if id_column not in ratio_list.columns:
            raise RuntimeError('Cannot find your id_column in the dataframe.')
        r = ratio_list[ratio_list[id_column].str.contains(f_re)]
        if r.empty:
            return None
        else:
            return [r[str(x)].as_matrix() for x in timepoints]


def features_to_dict(
        label,
        image,
        filename,
        ratio,
        ratio_placeholder=-1.):
    if ratio is None:
        ratio = ratio_placeholder
    if isinstance(label, int):
        label = int64_feature(label)
    else:
        label = floats_feature(label)

    return {  # Go ahead and store a None ratio if necessary
        'label': label,
        'image': bytes_feature(image.tostring()),
        'filename': bytes_feature(filename),
        'ratio': floats_feature(ratio)
    }


def extract_to_tf_records(
        files,
        label_list,
        ratio_list,
        output_pointer,
        config,
        k):
    print 'Building %s: %s' % (k, config.tfrecord_dir)
    max_array = np.zeros(len(files))
    min_array = np.zeros(len(files))
    nan_images = np.zeros(len(files))
    with tf.python_io.TFRecordWriter(output_pointer) as tfrecord_writer:
        for idx, (f, l) in tqdm(
            enumerate(
                zip(files, label_list)), total=len(files)):
            r = get_image_ratio(
                f,
                ratio_list,
                timepoints=config.channel,
                id_column=config.id_column,
                regex_match=config.ratio_regex)
            if isinstance(config.channel, list):
                image = []
                for c in config.channel:
                    image += [produce_patch(
                        f,
                        c,
                        config.panel,
                        divide_panel=config.divide_panel,
                        max_value=config.max_gedi,
                        min_value=config.min_gedi).astype(
                            np.float32)[None, :, :]]
                image = np.concatenate(image)
                l = (r > config.ratio_cutoff).astype(int)
            else:
                image = produce_patch(
                    f,
                    config.channel,
                    config.panel,
                    divide_panel=config.divide_panel,
                    max_value=config.max_gedi,
                    min_value=config.min_gedi).astype(np.float32)
            if np.isnan(image).sum() != 0:
                nan_images[idx] = 1
            max_array[idx] = np.max(image)
            # construct the Example proto boject
            feature_dict = features_to_dict(
                label=l,
                image=image,
                filename=f,
                ratio=r)
            example = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                    # Features has a map of string to Feature proto objects
                    feature=feature_dict
                )
            )
            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            tfrecord_writer.write(serialized)

    # Calculate ratio of +:-
    lab_counts = np.asarray(
        [np.sum(label_list == 0), np.sum(label_list == 1)]).astype(float)
    ratio = lab_counts / np.asarray((len(label_list))).astype(float)
    print 'Data ratio is %s' % ratio
    np.savez(
        os.path.join(
            config.tfrecord_dir, k + '_' + config.max_file),
        max_array=max_array,
        min_array=min_array,
        ratio=ratio,
        filenames=files)
    return max_array, min_array


def write_label_file(labels_to_class_names, dataset_dir,
                     filename='labels.txt'):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def find_timepoint(
        images,
        data,
        keep_experiments=None, 
        label_column='plate_well_neuron',
        remove_prefix='bs_',
        remove_thresh=-900):  # exclusion is some large value
    pre_len = len(images)
    images = [im for im in images if remove_prefix not in im]
    if keep_experiments is not None:
        filter_list = keep_experiments['plate+AF8-well+AF8-neuron'].as_matrix()
        images = [im for im in images if im.split('/')[-1].split('_')[1] in filter_list]
    post_len = len(images)
    print 'Removed %s images (%s remaining).' % ((
        pre_len - post_len), post_len)
    data_labels = data[label_column]
    data_labels = data_labels.str.replace('^(.*?)_', '')
    im_timepoints = np.ones((len(images)), dtype=int) * -1
    for imidx, im in tqdm(enumerate(images), total=len(images)):
        im_name = im.split('/')[-1]
        im_name = im_name.split('_')
        exp_name = im_name[1]
        well_name = im_name[2]
        cell_number = im_name[3]
        len_ref = data_labels.str.len()
        probe_name = '%s_%s_%s' % (exp_name, well_name, cell_number)
        cross_ref = data_labels.str.match(probe_name)
        mask = (len_ref == len(probe_name)) & cross_ref
        if mask.sum() == 1:
            it_data = data.loc[mask]['dead_tp'].as_matrix()[0]  # .astype(int)
            if it_data > remove_thresh:
                im_timepoints[imidx] = it_data
        elif mask.sum() > 1:
            print 'Found multiple entries??'
    # Remove images and timepoints where we have a -1 (i.e. no timecourse data)
    keep_idx = im_timepoints != -1
    np_images = np.asarray(images)
    images = list(np_images[keep_idx])
    im_timepoints = im_timepoints[keep_idx]
    return images, im_timepoints

