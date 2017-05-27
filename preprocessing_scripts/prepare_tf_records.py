import sys
import re
import os
from scipy import misc
sys.path.insert(0, re.split(__file__, os.path.realpath(__file__))[0])  # puts this experiment into path
from gedi_config import GEDIconfig
from exp_ops.tf_fun import make_dir
from exp_ops.preprocessing_tfrecords import get_file_list, write_label_list, split_files,\
     move_files, simple_tf_records, write_label_file

# Converts processed data into tfrecords files
def run():
    # Run build_image_data script
    print('Organizing files into tf records')

    # Make dirs if they do not exist
    config = GEDIconfig()
    dir_list = [config.train_directory, config.validation_directory,
        config.tfrecord_dir, config.train_checkpoint]
    [make_dir(d) for d in dir_list]

    # Prepare lists with file pointers
    files = get_file_list(config.processed_image_patch_dir, config.label_directories, config.im_ext)
    label_list = os.path.join(config.processed_image_patch_dir,
        'list_of_' + '_'.join(x for x in config.image_prefixes) + '_labels.txt')  # to be created by prepare
    write_label_list(files, label_list)

    # Copy data into the appropriate training/testing directories
    hw = misc.imread(files[0]).shape
    new_files = split_files(files, config.train_proportion, config.tvt_flags)
    move_files(new_files['train'], config.train_directory)
    #process_image_data('train',new_files,config.tfrecord_dir,config.im_ext,config.train_shards,hw,config.normalize)
    simple_tf_records('train', new_files, config.tfrecord_dir, config.im_ext,
        config.train_shards, hw, config.normalize)
    if 'val' in config.tvt_flags:
        move_files(new_files['val'], config.validation_directory)
        #process_image_data('val',new_files,config.tfrecord_dir,config.im_ext,config.train_shards,hw,config.normalize)
        simple_tf_records('val', new_files, config.tfrecord_dir, config.im_ext,
            config.train_shards, hw, config.normalize)
    if 'test' in config.tvt_flags:
        move_files(new_files['test'],config.test_directory)
        #process_image_data('test',new_files,config.tfrecord_dir,config.im_ext,config.train_shards,hw,config.normalize)
        simple_tf_records('test', new_files, config.tfrecord_dir, config.im_ext,
            config.train_shards, hw, config.normalize)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(config.label_directories)), 
        config.label_directories))
    write_label_file(labels_to_class_names, config.tfrecord_dir)

if __name__ == '__main__':
    run()
