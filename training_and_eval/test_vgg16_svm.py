import os
import sys
import time
import re
import itertools
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from scipy import misc
from glob import glob
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from argparse import ArgumentParser
from exp_ops.tf_fun import make_dir, find_ckpts
from gedi_config import GEDIconfig
from models import vgg16_trainable_batchnorm_shared as vgg16
from exp_ops.preprocessing_GEDI_images import produce_patch


image_dir = '/media/data/GEDI/drew_images/original_images/human_bs'
tf_dir = '/media/data/GEDI/drew_images/project_files/tfrecords/human_bs_gfp'


# Evaluate your trained model on GEDI images
def test_vgg16(validation_data, model_dir, label_file, selected_ckpts=-1, force=False):
    config = GEDIconfig()

    # Load metas
    meta_data = np.load(os.path.join(tf_dir, 'val_maximum_value.npz'))
    max_value = np.max(meta_data['max_array']).astype(np.float32)

    # Find model checkpoints
    ckpts, ckpt_names = find_ckpts(config, model_dir)
    # ds_dt_stamp = re.split('/', ckpts[0])[-2]
    out_dir = os.path.join(config.results, 'gfp_2017_02_19_17_41_19' + '/')
    try:
        config = np.load(os.path.join(out_dir, 'meta_info.npy')).item()
        # Make sure this is always at 1
        config.validation_batch = 64
        print '-'*60
        print 'Loading config meta data for:%s' % out_dir
        print '-'*60
    except:
        print '-'*60
        print 'Using config from gedi_config.py for model:%s' % out_dir
        print '-'*60

    sorted_index = np.argsort(np.asarray([int(x) for x in ckpt_names]))
    ckpts = ckpts[sorted_index]
    ckpt_names = ckpt_names[sorted_index]

    # CSV file
    svm_image_file = os.path.join(out_dir, 'svm_models.npz')
    if os.path.exists(svm_image_file) and force == False:
        svm_image_data = np.load(svm_image_file)
        image_array = svm_image_data['image_array']
        label_vec = svm_image_data['label_vec']
        tr_label_vec = svm_image_data['tr_label_vec']
        np_label_vec = svm_image_data['np_label_vec']
        missing_ims = svm_image_data['missing_ims']
    else:
        labels = pd.read_csv(
            os.path.join(
                config.processed_image_patch_dir,
                label_file))
        image_array = []
        label_vec = []
        missing_ims = []
        # Because looking up images form the csv doesn't work, let's find the label for each image
        for _, row in labels.iterrows():
            try:
                im = produce_patch(
                    os.path.join(image_dir, row['lf']),
                    config.channel,
                    config.panel,
                    divide_panel=config.divide_panel,
                    max_value=None).astype(
                    np.float32)
                im = np.repeat(
                    misc.imresize(
                        im / max_value,
                        config.model_image_size)[:, :, None], 3, axis=-1)
                image_array.append(im)
                label_vec.append(row['Sci_SampleID'])
            except:
                print 'Cannot find %s' % row['lf']
                missing_ims.append(row['lf'])
        np_label_vec = np.asarray(label_vec)
        le = preprocessing.LabelEncoder()
        tr_label_vec = le.fit_transform(np_label_vec)
        np.savez(
            svm_image_file,
            image_array=image_array,
            label_vec=label_vec,
            tr_label_vec=tr_label_vec,
            np_label_vec=np_label_vec,
            missing_ims=missing_ims)

    # Make output directories if they do not exist
    dir_list = [config.results, out_dir]
    [make_dir(d) for d in dir_list]

    # Make placeholder
    val_images = tf.placeholder(
        tf.float32, shape=[None] + config.model_image_size)

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            vgg = vgg16.Vgg16(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.fine_tune_layers)
            validation_mode = tf.Variable(False, name='training')
            # No batchnorms durign testing
            vgg.build(
                val_images, output_shape=config.output_shape,
                train_mode=validation_mode)

    # Set up saver
    svm_feature_file = os.path.join(out_dir, 'svm_scores.npz')
    if os.path.exists(svm_feature_file) and force == False:
        svm_features = np.load(svm_feature_file)
        dec_scores = svm_features['dec_scores']
        label_vec = svm_features['label_vec']
    else:
        saver = tf.train.Saver(tf.global_variables())
        ckpts = [ckpts[selected_ckpts]]
        image_array = np.asarray(image_array)
        for idx, c in enumerate(ckpts):
            dec_scores = []
            # Initialize the graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(
                tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()))

            # Set up exemplar threading
            saver.restore(sess, c)
            num_batches = np.ceil(
                len(image_array) / config.validation_batch).astype(int)
            batch_idx = np.arange(
                num_batches).repeat(config.validation_batch + 1)[:len(image_array)]
            for bi in np.unique(batch_idx):
                # move this above to image processing
                batch_images = image_array[batch_idx == bi] / 255.
                start_time = time.time()
                sc = sess.run(vgg.fc7, feed_dict={val_images: batch_images})
                dec_scores.append(sc)
                print 'Batch %d took %.1f seconds' % (
                    bi, time.time() - start_time)

    # Save everything
    np.savez(
        svm_feature_file,
        dec_scores=dec_scores, label_vec=label_vec)

    # Build SVM
    dec_scores = np.concatenate(dec_scores[:], axis=0)
    model_array, score_array, combo_array, masked_label_array = [], [], [], []
    for combo in tqdm(itertools.combinations(np.unique(np_label_vec), 2), total=len(np.unique(np_label_vec))):
        combo_array.append(combo)
        mask = np.logical_or(np_label_vec == combo[0], np_label_vec == combo[1])
        masked_labels = np_label_vec[mask]
        masked_scores = dec_scores[mask, :]
        clf = SGDClassifier(loss='hinge')
        scores = cross_val_score(clf, masked_scores, masked_labels, cv=5)
        model_array.append(clf)
        score_array.append(scores)
        masked_label_array.append(masked_labels)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    import ipdb; ipdb.set_trace()
    # Save everything
    np.savez(
        os.path.join(out_dir, 'svm_models'),
        combo_array=combo_array, model_array=model_array,
        score_array=score_array, masked_label_array=masked_label_array)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--validation_data", type=str, dest="validation_data",
        default=None, help="Validation data tfrecords bin file.")
    parser.add_argument(
        "--model_dir", type=str, dest="model_dir",
        default=None, help="Feed in a specific model for validation.")
    parser.add_argument(
        "--label_file", type=str, dest="label_file",
        default='neuron_info.csv',
        help="CSV file containing labels for the svm.")
    parser.add_argument(
        "--force", dest="force", 
        action="store_true",
        help="Force run analyses.")
    args = parser.parse_args()
    test_vgg16(**vars(args))
