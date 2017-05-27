#!/usr/bin/env python
# VGG16 model with customizable filter sizes built from different model weights

import tensorflow as tf
from vgg16_trainable_tf10 import Vgg16

class Vgg16Surgery(Vgg16):
    def __init__(self, self_npy_path, submodel_names, submodel_npy_paths, trainable=True, fine_tune_layers=None):
        ''' Surgery model constructor '''
        super(Vgg16Surgery, self).__init__(vgg16_npy_path=self_npy_path, trainable=trainable, fine_tune_layers=fine_tune_layers)
        # Init sub-models from given parameter list
        assert (len(submodel_names) >= 1)
        assert (len(submodel_names) == len(submodel_npy_paths))
        self.submodels = [Vgg16(p, trainable=trainable, fine_tune_layers=fine_tune_layers) for p in submodel_npy_paths]
        self.submodel_names = submodel_names
        for sm, sn in zip(self.submodels, self.submodel_names): sm.name = sn
        self.n_submodels = len(self.submodels)

    def build(self, rgb, output_shape = None, train_mode=None):
        ''' Build surgery model '''
        # Assign parameters
        if output_shape is None:
            output_shape = 1000
        self.train_mode = train_mode
        self.output_shape = output_shape
        # Use common preprocessing
        self.rgb = rgb
        self.build_preprocessing()
        # Conv stage of sub-models
        for sm,sn in zip(self.submodels, self.submodel_names):
            with tf.variable_scope('sub_' + sn):
                sm.bgr = self.bgr
                sm.build_conv_layers()
        # Concatenate own combined pool5
        print 'sm0 pool5 shape: ', self.submodels[0].pool5.get_shape()
        self.pool5 = tf.concat(axis=[sm.pool5 for sm in self.submodels], axis=3, name='pool5_concat')
        print 'pool5 shape: ', self.pool5.get_shape()
        # Build fully connected layers on top
        self.build_fc_layers()
        self.build_classifier()
        # Discard data dictionaries
        self.data_dict = None
        for sm in self.submodels:
            sm.data_dict = None