import numpy as np
import tensorflow as tf


class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(
                self, vgg16_npy_path=None, trainable=True,
                fine_tune_layers=None):
        try:
            if vgg16_npy_path is not None:
                self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
                if fine_tune_layers is not None:
                    for key in fine_tune_layers:
                        del self.data_dict[key]
            else:
                self.data_dict = None
        except:
            print 'Could not find file: %s' % vgg16_npy_path

        self.var_dict = {}
        self.trainable = trainable
        self.VGG_MEAN = [103.939, 116.779, 123.68]

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(
            self,
            rgb,
            output_shape=None,
            train_mode=None,
            batchnorm=None,
            include_GEDI=True):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """
        rgb_scaled = rgb * 255.0  # Scale up to imagenet's uint8

        # Convert RGB to BGR
        if int(rgb.get_shape()[-1]) == 1:
            red, green, blue = rgb_scaled, rgb_scaled, rgb_scaled
        else:
            red, green, blue = tf.split(
                axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ], name='bgr')

        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        input_bgr = tf.identity(bgr, name="lrp_input")
        self.conv1_1 = self.conv_layer(
            input_bgr, 3, 64, "conv1_1", trainable=False)
        self.conv1_2 = self.conv_layer(
            self.conv1_1, 64, 64, "conv1_2", trainable=False)
        self.pool1 = self.max_pool(
            self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(
            self.pool1, 64, 128, "conv2_1", trainable=False)
        self.conv2_2 = self.conv_layer(
            self.conv2_1, 128, 128, "conv2_2", trainable=False)
        self.pool2 = self.max_pool(
            self.conv2_2, 'pool2')
        if include_GEDI:
            self.conv3_1 = self.conv_layer(
                self.pool2, 128, 256, "conv3_1", trainable=False)
            self.conv3_2 = self.conv_layer(
                self.conv3_1, 256, 256, "conv3_2", trainable=False)
            self.conv3_3 = self.conv_layer(
                self.conv3_2, 256, 256, "conv3_3", trainable=False)
            self.pool3 = self.max_pool(
                self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(
                self.pool3, 256, 512, "conv4_1", trainable=False)
            self.conv4_2 = self.conv_layer(
                self.conv4_1, 512, 512, "conv4_2", trainable=False)
            self.conv4_3 = self.conv_layer(
                self.conv4_2, 512, 512, "conv4_3", trainable=False)
            self.pool4 = self.max_pool(
                self.conv4_3, 'pool4')

            self.conv5_1 = self.conv_layer(
                self.pool4, 512, 512, "conv5_1", batchnorm, trainable=False)
            self.conv5_2 = self.conv_layer(
                self.conv5_1, 512, 512, "conv5_2", batchnorm, trainable=False)
            self.conv5_3 = self.conv_layer(
                self.conv5_2, 512, 512, "conv5_3", batchnorm, trainable=False)
            self.pool5 = self.max_pool(
                self.conv5_3, 'pool5')

            # 25088 = ((224 / (2 ** 5)) ** 2) * 512
            self.fc6 = self.fc_layer(
                self.pool5, 25088, 4096, "fc6", trainable=False)
            self.relu6 = tf.nn.relu(self.fc6)
            # Consider changing these to numpy conditionals
            if train_mode is not None:
                self.relu6 = tf.cond(
                    train_mode,
                    lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
            elif self.trainable:
                self.relu6 = tf.nn.dropout(self.relu6, 0.5)
            if batchnorm is not None:
                if 'fc6' in batchnorm:
                    self.relu6 = self.batchnorm(self.relu6)

            self.fc7 = self.fc_layer(
                self.relu6, 4096, 4096, "fc7", trainable=False)
            self.relu7 = tf.nn.relu(self.fc7)
            if train_mode is not None:
                self.relu7 = tf.cond(
                    train_mode,
                    lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
            elif self.trainable:
                self.relu7 = tf.nn.dropout(self.relu7, 0.5)
            if batchnorm is not None:
                if 'fc7' in batchnorm:
                    self.relu7 = self.batchnorm(self.relu7)

            # Train a new output layer for VGG
            self.vgg_fc = tf.nn.selu(
                self.fc_layer(
                    self.fc7, 4096, output_shape, 'vgg_fc', trainable=True))
        else:
            output_shape *= 2
        # Next create a atrous conv model w/ VGG features
        self.a1 = self.dconv_layer(
            self.pool2, 128, 128, 'a1', selu=True)
        self.a1_1 = self.conv_layer(
            self.a1, 128, 128, 'a1_1', selu=True)
        self.a1_2 = self.conv_layer(
            self.a1_1, 128, 128, 'a1_2', selu=True) + self.a1
        self.a2 = self.dconv_layer(
            self.a1_2, 128, 128, 'a2', selu=True)
        self.a2_1 = self.conv_layer(
            self.a2, 128, 128, 'a2_1', selu=True)
        self.a2_2 = self.conv_layer(
            self.a2_1, 128, 128, 'a2_2', selu=True) + self.a2
        self.a3 = self.dconv_layer(
            self.a2_2, 128, 256, 'a3', selu=True)
        self.a3_1 = self.conv_layer(
            self.a3, 256, 256, 'a3_1', selu=True)
        self.a3_2 = self.conv_layer(
            self.a3_1, 256, 256, 'a3_2', selu=True) + self.a3
        self.poolfc1 = self.max_pool(
            self.a3_2, 'poolfc1', ksize=4)
        self.fc1 = self.conv_layer(
            self.poolfc1, 256, 128, 'f1', filter_size=1)
        self.poolfc2 = self.max_pool(
            self.fc1, 'poolfc2', ksize=4)
        self.fc2 = self.conv_layer(
            self.poolfc2, 128, output_shape, 'f2', filter_size=1)
        self.poolfc3 = self.max_pool(
            self.fc2, 'poolfc3', ksize=4)
        self.flat_fc2 = tf.contrib.layers.flatten(self.poolfc3)
        if include_GEDI:
            self.output = tf.concat([self.vgg_fc, self.flat_fc2], axis=-1)
        else:
            self.output = self.flat_fc2
        self.data_dict = None
        return self.output

    def batchnorm(self, layer):
        m, v = tf.nn.moments(layer, [0])
        return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name, ksize=2):
        return tf.nn.max_pool(
            bottom, ksize=[1, ksize, ksize, 1],
            strides=[1, ksize, ksize, 1], padding='SAME', name=name)

    def conv_layer(
                    self, bottom, in_channels,
                    out_channels, name, batchnorm=None, filter_size=3, trainable=True, selu=False):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name, trainable=trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            if selu:
                relu = tf.nn.selu(bias)
            else:
                relu = tf.nn.relu(bias)

            if batchnorm is not None:
                if name in batchnorm:
                    relu = self.batchnorm(relu)

            return relu

    def dconv_layer(
                    self, bottom, in_channels,
                    out_channels, name, batchnorm=None, trainable=True, selu=False):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                3, in_channels, out_channels, name, trainable=trainable)

            conv = tf.nn.atrous_conv2d(bottom, filt, 2, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            if selu:
                relu = tf.nn.selu(bias)
            else:
                relu = tf.nn.relu(bias)


            if batchnorm is not None:
                if name in batchnorm:
                    relu = self.batchnorm(relu)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name, trainable=True):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable=trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(
            self, filter_size, in_channels, out_channels,
            name, init_type='xavier', trainable=True):
        if init_type == 'xavier':
            weight_init = [
                [filter_size, filter_size, in_channels, out_channels],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [filter_size, filter_size, in_channels, out_channels],
                0.0, 0.001)
        bias_init = tf.truncated_normal([out_channels], .0, .001)
        filters = self.get_var(weight_init, name, 0, name + "_filters", trainable=trainable)
        biases = self.get_var(bias_init, name, 1, name + "_biases", trainable=trainable)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, init_type='xavier', trainable=True):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights", trainable=trainable)
        biases = self.get_var(bias_init, name, 1, name + "_biases", trainable=trainable)

        return weights, biases

    def get_var(
            self, initial_value, name, idx,
            var_name, in_size=None, out_size=None, trainable=True):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolian to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1], trainable=trainable)
            else:
                var = tf.get_variable(name=var_name, initializer=value, trainable=trainable)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if name not in data_dict.keys():
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
