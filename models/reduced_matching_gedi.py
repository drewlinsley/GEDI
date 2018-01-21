import numpy as np
import tensorflow as tf


class model_struct:
    """
    A trainable model for matching.
    """

    def __init__(self, trainable=True):
        self.data_dict = None
        self.var_dict = {}
        self.trainable = trainable

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(self, rgb, output_shape=128):
        """
        Build the model.

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        rgb_scaled = rgb * 255.0  # Scale up to imagenet's uint8
        input_bgr = tf.identity(rgb_scaled, name="matching_input")
        self.conv1_1 = self.conv_layer(input_bgr, 1, 64, "mconv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "mconv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'mpool1')

        # R1
        self.conv2_1 = self.conv_layer(self.pool1, 64, 96, "mconv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 96, 96, "mconv2_2") #  + self.conv2_1
        self.conv2_3 = self.conv_layer(self.conv2_2, 96, 96, "mconv2_3") #  + self.conv2_2 + self.conv2_1
        self.merge_2 = self.conv2_1 + self.conv2_3 #  + self.conv2_3
        self.pool2 = self.max_pool(self.merge_2, 'mpool2')

        # R2
        self.conv3_1 = self.conv_layer(self.pool2, 96, 128, "mconv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 128, 128, "mconv3_2") #  + self.conv3_1
        self.conv3_3 = self.conv_layer(self.conv3_2, 128, 128, "mconv3_3") #  + self.conv3_1 + self.conv3_2
        self.merge_3 = self.conv3_1 + self.conv3_3  #  + self.conv3_3
        self.pool3 = self.max_pool(self.merge_3, 'mpool3')

        self.conv4_1 = self.conv_layer(self.pool3, 128, 128, "mconv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 128, 128, "mconv4_2") #  + self.conv4_1
        self.conv4_3 = self.conv_layer(self.conv4_2, 128, 128, "mconv4_3") #  + self.conv4_1 + self.conv4_2
        self.merge_4 = self.conv4_1 + self.conv4_3  #  + self.conv4_3
        self.pool4 = self.max_pool(self.merge_4, 'mpool4')

        # FC output
        flattened_pool4 = tf.contrib.layers.flatten(self.pool4)
        in_dims = int(flattened_pool4.get_shape()[-1])
        self.pre_output = self.fc_layer(
            flattened_pool4, in_dims, output_shape * 2, "moutput")
        self.pre_output = tf.nn.selu(self.pre_output)
        self.pre_output = tf.nn.dropout(self.pre_output, 0.5)
        self.output = self.fc_layer(
            self.pre_output, output_shape * 2, output_shape, "poutput")
        self.output = tf.nn.selu(self.output)
        self.output = tf.nn.dropout(self.output, 0.5)
        self.data_dict = None
        return self.output

    def batchnorm(self, layer):
        m, v = tf.nn.moments(layer, [0])
        return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(
                    self, bottom, in_channels,
                    out_channels, name, batchnorm=None, filter_size=5):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                5, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.selu(bias)

            if batchnorm is not None:
                if name in batchnorm:
                    relu = self.batchnorm(relu)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(
            self, filter_size, in_channels, out_channels,
            name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [filter_size, filter_size, in_channels, out_channels],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [filter_size, filter_size, in_channels, out_channels],
                0.0, 0.001)
        bias_init = tf.truncated_normal([out_channels], .0, .001)
        filters = self.get_var(weight_init, name, 0, name + "_filters")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(
            self, initial_value, name, idx,
            var_name, in_size=None, out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolian to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1])
            else:
                var = tf.get_variable(name=var_name, initializer=value)
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
