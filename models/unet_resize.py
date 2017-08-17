import numpy as np
import tensorflow as tf
import gc
import re


class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(
                self, trainable=True,
                fine_tune_layers=None):
        self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.joint_label_output_keys = []

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(
            self,
            rgb,
            output_shape,
            train_mode=None,
            batchnorm=None,
            ):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """

        input_bgr = tf.identity(rgb, name="lrp_input")
        layer_structure = [
            {
                'layers': ['conv', 'res', 'pool'],
                'weights': [64, [64, 64], None],
                'names': ['conv1_1', 'conv1_2', 'pool1'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'res', 'pool'],
                'weights': [128, [128, 128], None],
                'names': ['conv2_1', 'conv2_2', 'pool2'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'res', 'pool'],
                'weights': [128, [128, 128], None],
                'names': ['conv3_1', 'conv3_2', 'pool3'],
                'filter_size': [3, 3, None]
            }]

        output_layer = self.create_conv_tower(
            input_bgr,
            layer_structure,
            tower_name='highres_conv')

        # Classification embedding
        self.fc1 = tf.nn.dropout(
            self.conv_layer(
                output_layer,
                int(output_layer.get_shape()[-1]),
                32,
                "fc1",
                filter_size=1),
            0.5)
        self.fc_pool1 = self.max_pool(self.fc1, 'fc_pool1')
        self.fc2 = tf.nn.dropout(
            self.conv_layer(
                self.fc_pool1,
                int(self.fc_pool1.get_shape()[-1]),
                64,
                "fc2",
                filter_size=1),
            0.5)
        self.fc_pool2 = self.max_pool(self.fc2, 'fc_pool2')
        self.flat_pool2 = tf.contrib.layers.flatten(self.fc_pool2)
        self.output = tf.nn.dropout(
            self.fc_layer(
                self.flat_pool2,
                int(self.flat_pool2.get_shape()[-1]),
                output_shape,
                "fc3"),
            0.5)
        self.prob = tf.nn.softmax(self.output)

        # Deconv to full image
        tl = [int(x) for x in self.pool2.get_shape()]
        input_layer = tf.image.resize_images(output_layer, tl[1:3])
        layer_structure = [
            {
                'layers': ['conv', 'res'],
                'weights': [tl[-1], [tl[-1], tl[-1]]],
                'names': ['up-conv2_1', 'up-conv2_2'],
                'filter_size': [3, 3],
            }
        ]
        self.d2 = self.create_conv_tower(
            input_layer,
            layer_structure,
            tower_name='d2')
        self.d2 += self.pool2

        tl = [int(x) for x in self.pool1.get_shape()]
        input_layer = tf.image.resize_images(self.d2, tl[1:3])
        layer_structure = [
            {
                'layers': ['conv', 'res'],
                'weights': [tl[-1], [tl[-1], tl[-1]]],
                'names': ['up-conv1_1', 'up-conv1_2'],
                'filter_size': [3]
            },
        ]
        self.d1 = self.create_conv_tower(
            input_layer,
            layer_structure,
            tower_name='d1')
        self.d1 += self.pool1

        tl = [int(x) for x in rgb.get_shape()]
        input_layer = tf.image.resize_images(self.d1, tl[1:3])
        layer_structure = [
            {
                'layers': ['conv'],
                'weights': [1],
                'names': ['up-conv0'],
                'filter_size': [1]
            },
        ]

        self.deconv_output = self.create_conv_tower(
            input_layer,
            layer_structure,
            tower_name='d0')

        self.data_dict = None

    def resnet_layer(
            self,
            bottom,
            layer_weights,
            layer_name):
        ln = '%s_branch' % layer_name
        branch_conv = tf.identity(bottom)
        for idx, lw in enumerate(layer_weights):
            ln = '%s_%s' % (layer_name, idx)
            branch_conv = self.conv_layer(
                branch_conv,
                int(branch_conv.get_shape()[-1]),
                lw,
                batchnorm=[ln],
                name=ln)
        return branch_conv + bottom

    def create_conv_tower(self, act, layer_structure, tower_name):
        print 'Creating tower: %s' % tower_name
        with tf.variable_scope(tower_name):
            for layer in layer_structure:
                for la, we, na, fs in zip(
                        layer['layers'],
                        layer['weights'],
                        layer['names'],
                        layer['filter_size']):
                    if la == 'pool':
                        act = self.max_pool(
                            bottom=act,
                            name=na)
                    elif la == 'conv':
                        act = self.conv_layer(
                            bottom=act,
                            in_channels=int(act.get_shape()[-1]),
                            out_channels=we,
                            name=na,
                            filter_size=fs
                        )
                    elif la == 'res':
                        act = self.resnet_layer(
                            bottom=act,
                            layer_weights=we,
                            layer_name=na
                        )
                    elif la == 'deconv':
                        act = self.deconv_layer(
                            bottom=act,
                            out_channels=we,
                            layer_name=na,
                            filter_size=fs
                        )
                    setattr(self, na, act)
                    print 'Added layer: %s' % na
        return act

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
                    out_channels, name, filter_size=3, batchnorm=None,
                    stride=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            if batchnorm is not None:
                if name in batchnorm:
                    relu = self.batchnorm(relu)

            return relu

    def deconv_layer(
            self,
            bottom,
            out_channels,
            filter_size,
            layer_name,
            strides=[1, 2, 2, 1],
            shape=None,
            padding='SAME'):

        pool_layer = 'pool%s' % re.search('\d+', layer_name).group()
        in_channels = bottom.get_shape()[3].value
        with tf.variable_scope(layer_name):

            if shape is None:
                # Compute shape out of Bottom
                in_shape = [int(s) for s in bottom.get_shape()]
                h = ((in_shape[1] - 1) * strides[1]) + 1
                w = ((in_shape[2] - 1) * strides[1]) + 1
                new_shape = [in_shape[0], h, w, out_channels]
            else:
                new_shape = [shape[0], shape[1], shape[2], out_channels]
            output_shape = tf.stack(new_shape)
            f_shape = [filter_size, filter_size, out_channels, in_channels]

            # create
            # num_input = filter_size * filter_size * in_channels / strides
            # stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(
                bottom,
                weights,
                output_shape,
                strides=strides,
                padding='SAME')
            if hasattr(self, pool_layer): 
                score = self.conv_layer(
                    bottom=self[pool_layer],
                    in_channels=int(self[pool_layer].get_shape()[-1]),
                    out_channels=out_channels,
                    name='%s_conv' % layer_name,
                    stride=[1, 1, 1, 1],
                    filter_size=filter_size
                )
            else:
                score = tf.constant(0.)
        return tf.pad(deconv, [[0, 0], [1, 0], [0, 1], [0, 0]]) + score
        # return deconv + score

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = np.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(
            value=weights,
            dtype=tf.float32)
        return tf.get_variable(
            name="up_filter",
            initializer=init,
            shape=weights.shape)

    def upsample_filt(size):
        """
        Make a 2D bilinear kernel suitable for upsampling
        of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

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
        num_files = 0

        for (name, idx), var in self.var_dict.items():
            # print(var.get_shape())
            if name == 'fc6':
                np.save(npy_path+str(num_files), data_dict)
                data_dict.clear()
                gc.collect()
                num_files += 1
                for i, item in enumerate(tf.split(var, 8, 0)):
                    # print(i)
                    name = 'fc6-' + str(i)
                    if name not in data_dict.keys():
                        data_dict[name] = {}
                    data_dict[name][idx] = sess.run(item)
                    np.save(npy_path+str(num_files), data_dict)
                    data_dict.clear()
                    gc.collect()
                    num_files += 1
            else:
                var_out = sess.run(var)
                if name not in data_dict.keys():
                    data_dict[name] = {}
                data_dict[name][idx] = var_out

        np.save(npy_path+str(num_files), data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
