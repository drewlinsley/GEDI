import os
import numpy as np
import tensorflow as tf
from exp_ops import data_loader
from gedi_config import GEDIconfig
from matplotlib import pyplot as plt

config = GEDIconfig()
train_data = os.path.join(config.tfrecord_dir, 'train.tfrecords')
print 'Using tfrecord: %s' % train_data
max_value = np.max(
    np.load(
        os.path.join(
            config.tfrecord_dir, 'train_' + config.max_file))['max_array'])
print 'Derived max: %s' % max_value

image, label, res_image = data_loader.read_and_decode_single_example(
    filename=train_data, im_size=[300, 300, 3],
    model_input_shape=[224, 224, 3],
    train=[None], max_value=config.max_gedi, normalize=True)

sess = tf.Session()

# Required. See below for explanation
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# first example from file
label_val, image_val, res_image = sess.run([label, image, res_image])
print label_val
plt.imshow(image_val / image_val.max())
plt.show()

plt.imshow(res_image)
plt.show()
