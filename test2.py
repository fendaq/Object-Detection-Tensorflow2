import tensorflow as tf
from skimage import io, transform
import numpy as np
# tf.compat.v1.disable_eager_execution()

from SSD import SSD

config = {
    'num_classes': 20,
    'batch_size':1,
    'mode': 'test',
    'backone_conv_trainable': True,
    'backone_bn_trainable': True,
    'backone_weight': 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
}
x1 = tf.ones([1, 512, 512, 3], dtype=tf.float32)*125.
y = tf.constant([20, 40, 50, 70, 3], dtype=tf.float32, shape=[1, 5])
y = tf.pad(
            y, [[0, 10-tf.shape(y)[0]], [0, 0]],
            constant_values=-1.0
         )
y = tf.expand_dims(y, 0)
ssd = SSD(config)
a = ssd(x1)

print(a)