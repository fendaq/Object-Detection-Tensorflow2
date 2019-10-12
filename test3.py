import tensorflow as tf
from skimage import io, transform
import numpy as np
# tf.compat.v1.disable_eager_execution()

img = io.imread('elephant.jpg')
img = transform.resize(img, (224, 224))
img = np.asarray(img, dtype=np.float32)
img = np.expand_dims(img, 0)
from YOLO3 import YOLO3

config = {
    'num_classes': 20,
    'batch_size':1,
    'mode': 'test',
    'backone_conv_trainable': False,
    'backone_bn_trainable': False,
    'backone_weight': 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
}
x1 = tf.ones([2, 224, 224, 3], dtype=tf.float32)*125.
y = tf.constant([20, 40, 50, 70, 3], dtype=tf.float32, shape=[1, 5])
y = tf.pad(
                                y, [[0, 10-tf.shape(y)[0]], [0, 0]],
                                constant_values=-1.0
          )
y = tf.expand_dims(y, 0)
y = tf.concat([y, y], axis=0)
ssd = YOLO3(config)
a = ssd(x1)

print(a)