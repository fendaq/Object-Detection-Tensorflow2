from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

layers = tf.keras.layers


def vgg16(input, conv_trainable=True, weight=None):
    endpoints = []
    conv = layers.Conv2D(64, 3, 1, 'same', activation='relu', name='block1_conv1', trainable=conv_trainable)(input)
    conv = layers.Conv2D(64, 3, 1, 'same', activation='relu', name='block1_conv2', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block1_pool')(conv)

    conv = layers.Conv2D(128, 3, 1, 'same', activation='relu', name='block2_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(128, 3, 1, 'same', activation='relu', name='block2_conv2', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block2_pool')(conv)

    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv2', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv3', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block3_pool')(conv)
    endpoints.append(pool)

    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv2', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv3', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block4_pool')(conv)
    endpoints.append(pool)

    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv2', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv3', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block5_pool')(conv)
    endpoints.append(pool)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='vgg16')
    if weight is not None:
        model.load_weights(weight)
    return model


def vgg19(input, conv_trainable=True, weight=None):
    endpoints = []
    conv = layers.Conv2D(64, 3, 1, 'same', activation='relu', name='block1_conv1', trainable=conv_trainable)(input)
    conv = layers.Conv2D(64, 3, 1, 'same', activation='relu', name='block1_conv2', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block1_pool')(conv)

    conv = layers.Conv2D(128, 3, 1, 'same', activation='relu', name='block2_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(128, 3, 1, 'same', activation='relu', name='block2_conv2', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block2_pool')(conv)

    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv2', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv3', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv4', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block3_pool')(conv)
    endpoints.append(pool)

    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv2', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv3', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block4_conv4', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block4_pool')(conv)
    endpoints.append(pool)

    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv1', trainable=conv_trainable)(pool)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv2', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv3', trainable=conv_trainable)(conv)
    conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block5_conv4', trainable=conv_trainable)(conv)
    pool = layers.MaxPool2D(2, 2, 'same', name='block5_pool')(conv)
    endpoints.append(pool)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='vgg16')
    if weight is not None:
        model.load_weights(weight)
    return model




