from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

layers = tf.keras.layers


def fpn_generator(endpoints, channels_output, nums_output, mode='dconv'):
    """

    :param endpoints: a list of feature maps for construct pyramid
    :param channels_output: the channels of every layers of fpn
    :param nums_output: number of layers of fpn
    :param mode: "dconv", "resize"
    :return: a list of feature maps whose length is nums_output
    """
    nums_input = len(endpoints)
    if nums_output <= nums_input:
        endpoints = endpoints[:nums_output]
    endpoints = [layers.Conv2D(channels_output, 1, 1, 'same')(x) for x in endpoints]
    endpoints = [layers.BatchNormalization(3, epsilon=1.001e-5)(x) for x in endpoints]
    endpoints = [layers.Activation('relu')(x) for x in endpoints]
    pyramid = [endpoints[-1]]
    top = endpoints[-1]
    for i in range(2, len(endpoints)+1):
        if mode == 'dconv':
            top_upsampling = layers.Conv2DTranspose(channels_output, 4, 2, 'same')(top)
            top_upsampling = layers.BatchNormalization(3, epsilon=1.001e-5)(top_upsampling)
            top_upsampling = layers.Activation('relu')(top_upsampling)
            top = layers.Add()([top_upsampling, endpoints[-i]])
            pyramid.insert(0, top)
        else:
            resize_shape = [tf.shape(endpoints[-i])[1],tf.shape(endpoints[-i])[2]]
            top_upsampling = tf.image.resize(top, resize_shape)
            top_upsampling = layers.BatchNormalization(3, epsilon=1.001e-5)(top_upsampling)
            top_upsampling = layers.Activation('relu')(top_upsampling)
            top = layers.Add()([top_upsampling, endpoints[-i]])
            pyramid.insert(0, top)
    if nums_output > nums_input:
        for i in range(nums_output-nums_input):
            down = layers.Conv2D(channels_output, 3, 2, 'same')(pyramid[-1])
            down = layers.BatchNormalization(3, epsilon=1.001e-5)(down)
            down = layers.Activation('relu')(down)
            pyramid.append(down)
    return pyramid

