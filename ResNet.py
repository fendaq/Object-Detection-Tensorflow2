from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

layers = tf.keras.layers


def resnetv1_bottleneck(bottom, filters, strides, kernel_size=3, conv_shortcut=False, name=None, conv_trainable=True, bn_trainable=True):
    if conv_shortcut == True:
        shortcut = layers.Conv2D(4*filters, 1, strides, 'same',  name=name+'_0_conv', trainable=conv_trainable)(bottom)
        shortcut = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_0_bn', trainable=bn_trainable)(shortcut)
    else:
        shortcut = bottom

    conv = layers.Conv2D(filters, 1, strides, 'same',  name=name+'_1_conv', trainable=conv_trainable)(bottom)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name=name+'_1_relu')(conv)

    conv = layers.Conv2D(filters, kernel_size, 1, 'same',  name=name+'_2_conv', trainable=conv_trainable)(conv)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_2_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name=name+'_2_relu')(conv)

    conv = layers.Conv2D(4*filters, 1, 1, 'same',  name=name+'_3_conv', trainable=conv_trainable)(conv)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_3_bn', trainable=bn_trainable)(conv)

    add = layers.Add(name=name+'_add')([shortcut, conv])
    relu = layers.Activation('relu', name=name+'_out')(add)
    return relu


def stack_resnetv1_bottleneck(bottom, filters, num_blocks, strides, kernel_size=3, name=None, conv_trainable=True, bn_trainable=True):
    block = resnetv1_bottleneck(bottom, filters, strides, kernel_size, conv_shortcut=True, name=name+'_block1', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    for i in range(2, num_blocks+1):
        block = resnetv1_bottleneck(block, filters, 1, kernel_size, name=name+'_block'+str(i), conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    return block


def resnetv2_bottleneck(bottom, filters, strides, kernel_size=3, conv_shortcut=False, name=None, conv_trainable=True, bn_trainable=True):
    preact = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_preact_bn', trainable=bn_trainable)(bottom)
    preact = layers.Activation('relu', name=name+'_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4*filters, 1, strides, 'same',  name=name+'_0_conv', trainable=conv_trainable)(preact)
    else:
        shortcut = bottom

    conv = layers.Conv2D(filters, 1, strides, 'same', use_bias=False, name=name+'_1_conv', trainable=conv_trainable)(preact)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name=name+'_1_relu')(conv)

    conv = layers.Conv2D(filters, kernel_size, 1, 'same', use_bias=False, name=name+'_2_conv', trainable=conv_trainable)(conv)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_2_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name=name+'_2_relu')(conv)

    conv = layers.Conv2D(4*filters, 1, 1, 'same',  name=name+'_3_conv', trainable=conv_trainable)(conv)
    conv = layers.Add(name=name+'_out')([shortcut, conv])

    return conv


def stack_resnetv2_bottleneck(bottom, filters, num_blocks, stride, name=None, conv_trainable=True, bn_trainable=True):
    block = resnetv2_bottleneck(bottom, filters, stride, conv_shortcut=True, name=name+'_block1', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    for i in range(2, num_blocks+1):
        block = resnetv2_bottleneck(block, filters, 1, name=name+'_block'+str(i), conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    return block


def resnext_bottlebeck(bottom, filters, strides, kernel_size=3, groups=32, conv_shortcut=False, name=None, conv_trainable=True, bn_trainable=True):
    assert filters % groups == 0
    if conv_shortcut:
        shortcut = layers.Conv2D((64//groups)*filters, 1, strides, 'same', use_bias=False, name=name+'_0_conv', trainable=conv_trainable)(bottom)
        shortcut = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_0_bn', trainable=bn_trainable)(shortcut)
    else:
        shortcut = bottom

    conv = layers.Conv2D(filters, 1, strides, 'same', use_bias=False, name=name+'_1_conv', trainable=conv_trainable)(bottom)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name=name+'_1_relu')(conv)

    c = filters // groups
    dwconv = layers.DepthwiseConv2D(kernel_size, 1, 'same', depth_multiplier=c, use_bias=False, name=name+'_2_conv', trainable=conv_trainable)(conv)
    dwconv_shape = tf.shape(dwconv)
    dwconv = tf.reshape(dwconv, [dwconv_shape[0], dwconv_shape[1], dwconv_shape[2], c, filters])
    dwconv = tf.reshape(dwconv, [dwconv_shape[0], dwconv_shape[1], dwconv_shape[2], c, groups, c])
    dwconv = tf.reduce_sum(dwconv, axis=-1)
    dwconv = tf.reshape(dwconv, [dwconv_shape[0], dwconv_shape[1], dwconv_shape[2], filters])
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_2_bn', trainable=bn_trainable)(dwconv)
    conv = layers.Activation('relu', name=name+'_2_relu')(conv)

    conv = layers.Conv2D((64//groups)*filters, 1, 1, 'same', use_bias=False, name=name+'_3_conv', trainable=conv_trainable)(conv)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name=name+'_3_bn', trainable=bn_trainable)(conv)

    add = layers.Add(name=name+'_add')([shortcut, conv])
    relu = layers.Activation('relu', name=name+'_out')(add)
    return relu


def stack_resnext_bottleneck(bottom, filters, num_blocks, stride, kernel_size=3, groups=32, name=None, conv_trainable=True, bn_trainable=True):
    block = resnext_bottlebeck(bottom, filters, stride, kernel_size, groups, conv_shortcut=True, name=name+'_block1', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    for i in range(2, num_blocks+1):
        block = resnext_bottlebeck(block, filters, 1, kernel_size, groups, name=name+'_block'+str(i), conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    return block


def resnetv1_50(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]
    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', trainable=conv_trainable)(input)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='conv1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='conv1_relu')(conv)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 64, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnetv1_bottleneck(conv, 128, 4, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 256, 6, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 512, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnetv1_50')
    if weight is not None:
        model.load_weights(weight)
    return model


def resnetv1_101(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]
    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', trainable=conv_trainable)(input)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='conv1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='conv1_relu')(conv)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 64, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnetv1_bottleneck(conv, 128, 4, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 256, 23, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 512, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnetv1_101')
    if weight is not None:
        model.load_weights(weight)
    return model


def resnetv1_152(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]
    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', trainable=conv_trainable)(input)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='conv1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='conv1_relu')(conv)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 64, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnetv1_bottleneck(conv, 128, 8, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 256, 36, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv1_bottleneck(conv, 512, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnetv1_152')
    if weight is not None:
        model.load_weights(weight)
    return model


def resnetv2_50(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]
    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', trainable=conv_trainable)(input)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 64, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnetv2_bottleneck(conv, 128, 4, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 256, 6, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 512, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='post_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='post_relu')(conv)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnetv2_50')
    if weight is not None:
        model.load_weights(weight)
    return model


def resnetv2_101(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]
    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', trainable=conv_trainable)(input)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 64, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnetv2_bottleneck(conv, 128, 4, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 256, 23, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 512, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='post_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='post_relu')(conv)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnetv2_101')
    if weight is not None:
        model.load_weights(weight)
    return model


def resnetv2_152(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]
    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', trainable=conv_trainable)(input)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 64, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnetv2_bottleneck(conv, 128, 8, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 256, 36, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnetv2_bottleneck(conv, 512, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='post_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='post_relu')(conv)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnetv2_152')
    if weight is not None:
        model.load_weights(weight)
    return model


def resnext_50(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]

    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', use_bias=False, trainable=conv_trainable)(input)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='conv1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='conv1_relu')(conv)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnext_bottleneck(conv, 128, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnext_bottleneck(conv, 256, 4, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnext_bottleneck(conv, 512, 6, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnext_bottleneck(conv, 1024, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnext_50')
    if weight is not None:
        model.load_weights(weight)
    return model


def resnext_101(input,conv_trainable=True, bn_trainable=True, weight=None):
    """

    :param input: tensor of 'nhwc'
    :param conv_trainable: whether the conv layers in net could trainable
    :param bn_trainable: whether the bn layers in net could trainable
    :param weight: if not None, the weight will load in net
    :return: [features maps stride 8, stride 16, stride 32]
    """
    endpoints = []
    conv = layers.Conv2D(64, 7, 2, 'same',  name='conv1_conv', use_bias=False, trainable=conv_trainable)(input)
    conv = layers.BatchNormalization(3, epsilon=1.001e-5, name='conv1_bn', trainable=bn_trainable)(conv)
    conv = layers.Activation('relu', name='conv1_relu')(conv)
    conv = layers.MaxPool2D(3, 2, 'same', name='pool1_pool')(conv)
    endpoints.append(conv)
    conv = stack_resnext_bottleneck(conv, 128, 3, 1, name='conv2', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    conv = stack_resnext_bottleneck(conv, 256, 4, 2, name='conv3', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnext_bottleneck(conv, 512, 23, 2, name='conv4', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    conv = stack_resnext_bottleneck(conv, 1024, 3, 2, name='conv5', conv_trainable=conv_trainable, bn_trainable=bn_trainable)
    endpoints.append(conv)
    model = tf.keras.Model(inputs=input, outputs=endpoints, name='resnext_101')
    if weight is not None:
        model.load_weights(weight)
    return model
