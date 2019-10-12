from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def smooth_l1_loss(x):
    """

    :param x: tensor of size n x m x k x ......
    :return: tensor that has same size as input
    """
    return tf.where(tf.abs(x) < 1., 0.5*x*x, tf.abs(x)-0.5)


def focal_loss(pos_pred, neg_pred=None, alpha=0.25, gamma=2.):
    """

    :param pos: tensor of size n x m x k x ......, whose ground_truth is 1.
    :param neg: tensor of size n x m x k x ......, whose ground_truth is 0.
    :param alpha: hyper-parameter for focal loss
    :param gamma: hyper-parameter for focal loss
    :return: tensor that has same size as input
    """
    pos_loss = -tf.log(pos_pred) * tf.pow(1.-pos_pred, gamma) * alpha
    if neg_pred is not None:
        neg_loss = -tf.log(1.-neg_pred) * tf.pow(neg_pred, gamma) * alpha
        return pos_loss, neg_loss
    return pos_loss


def iou_loss(bbox1, bbox2):
    """

    :param bbox1: tensor of size n x 4
    :param bbox2: tensor of size n x 4
    :return: size of n
    """
    bbox1_y1x1 = bbox1[:, :2]
    bbox1_y2x2 = bbox1[:, 2:]
    bbox2_y1x1 = bbox2[:, :2]
    bbox2_y2x2 = bbox2[:, 2:]
    iou_y1x1 = tf.maximum(bbox1_y1x1, bbox2_y1x1)
    iou_y2x2 = tf.minimum(bbox1_y2x2, bbox2_y2x2)
    iou_area = tf.reduce_prod(tf.maximum(iou_y2x2-iou_y1x1, 0.), axis=-1)
    bbox1_area = tf.reduce_prod(tf.maximum(bbox1_y2x2-bbox1_y1x1, 0.), axis=-1)
    bbox2_area = tf.reduce_prod(tf.maximum(bbox2_y2x2-bbox2_y1x1, 0.), axis=-1)
    iou = (iou_area+1.) / (bbox1_area + bbox2_area - iou_area + 1.)
    iou_loss = -tf.log(iou)
    return iou_loss


def giou_loss(bbox1, bbox2):
    """

    :param bbox1: tensor of size n x 4
    :param bbox2: tensor of size n x 4
    :return: tensor of size n
    """
    bbox1_y1x1 = bbox1[:, :2]
    bbox1_y2x2 = bbox1[:, 2:]
    bbox2_y1x1 = bbox2[:, :2]
    bbox2_y2x2 = bbox2[:, 2:]
    iou_y1x1 = tf.maximum(bbox1_y1x1, bbox2_y1x1)
    iou_y2x2 = tf.minimum(bbox1_y2x2, bbox2_y2x2)
    enc_y1x1 = tf.minimum(bbox1_y1x1, bbox2_y1x1)
    enc_y2x2 = tf.maximum(bbox1_y2x2, bbox2_y2x2)
    iou_area = tf.reduce_prod(tf.maximum(iou_y2x2-iou_y1x1, 0.), axis=-1)
    enc_area = tf.reduce_prod(tf.maximum(enc_y2x2-enc_y1x1, 0.), axis=-1)
    bbox1_area = tf.reduce_prod(tf.maximum(bbox1_y2x2-bbox1_y1x1, 0.), axis=-1)
    bbox2_area = tf.reduce_prod(tf.maximum(bbox2_y2x2-bbox2_y1x1, 0.), axis=-1)
    iou = (iou_area + 1.) / (bbox1_area + bbox2_area - iou_area + 1.)
    giou = iou - (enc_area - iou_area + 1.) / (enc_area + 1.)
    giou_loss = -tf.log(giou)
    return giou_loss
