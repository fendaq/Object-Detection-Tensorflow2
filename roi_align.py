from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def roi_align(features, bboxes, size, align_corner=True):
    """

    :param features: a list of feature maps of fpn, size h x w x c
    :param bboxes: a list of tensor, [nx4, mx4, ...]
    :param size: a list of size [[a, b], [c, d], ...]
    :param align_corner:
    :return:
    """
    def roi_align_one_layer(feat, bbox, s, align_c):
        s_y, s_x = s
        num_bbox = tf.shape(bbox)[0]
        feat_shape = tf.shape(feat)
        h_i, w_i, c_i = feat_shape[0], feat_shape[1], feat_shape[2]
        h_f, w_f = tf.cast(h_i, tf.float32), tf.cast(w_i, tf.float32)
        bbox_y1 = bbox[:, 0:1]
        bbox_x1 = bbox[:, 1:2]
        bbox_y2 = bbox[:, 2:3]
        bbox_x2 = bbox[:, 3:4]
        bbox_h = bbox_y2 - bbox_y1
        bbox_w = bbox_x2 - bbox_x1
        if align_c:
            off_y = 0. if s_y > 1 else bbox_h / 2.
            off_x = 0. if s_x > 1 else bbox_w / 2.
            grid_y = tf.linspace(0.0, 1.0, s_y)
            grid_x = tf.linspace(0.0, 1.0, s_x)
        else:
            off_y = bbox_h / (2. * s_y)
            off_x = bbox_w / (2. * s_x)
            grid_y = tf.linspace(0.0, 1.0, s_y + 1)[:-1]
            grid_x = tf.linspace(0.0, 1.0, s_x + 1)[:-1]
        grid_y = tf.expand_dims(tf.matmul(bbox_h, tf.expand_dims(grid_y, axis=0)) + off_y + bbox_y1, axis=-1)
        grid_x = tf.expand_dims(tf.matmul(bbox_w, tf.expand_dims(grid_x, axis=0)) + off_x + bbox_x1, axis=-1)

        grid_y = tf.where(grid_y < 0., 0., grid_y)
        grid_x = tf.where(grid_x < 0., 0., grid_x)
        grid_y = tf.where(grid_y > h_f-1., h_f-1., grid_y)
        grid_x = tf.where(grid_x > w_f-1., w_f-1., grid_x)
        grid_y = tf.tile(grid_y, [1, 1, s_y])
        grid_x = tf.tile(grid_x, [1, 1, s_x])

        grid_y = tf.reshape(grid_y, [num_bbox, -1])
        grid_x = tf.reshape(grid_x, [num_bbox, -1])
        grid_y1 = tf.math.floor(grid_y)
        grid_y2 = tf.math.floor(grid_y+1.)
        grid_x1 = tf.math.floor(grid_x)
        grid_x2 = tf.math.floor(grid_x+1.)
        wey1 = tf.expand_dims(grid_y - grid_y1, axis=-1)
        wey2 = tf.expand_dims(grid_y2 - grid_y, axis=-1)
        wex1 = tf.expand_dims(grid_x - grid_x1, axis=-1)
        wex2 = tf.expand_dims(grid_x2 - grid_x, axis=-1)
        grid_y2 = tf.where(grid_y2 > h_f - 1., h_f - 1., grid_y2)
        grid_x2 = tf.where(grid_x2 > w_f - 1., w_f - 1., grid_x2)
        grid_y1 = tf.cast(grid_y1, tf.int32)
        grid_y2 = tf.cast(grid_y2, tf.int32)
        grid_x1 = tf.cast(grid_x1, tf.int32)
        grid_x2 = tf.cast(grid_x2, tf.int32)
        grid_11 = grid_y1 * w_i + grid_x1
        grid_12 = grid_y1 * w_i + grid_x2
        grid_21 = grid_y2 * w_i + grid_x1
        grid_22 = grid_y2 * w_i + grid_x2

        feat = tf.reshape(feat, [h_i*w_i, c_i])
        feat_11 = tf.gather(feat, grid_11)
        feat_12 = tf.gather(feat, grid_12)
        feat_21 = tf.gather(feat, grid_21)
        feat_22 = tf.gather(feat, grid_22)

        feat_bilinear = wey2 * (feat_11 * wex2 + feat_12 * wex1) + wey1 * (feat_21 * wex2 + feat_22 * wex1)
        feat_bilinear = tf.reshape(feat_bilinear, [num_bbox, s_y, s_x, c_i])
        return feat_bilinear
    out_feat = []
    for f, b, s in zip(features, bboxes, size):
        out_feat.append(roi_align_one_layer(f, b, s, align_corner))
    return out_feat








