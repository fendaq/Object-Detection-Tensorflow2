from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def bbox_iou(bbox1, bbox2):
    """

    :param bbox1: boxes of size n x 4, mode 'y1x1y2x2'
    :param bbox2: boxes of size m x 4, mode 'y1x1y2x2'
    :return: iou of size n x m
    """
    bbox1_y1x1 = bbox1[:, :2]
    bbox1_y2x2 = bbox1[:, 2:]
    bbox2_y1x1 = bbox2[:, :2]
    bbox2_y2x2 = bbox2[:, 2:]
    num2 = tf.shape(bbox2)[0]
    bbox1_y1x1 = tf.reshape(bbox1_y1x1, [-1, 1, 2])
    bbox1_y2x2 = tf.reshape(bbox1_y2x2, [-1, 1, 2])
    bbox2_y1x1 = tf.reshape(bbox2_y1x1, [1, -1, 2])
    bbox2_y2x2 = tf.reshape(bbox2_y2x2, [1, -1, 2])
    bbox1_y1x1 = tf.tile(bbox1_y1x1, [1, num2, 1])
    bbox1_y2x2 = tf.tile(bbox1_y2x2, [1, num2, 1])

    iou_y1x1 = tf.maximum(bbox1_y1x1, bbox2_y1x1)
    iou_y2x2 = tf.minimum(bbox1_y2x2, bbox2_y2x2)
    iou_area = tf.reduce_prod(tf.maximum(iou_y2x2-iou_y1x1, 0.), axis=-1)
    bbox1_area = tf.reduce_prod(tf.maximum(bbox1_y2x2-bbox1_y1x1, 0.), axis=-1)
    bbox2_area = tf.reduce_prod(tf.maximum(bbox2_y2x2-bbox2_y1x1, 0.), axis=-1)
    iou = (iou_area+1.) / (bbox1_area + bbox2_area - iou_area + 1.)
    return iou

def bbox_iou2(bbox1, bbox2):
    """

    :param bbox1: boxes of size n x 4, mode 'y1x1y2x2'
    :param bbox2: boxes of size n x m x 4, mode 'y1x1y2x2'
    :return: iou of size n x m
    """
    bbox1_y1x1 = tf.expand_dims(bbox1[..., :2], axis=1)
    bbox1_y2x2 = tf.expand_dims(bbox1[..., 2:], axis=1)
    bbox2_y1x1 = bbox2[..., :2]
    bbox2_y2x2 = bbox2[..., 2:]

    iou_y1x1 = tf.maximum(bbox1_y1x1, bbox2_y1x1)
    iou_y2x2 = tf.minimum(bbox1_y2x2, bbox2_y2x2)
    iou_area = tf.reduce_prod(tf.maximum(iou_y2x2 - iou_y1x1, 0.), axis=-1)
    bbox1_area = tf.reduce_prod(tf.maximum(bbox1_y2x2 - bbox1_y1x1, 0.), axis=-1)
    bbox2_area = tf.reduce_prod(tf.maximum(bbox2_y2x2 - bbox2_y1x1, 0.), axis=-1)
    iou = (iou_area + 1.) / (bbox1_area + bbox2_area - iou_area + 1.)
    return iou



def partition_pos_neg_samples(gt_bboxes, anchors, labels, pos_threshold, neg_threshold, pred1, pred2):
    """

    :param gt_bboxes: ground truth bbox of size n x 4, mode 'y1x1y2x2'
    :param anchors: anchors of size n x 4, mode 'y1x1y2x2'
    :param pos_threshold: threshold for get positive samples
    :param neg_threshold: threshold for get negative samples
    :param labels: labels of corresponding gt_bboxes
    :param pred1: a list of tensor that only valid for positive samples
    for differenet fpn layers , for example, regressor
    :param pred2: a list of tensor that only valid for all positive and negitive samples
    for differenet fpn layers, for example, classifier
    :return:
    """
    def partition_one_layer(gt_bb, anch, lab, pos_thre, neg_thre, p1, p2):
        num_a = tf.shape(anch)[0]
        gaiou = bbox_iou(gt_bb, anch)
        best_index = tf.argmax(gaiou, axis=-1)
        best_index_ = tf.expand_dims(best_index, axis=1)
        best_mask = tf.sparse.to_dense(tf.sparse.SparseTensor(best_index_, tf.ones_like(best_index, tf.float32),
                                                              dense_shape=[num_a]), validate_indices=False)
        agiou = tf.transpose(gaiou)
        max_agiou = tf.reduce_max(agiou, axis=-1)
        pos_mask = max_agiou > pos_thre
        neg_mask = max_agiou < neg_thre
        pos_mask = (tf.cast(pos_mask, tf.float32) - best_mask) > 0.
        neg_mask = (tf.cast(neg_mask, tf.float32) - best_mask) > 0.

        pos_agiou = tf.boolean_mask(agiou, pos_mask)
        pos_index = tf.argmax(pos_agiou, axis=-1)

        pos_gt_bb = tf.concat([gt_bb, tf.gather(gt_bb, pos_index)], axis=0)
        pos_anch = tf.concat([tf.gather(anch, best_index), tf.boolean_mask(anch, pos_mask)], axis=0)
        neg_anch = tf.boolean_mask(anch, neg_mask)
        pos_lab = tf.concat([lab, tf.gather(lab, pos_index)], axis=0)
        pos_p1 = tf.concat([tf.gather(p1, best_index), tf.boolean_mask(p1, pos_mask)], axis=0)
        pos_p2 = tf.concat([tf.gather(p2, best_index), tf.boolean_mask(p2, pos_mask)], axis=0)
        neg_p2 = tf.boolean_mask(p2, neg_mask)
        return pos_gt_bb, pos_anch, neg_anch, pos_lab, pos_p1, pos_p2, neg_p2

    pos_gt_bboxes = []
    pos_anchors = []
    neg_anchors = []
    pos_labels = []
    pos_pred1 = []
    pos_pred2 = []
    neg_pred2 = []
    for g, a, l, pt, nt, pd1, pd2 in zip(gt_bboxes, anchors, labels, pos_threshold, neg_threshold, pred1, pred2):
        pgt, panch, nanch, pl, pp1, pp2, np2 = partition_one_layer(g, a, l, pt, nt, pd1, pd2)
        pos_gt_bboxes.append(pgt)
        pos_anchors.append(panch)
        neg_anchors.append(nanch)
        pos_labels.append(pl)
        pos_pred1.append(pp1)
        pos_pred2.append(pp2)
        neg_pred2.append(np2)
    return pos_gt_bboxes, pos_anchors, neg_anchors, pos_labels, pos_pred1, pos_pred2, neg_pred2



def partition_pos_neg_samples_yolo(gt_bboxes, anchors, labels, threshold, pred1, pred2):
    """

    :param gt_bboxes: a list of gt_bboxes that has corresponding size of layers of fpns, mode 'y1x1y2x2', [nx4 nx4....]
    :param anchors: a list of anchors that has corresponding size of layers of fpns, mode 'y1x1y2x2', [nx4 nx4....]
    :param threshold: threshold for get negative samples
    :param labels: labels of gt_bboxes
    :param pred1: a list of tensor that only valid for positive samples
    for differenet fpn layers , for example, regressor
    :param pred2: a of list tensor that only valid for all positive and negitive samples
    for differenet fpn layers, for example, classifier
    :return:
    """
    def partition_one_layer(gt, anch, p1, p2):

        anch_shape = tf.shape(anch)
        h, w, num_anch = anch_shape[0], anch_shape[1], anch_shape[2]
        anch = tf.reshape(anch, [-1, num_anch, 4])

        y1 = tf.cast(gt[:, 0], tf.int32)
        x1 = tf.cast(gt[:, 1], tf.int32)
        center_index = tf.cast(y1 * w + x1, tf.int64)
        center_anch = tf.gather(anch, center_index)
        center_index_ = tf.expand_dims(center_index, axis=1)
        center_mask = tf.sparse.to_dense(tf.sparse.SparseTensor(center_index_, tf.ones_like(center_index, tf.float32),
                                         dense_shape=[h*w]), validate_indices=False)
        other_mask = (1. - center_mask) > 0.
        center_mask = center_mask > 0.
        other_anch = tf.boolean_mask(anch, other_mask)

        center_p1 = tf.boolean_mask(p1, center_mask)
        center_p2 = tf.boolean_mask(p2, center_mask)
        other_p2 = tf.reshape(tf.boolean_mask(p2, other_mask), [-1, tf.shape(p2)[-1]])
        other_anch = tf.reshape(other_anch, [-1, 4])

        num_g = tf.shape(gt)[0]
        gciou = bbox_iou2(gt, center_anch)
        max_iou = tf.reduce_max(gciou, axis=-1)
        max_iou_index = tf.expand_dims(tf.argmax(gciou, axis=-1), axis=1)
        max_iou_index = tf.concat([tf.expand_dims(tf.range(num_g, dtype=tf.int64), 1), max_iou_index], axis=-1)
        center_anch = tf.gather_nd(center_anch, max_iou_index)
        center_p1 = tf.gather_nd(center_p1, max_iou_index)
        center_p2 = tf.gather_nd(center_p2, max_iou_index)

        ogiou = bbox_iou(gt, other_anch)
        max_ogiou = tf.reduce_max(ogiou, axis=-1)
        neg_mask = max_ogiou < threshold
        neg_p2 = tf.boolean_mask(other_p2, neg_mask)

        return max_iou, center_anch, other_anch, center_p1, center_p2, neg_p2

    max_iou = []
    center_anchors = []
    other_anchors = []
    center_pred1 = []
    center_pred2 = []
    neg_pred2 = []
    for g, a, p1, p2 in zip(gt_bboxes, anchors, pred1, pred2):
        mi, cen, oth, cen_p1, cen_p2, neg_p2 = partition_one_layer(g, a, p1, p2)
        max_iou.append(tf.expand_dims(mi, -1))
        center_anchors.append(tf.expand_dims(cen, 1))
        other_anchors.append(oth)

        center_pred1.append(tf.expand_dims(cen_p1, 1))
        center_pred2.append(tf.expand_dims(cen_p2, 1))
        neg_pred2.append(neg_p2)
    max_iou = tf.concat(max_iou, axis=-1)
    center_anchors = tf.concat(center_anchors, axis=1)
    center_pred1 = tf.concat(center_pred1, axis=1)
    center_pred2 = tf.concat(center_pred2, axis=1)

    neg_pred2 = tf.concat(neg_pred2, axis=0)

    max_iou_index = tf.expand_dims(tf.argmax(max_iou, axis=-1), axis=1)
    num_g = tf.shape(max_iou_index)[0]
    max_iou_index = tf.concat([tf.expand_dims(tf.range(num_g, dtype=tf.int64), 1), max_iou_index], axis=-1)
    best_anchors = tf.gather_nd(center_anchors, max_iou_index)
    best_pred1 = tf.gather_nd(center_pred1, max_iou_index)
    best_pred2 = tf.gather_nd(center_pred2, max_iou_index)
    gt_bboxes = [tf.expand_dims(x, 1) for x in gt_bboxes]
    gt_bboxes = tf.concat(gt_bboxes, axis=1)
    gt_bboxes = tf.gather_nd(gt_bboxes, max_iou_index)

    return gt_bboxes, best_anchors, labels, best_pred1, best_pred2, neg_pred2


def ohem(bboxes, scores, max_output, pred, iou_threshold):
    """

    :param bboxes: bboxes of size n x 4, mode 'y1x1y2x2'
    :param scores: scores of corresponding bbox, size n
    :param max_output: maximum number of reserved bboxes after ohem
    :param iou_threshold: iou_threshold for nms
    :param pred: a tensor needed select
    :return:
    """
    out_pred = []
    for b, s, m, p, i in zip(bboxes, scores, max_output, pred, iou_threshold):
        selected_indices = tf.image.non_max_suppression(
                b, s, m, iou_threshold=i
            )
        out_pred.append(tf.gather(p, selected_indices))
    return out_pred


def bbox_encode(gt_bboxes, anchors, normlization=(1., 1., 1., 1.)):
    normliaztion = tf.constant(normlization, dtype=tf.float32, shape=[1, 4])
    gt_bboxes_y1x1 = gt_bboxes[:, :2]
    gt_bboxes_y2x2 = gt_bboxes[:, 2:]
    anchors_y1x1 = anchors[:, :2]
    anchors_y2x2 = anchors[:, 2:]

    gt_bboxes_yx = gt_bboxes_y1x1/2. + gt_bboxes_y2x2/2.
    gt_bboxes_hw = gt_bboxes_y2x2 - gt_bboxes_y1x1
    anchors_yx = anchors_y1x1/2. + anchors_y2x2/2.
    anchors_hw = anchors_y2x2 - anchors_y1x1

    codes_yx = (gt_bboxes_yx - anchors_yx) / anchors_hw
    codes_hw = tf.math.log(gt_bboxes_hw/anchors_hw)
    codes = tf.concat([codes_yx, codes_hw], axis=-1) * normliaztion
    return codes


def bbox_decode(anchors, codes, normlization=(1., 1., 1., 1.)):
    normliaztion = tf.constant(normlization, dtype=tf.float32, shape=[1, 4])
    codes = codes / normliaztion
    anchors_yx = anchors[:, :2]
    anchors_hw = anchors[:, 2:]
    codes_yx = codes[:, :2]
    codes_hw = codes[:, 2:]

    pred_yx = codes_yx * anchors_hw + anchors_yx
    pred_hw = tf.exp(codes_hw) * anchors_hw
    pred_y1x1 = pred_yx - pred_hw/2.
    pred_y2x2 = pred_yx + pred_hw/2.
    pred = tf.concat([pred_y1x1, pred_y2x2], axis=-1)
    return pred


def bbox_encode_yolo(gt_bboxes, anchors, normlization=(1., 1.)):
    normliaztion = tf.constant(normlization, dtype=tf.float32, shape=[1, 2])
    gt_bboxes_y1x1 = gt_bboxes[:, :2]
    gt_bboxes_y2x2 = gt_bboxes[:, 2:]
    anchors_y1x1 = anchors[:, :2]
    anchors_y2x2 = anchors[:, 2:]

    gt_bboxes_yx = gt_bboxes_y1x1/2. + gt_bboxes_y2x2/2.
    gt_bboxes_hw = gt_bboxes_y2x2 - gt_bboxes_y1x1
    anchors_hw = anchors_y2x2 - anchors_y1x1

    codes_yx = gt_bboxes_yx - tf.cast(tf.floor(gt_bboxes_yx), tf.float32)
    codes_hw = tf.math.log(gt_bboxes_hw/anchors_hw) * normliaztion
    codes = tf.concat([codes_yx, codes_hw], axis=-1)
    return codes


def bbox_decode_yolo(anchors, codes, normlization=(1., 1.)):
    normlization = tf.constant(normlization, dtype=tf.float32, shape=[1, 2])
    anchors_y1x1 = anchors[:, :2]
    anchors_y2x2 = anchors[:, 2:]
    anchors_yx = anchors_y1x1 / 2. + anchors_y2x2 / 2.
    anchors_hw = anchors_y2x2 - anchors_y1x1
    codes_yx = codes[:, :2]
    codes_hw = codes[:, 2:] / normlization

    grid_y1x1 = anchors_yx - 0.5
    pred_yx = grid_y1x1 + codes_yx
    pred_hw = tf.exp(codes_hw) * anchors_hw
    pred_y1x1 = pred_yx - pred_hw/2.
    pred_y2x2 = pred_yx + pred_hw/2.
    pred = tf.concat([pred_y1x1, pred_y2x2], axis=-1)
    return pred
