from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from ResNet import resnetv1_50
from fpn_generator import fpn_generator
from anchor_generator import anchor_generator
from bboxfs import partition_pos_neg_samples, ohem, bbox_decode, bbox_encode
from lossfs import smooth_l1_loss
from roi_align import roi_align
from VGG import vgg16
layers = tf.keras.layers


class Faster_RCNN(tf.keras.Model):
    def __init__(self, config):
        super(Faster_RCNN, self).__init__()
        self.num_classes = config['num_classes']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if self.mode == 'train' else 1
        input_img = layers.Input(shape=[None, None, 3],
                                 batch_size=self.batch_size, dtype=tf.float32)
        self.backone = vgg16(input_img, conv_trainable=config['backone_conv_trainable'],
                                    weight=config['backone_weight'])
        if self.mode == 'train':
            gt = layers.Input(shape=[None, 5], batch_size=self.batch_size, dtype=tf.float32)
            self.forward = self.build_graph(input_img, gt)
        else:
            self.forward = self.build_graph(input_img)

    def build_graph(self, input_img, gt=None):
        num_fpn_layers = 5
        endpoints = self.backone(input_img)
        fpn_feat = fpn_generator(endpoints, channels_output=256, nums_output=num_fpn_layers, mode='resize')
        dw_rate = [4., 8., 16., 32., 64.]
        anchors = [
            [[8., 8.], [8./2., 8.*2.], [8.*2., 8./2.]],

            [[8., 8.], [8./2., 8.*2.], [8.*2., 8./2.]],

            [[8., 8.], [8./2., 8.*2.], [8.*2., 8./2.]],

            [[8., 8.], [8./2., 8.*2.], [8.*2., 8./2.]],

            [[8., 8.], [8./2., 8.*2.], [8.*2., 8./2.]],
        ]
        anchors_all = anchor_generator(
            fpn_feat, anchors, [1.]*len(dw_rate), flatten=True
        )
        out1 = self.rpn_head(fpn_feat[0], len(anchors[0]), 'rpn_head', flatten=True)
        out2 = self.rpn_head(fpn_feat[1], len(anchors[1]), 'rpn_head', flatten=True)
        out3 = self.rpn_head(fpn_feat[2], len(anchors[2]), 'rpn_head', flatten=True)
        out4 = self.rpn_head(fpn_feat[3], len(anchors[3]), 'rpn_head', flatten=True)
        out5 = self.rpn_head(fpn_feat[4], len(anchors[4]), 'rpn_head', flatten=True)
        out = [out1, out2, out3, out4, out5]
        rpn_cla = [x[..., :2] for x in out]
        rpn_reg = [x[..., 2:] for x in out]

        if self.mode == 'train':
            rpn_loss = []
            rcnn_loss = []
            for i in range(self.batch_size):
                rpn_l, proposal, labels, bbox_gt = self.rpn_match(anchors_all,
                                                        [x[i, ...] for x in rpn_cla],
                                                        [x[i, ...] for x in rpn_reg],
                                                        gt[i, ...],
                                                        dw_rate
                                                        )
                rpn_loss.append(rpn_l)
                fpn_feat_i = [x[0] for x in fpn_feat]
                roi_feat = roi_align(fpn_feat_i, proposal, [[14, 14]] * len(fpn_feat))
                proposal = tf.concat(proposal, axis=0)
                roi_feat = tf.concat(roi_feat, axis=0)
                roi_feat = tf.reshape(roi_feat, [-1, 14, 14, 256])
                rcnn_cla, rcnn_reg = self.rcnn_head(roi_feat, name='rcnn_head')
                rcnn_reg_target = bbox_encode(proposal, bbox_gt)
                rcnn_reg_loss = tf.reduce_sum(rcnn_reg_target - rcnn_reg)
                rcnn_cla_loss = tf.reduce_sum(
                    tf.losses.sparse_categorical_crossentropy(labels, rcnn_cla, from_logits=True))
                num_pos = tf.cast(tf.shape(labels)[0], tf.float32)
                rcnn_loss.append((rcnn_reg_loss + rcnn_cla_loss) / (num_pos + 1.))
            rpn_loss = tf.reduce_sum(rpn_loss)
            total_loss = (rpn_loss + rcnn_loss) / tf.cast(self.batch_size, tf.float32)
            model = tf.keras.Model(inputs=[input_img, gt], outputs=total_loss, name='faster_rcnn')
            return model
        else:
            scores_threshold = 0.5
            iou_threshold = 0.45
            max_outputs = 100
            rpn_cla = [tf.nn.softmax(x[0]) for x in rpn_cla]
            rpn_reg = [x[0] for x in rpn_reg]
            rpn_pos_mask = [x[:, 1] > 0.5 for x in rpn_cla]
            rpn_reg = [tf.boolean_mask(x, y) for x, y in zip(rpn_reg, rpn_pos_mask)]
            anchors = [tf.boolean_mask(x, y) for x, y in zip(anchors_all, rpn_pos_mask)]
            proposals = [bbox_decode(x, y, normlization=[10., 10., 5., 5.]) for x, y in zip(anchors, rpn_reg)]
            fpn_feat = [x[0] for x in fpn_feat]
            roi_feat = roi_align(fpn_feat, proposals, [[14, 14]]*len(fpn_feat))
            roi_feat = tf.concat(roi_feat, axis=0)
            roi_feat = tf.reshape(roi_feat, [-1, 14, 14, 256])
            proposals = tf.concat([x*y for x, y in zip(proposals, dw_rate)], axis=0)
            rcnn_cla, rcnn_reg = self.rcnn_head(roi_feat, name='rcnn_head')
            pred_bboxes = bbox_decode(proposals, rcnn_reg)
            rcnn_cla = tf.sigmoid(rcnn_cla)
            filter_mask = rcnn_cla > scores_threshold
            scores = []
            bbox = []
            class_id = []
            for i in range(self.num_classes):
                scoresi = tf.boolean_mask(rcnn_cla[:, i], filter_mask[:, i])
                bboxi = tf.boolean_mask(pred_bboxes, filter_mask[:, i])
                selected_indices = tf.image.non_max_suppression(
                    bboxi, scoresi, max_outputs, iou_threshold=iou_threshold,
                )
                scores.append(tf.gather(scoresi, selected_indices))
                bbox.append(tf.gather(bboxi, selected_indices))
                class_id.append(tf.ones_like(tf.gather(scoresi, selected_indices), tf.int32) * i)
            bbox = tf.concat(bbox, axis=0)
            scores = tf.concat(scores, axis=0)
            class_id = tf.concat(class_id, axis=0)
            pred = [class_id, scores, bbox]
            model = tf.keras.Model(inputs=input_img, outputs=pred, name='faster_rcnn')
            return model

    def rpn_match(self, anchors, cla, reg, gt, dw_rate):
        slice_index = tf.argmin(gt, axis=0)[0]
        gt = tf.gather(gt, tf.range(0, slice_index, dtype=tf.int64))
        gt_bboxes = [gt[:, :4]/r for r in dw_rate]
        gt_labels = [tf.cast(gt[:, 4], tf.int32)] * len(dw_rate)
        pos_threshold = [0.7] * len(dw_rate)
        neg_threshold = [0.3] * len(dw_rate)

        pos_gt_bboxes, pos_anchors, neg_anchors, pos_labels, reg, pos_cla, neg_cla = \
            partition_pos_neg_samples(gt_bboxes, anchors, gt_labels, pos_threshold, neg_threshold, reg, cla)
        neg_scores = [1. - x[:, 0] for x in neg_cla]
        num_pos = [tf.shape(x)[0] for x in pos_labels]
        chosen_neg = [3 * x for x in num_pos]
        ohem_threshold = [0.7] * len(chosen_neg)
        neg_cla = ohem(neg_anchors, neg_scores, max_output=chosen_neg, iou_threshold=ohem_threshold, pred=neg_cla)
        pos_cla = tf.concat(pos_cla, axis=0)
        neg_cla = tf.concat(neg_cla, axis=0)
        proposal = [bbox_decode(x, y, normlization=[10., 10., 5., 5.]) for x, y in zip(pos_anchors, reg)]
        proposal_labels = tf.concat(pos_labels, axis=0)
        reg = tf.concat(reg, axis=0)
        pos_gt_bboxes = tf.concat(pos_gt_bboxes, axis=0)
        pos_anchors = tf.concat(pos_anchors, axis=0)
        reg_target = bbox_encode(pos_gt_bboxes, pos_anchors, normlization=[10., 10., 5., 5.])

        reg_loss = smooth_l1_loss(reg_target - reg)
        pos_cla_loss = tf.losses.sparse_categorical_crossentropy(tf.zeros_like(pos_cla[:, 0]), pos_cla, from_logits=True)
        neg_cla_loss = tf.losses.sparse_categorical_crossentropy(tf.zeros_like(neg_cla[:, 0]), neg_cla, from_logits=True)
        loss = tf.reduce_sum(reg_loss) + tf.reduce_sum(pos_cla_loss) + tf.reduce_sum(neg_cla_loss)
        loss = loss / tf.cast(tf.reduce_sum(num_pos) + 1, tf.float32)

        return loss, proposal, proposal_labels, pos_gt_bboxes

    def call(self, inputs):
            return self.forward(inputs)

    def rpn_head(self, inputs, num_anchors, name, flatten=False):
        out_channels = (2 + 4) * num_anchors
        with tf.compat.v1.variable_scope(name, reuse=True):
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name+'_conv1')(inputs)
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name+'_conv2')(conv)
            out = layers.Conv2D(out_channels, 3, 1, 'same', name=name+'output')(conv)
            if flatten is True:
                batch_size = tf.shape(out)[0]
                out = tf.reshape(out, [batch_size, -1, num_anchors, 2+4])
                out = tf.reshape(out, [batch_size, -1, 2+4])
            return out

    def rcnn_head(self, inputs, name):
        with tf.compat.v1.variable_scope(name, reuse=True):
            conv = layers.Conv2D(128, 1, 1, 'same', activation='relu', name=name + '_conv1')(inputs)
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name + '_conv2')(conv)
            conv = layers.Conv2D(128, 1, 1, 'same', activation='relu', name=name + '_conv3')(conv)
            flatten = layers.Flatten()(conv)
            cla = layers.Dense(self.num_classes)(flatten)
            reg = layers.Dense(4)(flatten)
            return cla, reg

