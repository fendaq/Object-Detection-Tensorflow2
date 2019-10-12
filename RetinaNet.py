from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from ResNet import resnetv1_50
from fpn_generator import fpn_generator
from anchor_generator import anchor_generator
from bboxfs import partition_pos_neg_samples, bbox_decode, bbox_encode
from lossfs import smooth_l1_loss

layers = tf.keras.layers


class RetinaNet(tf.keras.Model):
    def __init__(self, config):
        super(RetinaNet, self).__init__()
        self.num_classes = config['num_classes'] + 1
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if self.mode == 'train' else 1
        input_img = layers.Input(shape=[None, None, 3], batch_size=self.batch_size, dtype=tf.float32)
        self.backone = resnetv1_50(input_img, conv_trainable=config['backone_conv_trainable'],
                                   bn_trainable=config['backone_bn_trainable'], weight=config['backone_weight'])
        if self.mode == 'train':
            gt = layers.Input(shape=[None, 5], batch_size=self.batch_size, dtype=tf.float32)
            self.forward = self.build_graph(input_img, gt)
        else:
            self.forward = self.build_graph(input_img)

    def build_graph(self, input_img, gt=None):
        num_fpn_layers = 5
        endpoints = self.backone(input_img)
        fpn_feat = fpn_generator(endpoints[1:], channels_output=256, nums_output=num_fpn_layers, mode='dconv')
        dw_rate = [8., 16., 32., 64., 128.]
        anchors = [
            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],

            [[4, 4], [4. * 2., 4. / 2.], [4. / 2., 4. * 2.], [4. * 3, 4. / 3.], [4. / 3., 4. * 3.]],
        ]
        anchors_all = anchor_generator(
            fpn_feat, anchors, dw_rate, flatten=True
        )
        out1 = self.head(fpn_feat[0], len(anchors[0]), 'head', flatten=True)
        out2 = self.head(fpn_feat[1], len(anchors[1]), 'head', flatten=True)
        out3 = self.head(fpn_feat[2], len(anchors[2]), 'head', flatten=True)
        out4 = self.head(fpn_feat[3], len(anchors[3]), 'head', flatten=True)
        out5 = self.head(fpn_feat[4], len(anchors[4]), 'head', flatten=True)
        out = [out1, out2, out3, out4, out5]
        cla = [x[..., :self.num_classes] for x in out]
        reg = [x[..., self.num_classes:] for x in out]

        if self.mode == 'train':
            total_loss = []
            for it in range(self.batch_size):
                total_loss.append(
                    self.cal_one_image_loss(anchors_all,
                                            [x[it, ...] for x in cla],
                                            [x[it, ...] for x in reg],
                                            gt[it, ...],
                                            )
                )
            total_loss = tf.reduce_mean(total_loss)
            model = tf.keras.Model(inputs=[input_img, gt], outputs=total_loss, name='retinanet')
            return model
        else:
            scores_threshold = 0.5
            iou_threshold = 0.45
            max_outputs = 100
            cla = [tf.nn.softmax(x[0]) for x in cla]
            reg = [x[0] for x in reg]
            cla = tf.concat(cla, axis=0)
            reg = tf.concat(reg, axis=0)
            anchors = tf.concat(anchors_all, axis=0)
            pred_bboxes = bbox_decode(anchors, reg, normlization=[10., 10., 5., 5.])
            filter_mask = cla > scores_threshold
            scores = []
            bbox = []
            class_id = []
            for i in range(1, self.num_classes):
                scoresi = tf.boolean_mask(cla[:, i], filter_mask[:, i])
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
            model = tf.keras.Model(inputs=input_img, outputs=pred, name='retinanet')
            return model

    def cal_one_image_loss(self, anchors, cla, reg, gt):
        slice_index = tf.argmin(gt, axis=0)[0]
        gt = tf.gather(gt, tf.range(0, slice_index, dtype=tf.int64))
        gt_bboxes = [gt[:, :4]] * len(anchors)
        gt_labels = [tf.cast(gt[:, 4], tf.int32)] * len(anchors)
        pos_threshold = [0.5] * len(anchors)
        neg_threshold = [0.4] * len(anchors)
        pos_gt_bboxes, pos_anchors, neg_anchors, pos_labels, reg, pos_cla, neg_cla = \
            partition_pos_neg_samples(gt_bboxes, anchors, gt_labels, pos_threshold, neg_threshold, reg, cla)

        pos_cla = tf.concat(pos_cla, axis=0)
        neg_cla = tf.concat(neg_cla, axis=0)
        reg = tf.concat(reg, axis=0)
        pos_labels = tf.concat(pos_labels, axis=0)
        pos_gt_bboxes = tf.concat(pos_gt_bboxes, axis=0)
        pos_anchors = tf.concat(pos_anchors, axis=0)
        reg_target = bbox_encode(pos_gt_bboxes, pos_anchors, normlization=[10., 10., 5., 5.])
        reg_loss = smooth_l1_loss(reg_target-reg)

        num_pos = tf.shape(pos_labels)[0]
        pos_cla_loss = tf.losses.sparse_categorical_crossentropy(pos_labels, pos_cla, from_logits=True)
        neg_cla_loss = tf.losses.sparse_categorical_crossentropy(tf.zeros_like(neg_cla[:, 0]), neg_cla, from_logits=True)
        pos_inds = tf.concat([tf.expand_dims(tf.range(num_pos), axis=-1), tf.expand_dims(pos_labels, axis=-1)], axis=-1)
        pos_cla_loss *= tf.pow(tf.gather_nd(pos_cla, pos_inds), 2.) * 0.25
        neg_cla_loss *= tf.pow(neg_cla[:, 0], 2.) * 0.25
        loss = tf.reduce_sum(reg_loss) + tf.reduce_sum(pos_cla_loss) + tf.reduce_sum(neg_cla_loss)
        loss = loss / tf.cast(num_pos+1, tf.float32)
        return loss

    def call(self, inputs):
            return self.forward(inputs)

    def head(self, input, num_anchors, name, flatten=False):
        out_channels = (self.num_classes + 4) * num_anchors
        with tf.compat.v1.variable_scope(name, reuse=True):
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name+'_conv1')(input)
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name+'_conv2')(conv)
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name+'_conv3')(conv)
            out = layers.Conv2D(out_channels, 3, 1, 'same', name=name+'output')(conv)
            if flatten is True:
                batch_size = tf.shape(out)[0]
                out = tf.reshape(out, [batch_size, -1, num_anchors, self.num_classes+4])
                out = tf.reshape(out, [batch_size, -1, self.num_classes+4])
            return out