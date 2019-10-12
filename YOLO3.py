from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from ResNet import resnetv1_50
from fpn_generator import fpn_generator
from anchor_generator import anchor_generator
from bboxfs import partition_pos_neg_samples_yolo, bbox_decode_yolo, bbox_encode_yolo

layers = tf.keras.layers


class YOLO3(tf.keras.Model):
    def __init__(self, config):
        super(YOLO3, self).__init__()
        self.num_classes = config['num_classes']
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
        num_fpn_layers = 3
        endpoints = self.backone(input_img)
        fpn_feat = fpn_generator(endpoints[1:], channels_output=256, nums_output=num_fpn_layers, mode='dconv')
        dw_rate = [8., 16., 32.]
        anchors = [
            [[10./8., 13./8.], [16./8., 30./8.], [33./8, 23./8.]],

            [[30./16., 61./16.], [62./16., 45./16.], [59./16., 119./16.]],

            [[116./32., 90./32.], [156./32., 198./32.], [373./32., 326./32.]],
        ]

        flatten = False if self.mode == 'train' else True
        if self.mode == 'train':
            anchors_all = anchor_generator(
                fpn_feat, anchors, [1.] * len(dw_rate), flatten=flatten
            )
        else:
            anchors_all = anchor_generator(
                fpn_feat, anchors, dw_rate, flatten=True
            )

        out1 = self.head(fpn_feat[0], len(anchors[0]), 'head', flatten=flatten)
        out2 = self.head(fpn_feat[1], len(anchors[1]), 'head', flatten=flatten)
        out3 = self.head(fpn_feat[2], len(anchors[2]), 'head', flatten=flatten)
        out = [out1, out2, out3]
        cla_reg = [x[..., :self.num_classes+4] for x in out]
        obj = [x[..., self.num_classes+4:] for x in out]

        if self.mode == 'train':
            total_loss = []
            for it in range(self.batch_size):
                total_loss.append(
                    self.cal_one_image_loss(anchors_all,
                                            [x[it, ...] for x in cla_reg],
                                            [x[it, ...] for x in obj],
                                            gt[it, ...],
                                            dw_rate
                                            )
                )
            total_loss = tf.reduce_mean(total_loss)
            model = tf.keras.Model(inputs=[input_img, gt], outputs=total_loss, name='yolo3')
            return model
        else:
            cla = []
            reg = []
            score_threshold = 0.5
            iou_threshold = 0.45
            max_outputs = 100
            cla_reg = [x[0] for x in cla_reg]
            obj = [x[0] for x in obj]
            for i in range(num_fpn_layers):
                c = cla_reg[i][:, :self.num_classes] * obj[i]
                r_yx = tf.sigmoid(cla_reg[i][:, self.num_classes:self.num_classes+2])
                r_hw = cla_reg[i][:, self.num_classes+2:]
                r = tf.concat([r_yx, r_hw], axis=-1)
                cla.append(c)
                reg.append(r)
            cla = tf.concat(cla, axis=0)
            reg = tf.concat(reg, axis=0)
            anchors = tf.concat(anchors_all, axis=0)
            pred_bboxes = bbox_decode_yolo(anchors, reg)
            filter_mask = cla > score_threshold

            scores = []
            bbox = []
            class_id = []
            for i in range(self.num_classes):
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
            model = tf.keras.Model(inputs=input_img, outputs=pred, name='yolo3')
            return model

    def cal_one_image_loss(self, anchors,cla_reg, obj, gt, dw_rate):
        slice_index = tf.argmin(gt, axis=0)[0]
        gt = tf.gather(gt, tf.range(0, slice_index, dtype=tf.int64))
        gt_bboxes = [gt[:, :4] / r for r in dw_rate]
        gt_labels = tf.cast(gt[:, 4], tf.int32)
        threshold = 0.5

        pos_gt_bboxes, pos_anchors, pos_labels, pos_cla_reg, pos_obj, neg_obj = \
            partition_pos_neg_samples_yolo(gt_bboxes, anchors, gt_labels, threshold, cla_reg, obj)
        num_pos = tf.shape(pos_labels)[0]
        pos_obj = tf.sigmoid(pos_obj)
        neg_obj = tf.sigmoid(neg_obj)
        pos_cla = tf.sigmoid(pos_cla_reg[:, :self.num_classes])
        pos_reg_yx = tf.sigmoid(pos_cla_reg[:, self.num_classes:self.num_classes+2])
        pos_reg_hw = pos_cla_reg[:, self.num_classes+2:]

        reg_target = bbox_encode_yolo(pos_gt_bboxes, pos_anchors)
        reg_yx_target = reg_target[:, :2]
        reg_hw_target = reg_target[:, 2:]
        pos_cla_loss = tf.losses.sparse_categorical_crossentropy(gt_labels, pos_cla, from_logits=False)
        reg_yx_loss = tf.losses.categorical_crossentropy(reg_yx_target, pos_reg_yx, from_logits=False)
        reg_hw_loss = tf.square(pos_reg_hw - reg_hw_target)
        pos_obj_loss = tf.losses.categorical_crossentropy(tf.ones_like(pos_obj), pos_obj, from_logits=False)
        neg_obj_loss = tf.losses.categorical_crossentropy(tf.zeros_like(neg_obj), neg_obj, from_logits=False)
        loss = tf.reduce_sum(pos_cla_loss) + tf.reduce_sum(reg_yx_loss) + tf.reduce_sum(reg_hw_loss) + \
            tf.reduce_sum(pos_obj_loss) + tf.reduce_sum(neg_obj_loss)
        loss = loss / tf.cast(tf.reduce_sum(num_pos)+1, tf.float32)
        return loss

    def call(self, inputs):
            return self.forward(inputs)

    def head(self, input, num_anchors, name, flatten=False):
        out_channels = (self.num_classes + 5) * num_anchors
        with tf.compat.v1.variable_scope(name, reuse=True):
            conv = layers.Conv2D(128, 1, 1, 'same', activation='relu', name=name+'_conv1')(input)
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name+'_conv2')(conv)
            conv = layers.Conv2D(128, 1, 1, 'same', activation='relu', name=name+'_conv3')(conv)
            conv = layers.Conv2D(256, 3, 1, 'same', activation='relu', name=name + '_conv4')(conv)
            out = layers.Conv2D(out_channels, 3, 1, 'same', name=name+'output')(conv)
            batch_size = tf.shape(out)[0]
            out = tf.reshape(out, [batch_size, -1, num_anchors, self.num_classes + 5])
            if flatten is True:
                out = tf.reshape(out, [batch_size, -1, self.num_classes+5])
            return out