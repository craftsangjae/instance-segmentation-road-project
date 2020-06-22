"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from engine.layers.detection import CalculateIOU


class ConfusionMatrixMetric(Layer):
    def __init__(self, threshold=0.3, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs, **kwargs):
        eps = K.epsilon()
        cls_true = inputs[0]
        cls_pred = inputs[1]
        mask = inputs[2]

        num_classes = tf.shape(cls_pred)[2]
        cls_true = tf.reshape(cls_true, [-1, num_classes])
        mask = tf.reshape(mask, [-1, ])
        cls_pred = tf.reshape(cls_pred, [-1, num_classes])

        pos_mask = tf.where(tf.equal(mask, 0.),
                            tf.ones_like(mask),
                            tf.zeros_like(mask))
        ignore_mask = tf.where(tf.equal(mask, -1.),
                               tf.zeros_like(mask,tf.float32),
                               tf.ones_like(mask,tf.float32))

        num_classes = tf.cast(tf.shape(cls_pred)[1], dtype=tf.int64)
        y_true_idx = tf.argmax(cls_true, axis=1)
        y_true_idx = tf.where(tf.greater(pos_mask, 0),
                              y_true_idx,
                              (tf.ones_like(y_true_idx) * num_classes))
        y_pred_idx = tf.argmax(cls_pred, axis=1)
        y_pred_idx = tf.where(tf.greater(
            tf.reduce_max(cls_pred, axis=1), self.threshold),
            y_pred_idx,
            (tf.ones_like(y_pred_idx) * num_classes))

        true_mask = tf.equal(y_true_idx, y_pred_idx)
        false_mask = tf.logical_not(true_mask)
        pos_mask = tf.less(y_pred_idx, num_classes)
        neg_mask = tf.logical_not(pos_mask)

        # Drop Ignore Mask
        tp = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pos_mask), tf.float32) * ignore_mask)
        fp = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, pos_mask), tf.float32) * ignore_mask)
        fn = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, neg_mask), tf.float32) * ignore_mask)
        tn = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, neg_mask), tf.float32) * ignore_mask)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        fmeasure = 2 * (precision * recall) / (precision + recall + eps)
        return precision, recall, accuracy, fmeasure

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold
        })
        return config


class ClassBinaryIOU(Layer):
    """
    Build an Intersection over Union (IoU) metric for binary classification per class.

    """
    def __init__(self, threshold=0.5, **kwargs):
        """

        :param threshold: the threshold for binary classification
        """
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs, **kwargs):
        seg_true = inputs[0]
        seg_pred = inputs[1]

        seg_true = tf.cast(seg_true > self.threshold,tf.float32)
        seg_pred = tf.cast(seg_pred > self.threshold,tf.float32)

        # calculate the Insersection of the labels
        intersection = tf.reduce_sum(seg_true * seg_pred, axis=(1, 2))

        # calculate the Union of the labels
        area_true = tf.reduce_sum(seg_true, axis=(1, 2))
        area_pred = tf.reduce_sum(seg_pred, axis=(1, 2))
        union = area_true + area_pred - intersection

        iou = tf.where(union > 0., intersection / union, tf.ones_like(union))
        return tf.unstack(iou, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold
        })
        return config


class DetectionIOUMetric(Layer):
    """
    inputs:
        - pred_boxes : [batch, num boxes, 6(cx, cy, w, h, class id, class confidence)]
        - gt_boxes : [batch, num boxes, 6(cx, cy, w, h, class id, class confidence)]
    Outputs:

    """

    def call(self, inputs, **kwargs):
        eps = K.epsilon()
        proposed_boxes = inputs[0]
        gt_boxes = inputs[1]

        batch_size, num_proposed, _ = tf.unstack(tf.shape(proposed_boxes))
        batch_size, num_gt, _ = tf.unstack(tf.shape(gt_boxes))

        b_proposed = tf.transpose(
            tf.tile(tf.range(0, batch_size)[None], [num_proposed, 1]))
        b_proposed = tf.reshape(b_proposed, (-1,))

        b_gt = tf.transpose(
            tf.tile(tf.range(0, batch_size)[None], [num_gt, 1]))
        b_gt = tf.reshape(b_gt, (-1,))
        b_mask = tf.cast(tf.equal(b_proposed[:, None], b_gt[None, :]),tf.float32)

        ignore_proposed = tf.reshape(proposed_boxes[..., 0], (-1,))
        ignore_proposed = tf.not_equal(ignore_proposed, -1)
        ignore_gt = tf.reshape(gt_boxes[..., 0], (-1,))
        ignore_gt = tf.not_equal(ignore_gt, -1)
        ignore_mask = tf.cast(
            tf.logical_or(ignore_proposed[:, None], ignore_gt[None]),tf.float32)

        iou_mask = b_mask * ignore_mask

        iou = CalculateIOU()([tf.reshape(proposed_boxes[..., :4], (-1, 4)),
                              tf.reshape(gt_boxes[..., :4], (-1, 4))])
        iou = iou * iou_mask
        iou = tf.reshape(iou, (batch_size, num_proposed, batch_size, num_gt))
        iou = tf.transpose(iou, (0, 2, 1, 3))
        iou = tf.gather_nd(iou, tf.stack([tf.range(0, batch_size)] * 2, axis=1))

        num_pos = tf.reduce_sum(
            tf.cast(tf.reduce_max(iou, axis=2) > 0.5,tf.float32), axis=1)
        num_true = tf.reduce_sum(
            tf.cast(tf.reduce_max(iou, axis=1) > 0.5,tf.float32), axis=1)

        num_pred = tf.reduce_sum(
            tf.cast(tf.not_equal(proposed_boxes[..., 0], -1.),tf.float32), axis=1)
        num_gt = tf.reduce_sum(
            tf.cast(tf.not_equal(gt_boxes[..., 0], -1.),tf.float32), axis=1)

        precision = num_pos / (num_pred + eps)
        recall = num_true / (num_gt + eps)
        fmeasure = 2 * (precision * recall) / (precision + recall + eps)

        return precision, recall, fmeasure


__all__ = [
    "ConfusionMatrixMetric",
    "ClassBinaryIOU",
    "DetectionIOUMetric"]

