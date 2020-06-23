"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from engine.layers import MoldBatch


class ClassLoss(Layer):
    """ Focal Loss For Object Detection
    """
    def __init__(self, weight=1., alpha=.25, gamma=2., **kwargs):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        eps = K.epsilon()
        cls_true = inputs[0]
        cls_pred = inputs[1]
        mask = inputs[2]
        cls_exists = inputs[3]
        batch_size, num_classes = tf.unstack(tf.shape(cls_exists))
        cls_exists = tf.reshape(cls_exists, (batch_size, 1, num_classes))

        # split foreground & background
        neg_mask, pos_mask, ignore_mask = split_neg_pos_mask(mask)
        cls_true = tf.where(tf.not_equal(cls_true, 0),
                            tf.ones_like(cls_true),
                            tf.zeros_like(cls_true))
        num_tot = tf.reduce_sum(pos_mask + neg_mask, axis=(1, 2))
        # Focal Loss
        loss = focal_loss(cls_true, cls_pred, self.gamma, self.alpha)
        loss = loss * tf.cast(cls_exists, tf.float32)
        loss = tf.reduce_sum(ignore_mask * loss, axis=(1, 2)) / (num_tot + eps)
        weight = tf.cast(self.weight,tf.float32)
        return weight * loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight": self.weight,
            "alpha": self.alpha,
            "gamma": self.gamma,
        })
        return config


class BoxLoss(Layer):
    """ Adjust SmoothL1 Loss for Object Detection

    RetinaMask:
    Learning to predict masks improves state-of-the-art single-shot detection for free
    """
    def __init__(self, weight=1., momentum=0.9, beta=.11,
                 use_adjust=False, **kwargs):
        self.momentum = momentum
        self.weight = weight
        self.beta = beta
        self.use_adjust = use_adjust
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.moving_mean = self.add_weight('moving_mean', shape=(4,),
                                           initializer=Constant(self.beta),
                                           trainable=False)
        self.moving_var = self.add_weight('moving_var', shape=(4,),
                                          initializer=Constant(0.),
                                          trainable=False)
        self.built = True

    def call(self, inputs, **kwargs):
        eps = K.epsilon()
        loc_true = inputs[0]
        loc_pred = inputs[1]
        mask = inputs[2]

        # split foreground & background
        neg_mask, pos_mask, ignore_mask = split_neg_pos_mask(mask)
        num_pos = tf.reduce_sum(pos_mask,axis=(1,2))

        if self.use_adjust:
            offsets = tf.abs(loc_true - loc_pred) * pos_mask
            mean = tf.reduce_mean(offsets, axis=(0,1))
            var = tf.reduce_mean((offsets - mean) ** 2, axis=(0,1))
            next_mean = self.moving_mean * self.momentum + mean * (1 - self.momentum)
            next_var = self.moving_var * self.momentum + var * (1 - self.momentum)
            moving_op1 = self.moving_mean.assign(next_mean)
            moving_op2 = self.moving_var.assign(next_var)
            with tf.control_dependencies([moving_op1, moving_op2]):
                beta = tf.clip_by_value(self.moving_mean - self.moving_var, 1e-3, self.beta)
        else:
            beta = self.beta

        # smooth l1 loss
        loss = smooth_l1(loc_true, loc_pred, beta=beta)
        loss = (tf.reduce_sum(tf.squeeze(pos_mask, axis=-1)
                              * loss, axis=1) / (num_pos + eps))
        weight = tf.cast(self.weight, tf.float32)
        return weight * loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "momentum": self.momentum,
            "weight": self.weight,
            "beta": self.beta,
            "use_adjust": self.use_adjust
        })
        return config


class MaskLoss(Layer):
    """ Binary Cross Entropy for Instance Segmentation
    """
    def __init__(self, weight=1., label_smoothing=0, max_batch_size=64, **kwargs):
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.max_batch_size = max_batch_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask_true = inputs[0]
        mask_pred = inputs[1]
        batch_size = tf.shape(mask_pred)[0]
        num_classes = mask_pred.get_shape().as_list()[-1]
        mask_classes = tf.reduce_min(mask_true, axis=(2, 3))
        mask_true = tf.one_hot(mask_true, num_classes + 1, 1, 0)
        mask_true = mask_true[..., :-1]

        transposed_pred = tf.transpose(mask_pred, (4, 0, 1, 2, 3))
        transposed_true = tf.transpose(mask_true, (4, 0, 1, 2, 3))

        mask_indices = tf.where(mask_classes < num_classes)
        mask_indices = tf.cast(mask_indices,tf.int32)
        class_indices = tf.gather_nd(mask_classes, mask_indices)
        target_indices = tf.concat([class_indices[:, None], mask_indices], axis=-1)

        chosen_pred = tf.gather_nd(transposed_pred, target_indices)
        chosen_true = tf.gather_nd(transposed_true, target_indices)
        chosen_true = tf.cast(chosen_true,tf.float32)

        loss = binary_cross_entropy(chosen_true, chosen_pred,
                                    self.label_smoothing)

        molded_loss = MoldBatch(self.max_batch_size)(
            loss, batch_indices=target_indices[:, 1], batch_size=batch_size)
        molded_loss = tf.where(tf.equal(molded_loss, -1.),
                               tf.zeros_like(molded_loss),
                               molded_loss)
        molded_loss = tf.reduce_mean(molded_loss, axis=(2, 3))
        molded_loss = (tf.reduce_sum(molded_loss, axis=1) /
                       tf.cast(tf.count_nonzero(molded_loss, axis=1) + 1, tf.float32))
        weight = tf.cast(self.weight, tf.float32)
        return weight * molded_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight": self.weight,
            "label_smoothing": self.label_smoothing,
            "max_batch_size": self.max_batch_size
        })
        return config


class SegLoss(Layer):
    """ Binary Cross Entropy for Segmentation
    """
    def __init__(self, weight=1., label_smoothing=0., **kwargs):
        self.weight = weight
        self.label_smoothing = label_smoothing
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask_true = inputs[0]
        mask_pred = inputs[1]
        mask_exists = inputs[2]

        mask_exists = tf.cast(mask_exists, tf.float32)

        loss = binary_cross_entropy(mask_true, mask_pred,
                                    self.label_smoothing)
        loss = tf.reduce_mean(loss, axis=(1, 2))
        loss = mask_exists * loss

        loss = tf.reduce_mean(loss, axis=1)
        weight = tf.cast(self.weight, tf.float32)
        return weight * loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight": self.weight,
            "label_smoothing": self.label_smoothing,
        })
        return config


def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    """ Focal loss function

    :param y_true: tf.Tensor (#Batches, #Classes)
    :param y_pred: tf.Tensor (#Batches, #Classes)
    :param gamma: gamma value in focal loss formula
    :param alpha: alpha value in focal loss formula
    :return:
    """
    eps = K.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    pt = tf.where(tf.equal(y_true, 1.),
                  y_pred, 1. - y_pred)
    loss = -tf.pow(1. - pt, gamma) * tf.math.log(pt)
    return alpha * loss


def smooth_l1(y_true, y_pred, beta=0.11):
    """ Generalized Smooth L1 Function

    :param y_true: (#Batches, 4:(cx,cy,w,h))
    :param y_pred: (#Batches, 4:(cx,cy,w,h))
    :param beta: beta value in smooth l1 loss formula
    :return:
    """
    l1_loss = tf.abs(y_true - y_pred) - 0.5 * beta
    l2_loss = 0.5 * (y_true - y_pred) ** 2 / beta
    loss = tf.where(tf.less(l1_loss, beta),
                    l2_loss, l1_loss)
    loss = tf.reduce_mean(loss, axis=-1)
    return loss


def binary_cross_entropy(y_true, y_pred, label_smoothing=0.1):
    """ Generalized binary cross entropy function

    :param y_true:
    :param y_pred:
    :param label_smoothing: Float in [0, 1].
    :return:
    """
    eps = K.epsilon()
    y_true = (1-label_smoothing) * y_true + label_smoothing/2.
    return -(y_true*tf.math.log(y_pred+eps)
             + (1-y_true)*tf.math.log(1-y_pred+eps))


def split_neg_pos_mask(mask):
    """
    Mask의 Value를 바탕으로, Negative Mask, Positive Mask, Ignore Mask를 분리하는 Method
    :param mask:
        - 1. : Negative
        - 0. : Positive
        - -1. : Ignore
    :return:
    """
    neg_mask = tf.where(tf.equal(mask, 1.),
                        tf.ones_like(mask),
                        tf.zeros_like(mask))
    pos_mask = tf.where(tf.equal(mask, 0.),
                        tf.ones_like(mask),
                        tf.zeros_like(mask))
    ignore_mask = tf.where(tf.equal(mask, -1.),
                           tf.zeros_like(mask),
                           tf.ones_like(mask))
    return neg_mask, pos_mask, ignore_mask

