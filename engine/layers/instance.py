"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from engine.layers.misc import SqueezeExcite, MobileSeparableConv2D, MoldBatch
from engine.layers.detection import CalculateIOU
from engine.layers.detection import NormalizeBoxes
from engine.normalization import GroupNormalization
from tensorflow.python.keras.layers import Layer, Concatenate
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras import backend as K
import tensorflow as tf


"""
Refinement Network For Instance Segmentation

Detection Network의 출력값을 통해, Mask Prediction Network의 Input 값을 구성

MaskDistribute
    사물의 크기에 따라 Feature Map을 배치하는 Layer
    사물의 크기가 작은 경우, low level feature map에서 사물 정보가 가져오는 것이 유리하고,
    사물의 크기가 큰 경우, high level feature map에서 사물 정보가 가져오는 것이 유리하다.

PyramidRoiAlign
    MaskDistribute에서 배치받은 Feature Map에서 사물의 크기만큼 Feature Map에서 Crop하여
    가져오는 Layer
"""


class MaskDistribute(Layer):
    """ Object의 크기에 따라, Roi align에서 추출할 Feature Map을 결정하는 Layer

    Confidence Value가 0이하인 경우는 제외하고 각 Object 별로, 어느 Feature Map에서 가져올 지를 결정

    inputs:
        - proposed_boxes :
            Non-Maximum Suppression을 거쳐 추려진 Object Detection 정보
            > shape : [batch size, num boxes, 6(cx, cy, w, h, class id, class confidence)]
    Outputs:
        - dist_boxes :
            각 proposed box 별로 배치된 feature map의 index 정보가 추가된 Object Detection 정보
            > shape : [batch size, num boxes, 7(fmap id, cx, cy, w, h, class id, class confidence)]
    """

    def __init__(self, max_k=2, base_size=64, **kwargs):
        self.max_k = max_k
        self.base_size = base_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        eps = K.epsilon()
        boxes = inputs[..., :4]

        H, W = boxes[..., 2], boxes[..., 3]
        size = tf.sqrt(H * W)
        delta_k = tf.math.log((size + eps) / (self.base_size + eps)) / tf.math.log(2.)
        k = tf.floor(delta_k)
        k = tf.clip_by_value(k, 0, self.max_k)
        k = tf.where(tf.equal(inputs[..., 0], -1.),
                     inputs[..., 0], k)

        dist_boxes = tf.concat([k[..., None],
                                inputs], axis=-1)
        return dist_boxes

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_k": self.max_k,
            "base_size": self.base_size,
        })
        return config


class PyramidRoiAlign(Layer):
    """ 여러 Feature Map 중에서 해당 박스의 크기에 맞는 Feature Map에서 Roi Align을 수행하여 Crop하는 Layer

    inputs:
        - fmap_outputs :
            the list of the tensor,
            > shape : num fmap x [batch size, fmap height, fmap width, channel]
        - dist_boxes :
            각 proposed box 별로 배치된 feature map의 index 정보가 추가된 Object Detection 정보
            > shape : [batch size, num boxes, 7(fmap id, cx, cy, w, h, class id, class confidence)]
        - images :
            입력 이미지, 좌표를 [0,1]로 Normalization하는 데에 쓰임
            > shape : [batch size, height, width, channel]

    outputs:
        - roi_fmaps :
            dist_boxes에 맞춰 각 Feature map에서 crop한 후 (crop height, crop width) 크기로 resize한 정보
            > shape : num fmap x [num boxes per fmap, crop height, crop width, channel]

        - roi_boxes :
            roi_fmaps의 순서에 맞춰 dist boxes를 재정렬
            > shape : [batch size, num boxes, 7(batch id, cx, cy, w, h, class id, class confidence)]
            ( c.f) num boxes
                    = num boxes of 1st fmap + num boxes of 2nd fmap ... + num boxes of nth fmap)

    """

    def __init__(self, crop_size=(14, 14), max_batch_size=64, **kwargs):
        self.crop_size = crop_size
        self.max_batch_size = max_batch_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        fmap_outputs = inputs[0]
        dist_boxes = inputs[1]
        images = inputs[2]

        batch_size, num_boxes, _ = tf.unstack(tf.shape(dist_boxes))
        norm_boxes = NormalizeBoxes()(dist_boxes[..., 1:5],
                                      shape=tf.shape(images)[1:3])

        roi_fmaps = []
        roi_boxes = []
        for fmap_id, target_fmap in enumerate(fmap_outputs):
            idx = tf.where(tf.equal(dist_boxes[..., 0], fmap_id))
            target_norm_boxes = tf.gather_nd(norm_boxes, idx)

            box_indices = tf.cast(idx[:, 0], tf.int32)
            target_outputs = tf.image.crop_and_resize(target_fmap, target_norm_boxes,
                                                      box_indices, self.crop_size)
            target_outputs = MoldBatch(self.max_batch_size)(
                target_outputs, batch_indices=box_indices, batch_size=batch_size)
            roi_fmaps.append(target_outputs)

            target_boxes = tf.gather_nd(dist_boxes[..., 1:], idx)
            target_boxes = MoldBatch(self.max_batch_size)(
                target_boxes, batch_indices=box_indices, batch_size=batch_size)
            roi_boxes.append(target_boxes)
        if len(fmap_outputs) > 1:
            roi_boxes = tf.concat(roi_boxes, axis=1)
        else:
            roi_boxes = roi_boxes[0]
        return [roi_fmaps, roi_boxes]

    def get_config(self):
        config = super().get_config()
        config.update({
            "crop_size": self.crop_size,
            "max_batch_size": self.max_batch_size
        })
        return config


"""
Head Networks For Instance Segmentation

주어진 num blocks에 맞춰 Mask Prediction Networks가 구성되어 있음. 

"""


class MaskSubNet(Layer):
    """ Mask Prediction Module In RetinaMask
    """

    def __init__(self, num_blocks, num_classes, num_depth=4, num_features=256,
                 use_separable_conv=False, expand_ratio=4.,
                 use_squeeze_excite=False, squeeze_ratio=16.,
                 groups=16, **kwargs):
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_depth = num_depth
        self.num_features = num_features
        self.use_separable_conv = use_separable_conv
        self.expand_ratio = expand_ratio
        self.use_squeeze_excite = use_squeeze_excite
        self.squeeze_ratio = squeeze_ratio
        self.groups = groups
        super().__init__(**kwargs)
        self.blocks = []
        for idx in range(self.num_blocks):
            block = []
            for i in range(self.num_depth):
                if self.use_squeeze_excite:
                    layer = SqueezeExcite(self.squeeze_ratio)
                    block.append(layer)

                if self.use_separable_conv:
                    layer = MobileSeparableConv2D(num_features, (3, 3),
                                                  expand_ratio=expand_ratio)
                else:
                    layer = Conv2D(num_features, (3, 3), activation='relu', padding='same',
                                   kernel_initializer=RandomNormal(stddev=0.01))
                block.append(layer)

                layer = GroupNormalization(self.groups)
                block.append(layer)

            layer = Conv2DTranspose(num_features, (2, 2), (2, 2), padding='same', activation='relu',
                                    kernel_initializer=RandomNormal(stddev=0.01))
            block.append(layer)
            layer = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid',
                           kernel_initializer=RandomNormal(stddev=0.01))
            block.append(layer)
            self.blocks.append(block)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        heads = []
        for idx, head in enumerate(inputs):
            src_dynamic_shape = tf.shape(head)
            src_static_shape = head.get_shape().as_list()

            head = tf.reshape(head, (src_dynamic_shape[0]*src_dynamic_shape[1],
                                     src_static_shape[2],src_static_shape[3],
                                     src_static_shape[4]))
            block = self.blocks[idx]
            for layer in block:
                head = layer(head)
            dst_static_shape = head.get_shape().as_list()
            head = tf.reshape(head, (src_dynamic_shape[0], src_dynamic_shape[1],
                                     dst_static_shape[1], dst_static_shape[2],
                                     dst_static_shape[3]))
            heads.append(head)
        if len(heads)>1:
            return Concatenate(axis=1)(heads)
        else:
            return heads[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_blocks": self.num_blocks,
            "num_classes": self.num_classes,
            "num_depth": self.num_depth,
            "num_features": self.num_features,
            "use_separable_conv": self.use_separable_conv,
            "expand_ratio": self.expand_ratio,
            "use_squeeze_excite": self.use_squeeze_excite,
            "squeeze_ratio": self.squeeze_ratio,
            "groups": self.groups
        })
        return config


"""
Post Process For Instance Segmentation

불필요한 Padding을 추려내어주는 역할
"""


class TrimInstances(Layer):
    """ Instance Segmentation의 출력값 중에서 불필요한 Padding들을 제거해주는 역할
    """
    def __init__(self, mold=True, max_batch_size=64, **kwargs):
        self.mold = mold
        self.max_batch_size = max_batch_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        roi_boxes = inputs[0]
        roi_masks = inputs[1]

        batch_size = tf.shape(roi_boxes)[0]
        indices = tf.where(
            tf.not_equal(roi_boxes[:, :, -2], -1))
        class_indices = tf.cast(tf.gather_nd(roi_boxes[:, :, -2], indices),
                                tf.int64)[:, None]
        indices = tf.concat([indices, class_indices], axis=1)
        target_masks = tf.gather_nd(tf.transpose(roi_masks, (0, 1, 4, 2, 3)), indices)
        target_boxes = tf.gather_nd(roi_boxes, indices[:, :2])
        if self.mold:
            mold_target_masks = MoldBatch(self.max_batch_size)(
                target_masks, batch_indices=indices[:, 0], batch_size=batch_size)
            mold_target_boxes = MoldBatch(self.max_batch_size)(
                target_boxes, batch_indices=indices[:, 0], batch_size=batch_size)
            return mold_target_boxes, mold_target_masks
        else:
            return target_boxes, target_masks

    def get_config(self):
        config = super().get_config()
        config.update({
            "mold": self.mold,
            "max_batch_size": self.max_batch_size
        })
        return config


"""
Layers Related On Training

학습할 Mask 정보를 Model의 출력 형태에 맞춰 배정하는 Layer

"""


class AssignMasks(Layer):
    """ mask_predictions과 gt_inputs 중 IOU가 일정 이상(match_iou_threshold)
    겹치는 것을 기준으로 학습시키는 데에 목적을 둠

    inputs:
        - roi_boxes :
             > shape : [batch size, num roi boxes, 6(cx, cy, w, h, class id, class confidence)]
        - roi_masks :
            roi_masks는 matched_gt_masks의 크기 계산을 위해 들어감 (mask height, mask width, Channel 갯수)
            > shape : [batch size, num roi masks, mask height, mask width, C]
        - gt_boxes  :

            > shape : [batch size, num gt boxes,  6(cx, cy, w, h, class id, class confidence)]
        - gt_masks  :

            > shape : [batch size, num gt boxes,  height, width] -> 0과 1로 채워져 있는 Mask.

    outputs:
        - match_gt_masks :
            > shape : [batch size, num roi masks, mask height, mask width, C]
        - match_gt_classes :
            값들은 각 roi masks 별 Class를 지정.
            > shape : [batch size, num roi masks,]
    """

    def __init__(self, match_iou_threshold=0.5, **kwargs):
        self.match_iou_threshold = match_iou_threshold
        kwargs.update({
            "trainable": False,
        })
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        roi_boxes, roi_masks = inputs[0], inputs[1]
        gt_boxes, gt_masks = inputs[2], inputs[3]

        batch_size = tf.shape(roi_masks)[0]
        num_classes = tf.shape(roi_masks)[-1]

        roi_shape = roi_masks.get_shape().as_list()[2:4]

        def aggregate_image_id(image_id):
            target_gt_boxes = gt_boxes[image_id]
            target_gt_masks = gt_masks[image_id]
            target_roi_boxes = roi_boxes[image_id]
            target_roi_norm_boxes = NormalizeBoxes()(target_roi_boxes,
                                                     shape=tf.shape(target_gt_masks)[1:3])
            iou = CalculateIOU()([tf.reshape(target_gt_boxes[..., :4], (-1, 4)),
                                  tf.reshape(target_roi_boxes[..., :4], (-1, 4))])

            ignore_mask = tf.logical_and(tf.not_equal(target_gt_boxes[:, None, -1], -1.),
                                         tf.not_equal(target_roi_boxes[None, :, -1], -1.))
            ignore_mask = tf.cast(ignore_mask,tf.float32)
            class_mask = tf.cast(tf.equal(target_gt_boxes[:, None, -2],
                                          target_roi_boxes[None, :, -2]),
                                 tf.float32)

            iou = iou * ignore_mask * class_mask

            match_or_not = tf.reduce_max(iou, axis=0) >= self.match_iou_threshold
            match_gt_indices = tf.argmax(iou, axis=0, output_type=tf.int32)

            match_gt_classes = tf.gather(target_gt_boxes[:, 4], match_gt_indices)
            match_gt_classes = tf.where(match_or_not,
                                        match_gt_classes,
                                        tf.ones_like(match_gt_classes)
                                        * tf.cast(num_classes,tf.float32))

            match_gt_masks = tf.image.crop_and_resize(target_gt_masks[..., None],
                                                      target_roi_norm_boxes,
                                                      box_indices=match_gt_indices,
                                                      crop_size=roi_shape)
            match_gt_masks = tf.squeeze(match_gt_masks, axis=-1)

            match_gt_classes = match_gt_classes[:, None, None]
            match_gt_masks = tf.where(match_gt_masks > 0.5,
                                      tf.ones_like(match_gt_masks) * match_gt_classes,
                                      tf.ones_like(match_gt_masks) * tf.cast(num_classes,tf.float32))
            match_gt_masks = tf.cast(match_gt_masks, tf.int32)
            return match_gt_masks

        unique_batch_ids = tf.range(0, tf.cast(batch_size, tf.int32), dtype=tf.int32)
        match_gt_masks = tf.map_fn(aggregate_image_id, unique_batch_ids, dtype=tf.int32)
        return match_gt_masks

    def get_config(self):
        config = super().get_config()
        config.update({
            "match_iou_threshold": self.match_iou_threshold
        })
        return config


__all__ = ["MaskDistribute",
           "MaskSubNet",
           "PyramidRoiAlign",
           "TrimInstances",
           "AssignMasks"]

