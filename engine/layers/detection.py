"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.python.keras.layers import Layer, Conv2D, Add
from tensorflow.python.keras.layers import Concatenate, Reshape
from tensorflow.python.keras.initializers import RandomNormal, Constant
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
from engine.prior import PriorBoxes
from engine.normalization import GroupNormalization
from engine.layers.misc import SqueezeExcite
from engine.layers.misc import MobileSeparableConv2D, MoldBatch
from engine.layers.semantic import ResizeLike


"""
Refinement Network For Detection

Feature Pyramid Networks for object Detection, 2016, Tsung-Yi Lin

High Level의 Feature Map의 정보를 Resolution(Stride)에 맞춰 Low Level Feature Map에 정보를 추가함으로써
즉 작은 사물을 보다 잘 잡을 수 있도록 설계된 Refinement Network

BackBone Network의 Feature Map의 구조를 변경하지 않고도, 덧붙여 성능을 높임 
"""


class FeaturePyramid(Layer):
    """ Build Feature Pyramid Network
    """

    def __init__(self, strides, num_features=256, **kwargs):
        self.strides = strides
        self.num_features = num_features
        super().__init__(**kwargs)
        self.blocks = []
        strides = sorted(self.strides, reverse=True)
        for idx, layer_stride in enumerate(strides):
            block = []
            layer = Conv2D(num_features, (1, 1), padding='same')
            block.append(layer)

            p_num = int(np.round(np.log2(layer_stride)))
            layer = Conv2D(num_features, (3, 3), padding='same',
                           name=f'P{p_num}')
            block.append(layer)
            self.blocks.append(block)

    def call(self, inputs, **kwargs):
        prev = None
        pyramid_outputs = []
        for idx, head in enumerate(inputs[::-1]):
            block = self.blocks[idx]
            lateral = block[0](head)
            if prev is not None:
                upsampled = ResizeLike()(prev, target=lateral)
                output = Add()([lateral, upsampled])
                prev = output
            else:
                output = lateral
                prev = output
            pyramid_output = block[1](output)
            pyramid_outputs.append(pyramid_output)
        return pyramid_outputs[::-1]

    def get_config(self):
        config = super().get_config()
        config.update({
            "strides": self.strides,
            "num_features": self.num_features,
        })
        return config


"""
Head Networks For Detection (Classification & Box Regression)

SSD 류 Head Networks은 각 Grid 내 Point 별로 사물이 있는지를 분류하고(Classification), 
있는 경우 그 위치를 포착한다.

Head Networks는 각 Feauture Map Level 별로 존재하여, Detection에 관련된 총 head Networks의 수는
2 * (# Feature Map Level) 만큼 존재한다.

"""


class BoxRegressionSubNet(Layer):
    """ Box Regression Module in RetinaMask
    """

    def __init__(self, num_blocks,
                 num_depth=4, num_features=256, num_priors=9,
                 use_separable_conv=False, expand_ratio=4.,
                 use_squeeze_excite=False, squeeze_ratio=16.,
                 groups=16, **kwargs):
        self.num_blocks = num_blocks
        self.num_depth = num_depth
        self.num_features = num_features
        self.num_priors = num_priors
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

            layer = Conv2D(num_priors * 4, (3, 3), padding='same',
                           kernel_initializer=RandomNormal(stddev=0.01))
            block.append(layer)
            self.blocks.append(block)

    def call(self, inputs, **kwargs):
        heads = []
        for idx, head in enumerate(inputs):
            block = self.blocks[idx]
            for layer in block:
                head = layer(head)
            head = Reshape((-1, 4))(head)
            heads.append(head)
        return Concatenate(axis=1)(heads)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_blocks": self.num_blocks,
            "num_depth": self.num_depth,
            "num_features": self.num_features,
            "num_priors": self.num_priors,
            "use_separable_conv": self.use_separable_conv,
            "expand_ratio": self.expand_ratio,
            "use_squeeze_excite": self.use_squeeze_excite,
            "squeeze_ratio": self.squeeze_ratio,
            'groups': self.groups
        })
        return config


class ClassificationSubNet(Layer):
    """ Classifcation Module in RetinaMask
    """

    def __init__(self, num_blocks, num_classes,
                 num_depth=4, num_features=256, num_priors=9,
                 use_separable_conv=False, expand_ratio=4.,
                 use_squeeze_excite=False, squeeze_ratio=16.,
                 groups=16, **kwargs):
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_depth = num_depth
        self.num_features = num_features
        self.num_priors = num_priors
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

            layer = Conv2D(num_priors * num_classes, (3, 3),
                           padding='same', activation='sigmoid',
                           kernel_initializer=RandomNormal(stddev=0.01),
                           bias_initializer=Constant(value=-np.log((1 - 0.01) / 0.01)))
            block.append(layer)
            self.blocks.append(block)

    def call(self, inputs, **kwargs):
        heads = []
        for idx, head in enumerate(inputs):
            block = self.blocks[idx]
            for layer in block:
                head = layer(head)
            head = Reshape((-1, self.num_classes))(head)
            heads.append(head)
        return Concatenate(axis=1)(heads)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_blocks": self.num_blocks,
            "num_classes": self.num_classes,
            "num_depth": self.num_depth,
            "num_features": self.num_features,
            "num_priors": self.num_priors,
            "use_separable_conv": self.use_separable_conv,
            "expand_ratio": self.expand_ratio,
            "use_squeeze_excite": self.use_squeeze_excite,
            "squeeze_ratio": self.squeeze_ratio,
            'groups': self.groups
        })
        return config


"""
Layers Related On Prior Boxes(same as Anchor)
"""


class PriorLayer(Layer):
    """ Prior Boxes을 이미지의 크기에 따라 구성하는 Layer

    parameter:
        padding : Resnet의 Stride와 MobileNet의 padding이 다름.
            - Resnet 계열 : padding==same
            - mobileNet 계열 : padding==valid

    Inputs:
        - images :
            Input Images,
            > shape : [batch size, H, W, 3]
    Outputs:
        - pr_boxes :
            모델의 Prior Boxes 정보, 이미지 크기에 따라 결정됨.
            > shape : [batch, num pr boxes, 4(cx, cy, w, h)]
    """
    def __init__(self, prior, padding='same', **kwargs):
        if isinstance(prior, dict):
            self.prior = PriorBoxes(**prior)
        elif isinstance(prior, PriorBoxes):
            self.prior = prior
        else:
            raise ValueError('prior는 PriorBoxes class의 instance이어야 합니다.')
        self.prior_values = [df.values
                             for _, df
                             in self.prior.boxes.groupby('stride')]
        self.padding = padding
        kwargs.update({
            "trainable": False,
        })
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_size, height, width, _ = tf.unstack(tf.shape(inputs),axis=0)
        pr_boxes = []
        for points in self.prior_values:
            boxes = []
            for row in points:
                stride, box_width, box_height = row
                if self.padding == 'same':
                    target_height = tf.cast(tf.math.ceil(height / stride) * stride, tf.int32)
                    target_width = tf.cast(tf.math.ceil(width / stride) * stride, tf.int32)
                else:
                    target_height = tf.cast(tf.math.floor(height / stride) * stride, tf.int32)
                    target_width = tf.cast(tf.math.floor(width / stride) * stride, tf.int32)

                ys = tf.range(stride // 2, target_height, stride)
                xs = tf.range(stride // 2, target_width, stride)
                xs, ys = tf.meshgrid(xs, ys)

                box_width = tf.ones_like(xs) * box_width
                box_height = tf.ones_like(ys) * box_height
                block_centers = tf.stack((xs, ys, box_width, box_height),
                                         axis=-1)
                boxes.append(block_centers)
            boxes = tf.stack(boxes, axis=2)
            boxes = tf.reshape(boxes, (-1, 4))
            pr_boxes.append(boxes)
        pr_boxes = tf.concat(pr_boxes, axis=0)
        pr_boxes = K.repeat(pr_boxes,batch_size)
        pr_boxes = tf.transpose(pr_boxes, (1, 0, 2))
        return pr_boxes

    def get_config(self):
        config = super().get_config()
        config.update({
            "prior": self.prior.config,
            'padding': self.padding
        })
        return config


class RestoreBoxes(Layer):
    """ Model의 Box Regression output와 Prior boxes을 바탕으로
    predicted box의 (cx,cy,w,h)을 복원하는 Layer

    inputs:
        - loc_pred :
            모델의 Box Regression 정보
            > shape : [batch size, num pr boxes, 4(delta cx, delta cy, delta w, delta h)]
        - pr_boxes :
            모델의 Prior Boxes 정보, 이미지 크기에 따라 결정됨.
            > shape : [batch, num pr boxes, 4(cx, cy, w, h)]
    Outputs: 꼭지점이 기준인 좌표 표현
        - restore_boxes :
            복원된 Box Regression 정보.
            > [batch size, num pr boxes, 4(cx, cy, w, h)]
    """
    def call(self, inputs, **kwargs):
        loc_pred = inputs[0]

        pr_boxes = inputs[1]
        loc_pred = tf.cast(loc_pred, tf.float32)
        pr_boxes = tf.cast(pr_boxes, tf.float32)

        res_cx = (loc_pred[..., 0]
                  * pr_boxes[..., 2]
                  + pr_boxes[..., 0])
        res_cy = (loc_pred[..., 1]
                  * pr_boxes[..., 3]
                  + pr_boxes[..., 1])
        res_w = (tf.exp(loc_pred[..., 2])
                 * pr_boxes[..., 2])
        res_h = (tf.exp(loc_pred[..., 3])
                 * pr_boxes[..., 3])
        restore_boxes = tf.stack([
            res_cx, res_cy, res_w, res_h], axis=-1)
        return restore_boxes


class NormalizeBoxes(Layer):
    """ Model의 Predicted Box의 좌표 정보(cx,cy,w,h)를 normalized된 (y1,x1,y2,x2)으로
    바꾸는 Layer

    inputs:
        - boxes :
            중심점이 기준인 좌표 표현
            > shape : [batch size, num boxes, 4(cx,cy,w,h)]
    Outputs:
        - norm_boxes :
            꼭지점이 기준인 좌표 표현
            > shape : [batch size, num boxes, 4(y1,x1,y2,x2)]
    """
    def call(self, inputs, **kwargs):
        boxes = inputs
        shape = kwargs.get('shape', tf.ones((2,)))

        image_height = tf.cast(shape[0], dtype=tf.float32)
        image_width = tf.cast(shape[1], dtype=tf.float32)

        cx, cy, w, h = tf.unstack(boxes[..., :4], axis=-1)

        x1 = (cx - w / 2) / image_width
        y1 = (cy - h / 2) / image_height
        x2 = (cx + w / 2) / image_width
        y2 = (cy + h / 2) / image_height

        norm_boxes = tf.stack([y1, x1, y2, x2], axis=-1)
        return norm_boxes


class CalculateIOU(Layer):
    """ 주어진 두개의 Boxes들에서 각 쌍 별로 IOU를 계산하는 Layer

    inputs:
        - aa_boxes :
            > shape : [num aa boxes, 4(cx, cy, w, h)]
        - bb_boxes :
            > shape : [num bb boxes, 4(cx, cy, w, h)]
    outputs:
        - iou value matrix
            각 행(aa boxes)과 열(bb boxes)에 맞추어 iou value가 매칭되어 있는 행렬
            > shape : [num aa boxes, num bb boxes]
    """
    def call(self, inputs, **kwargs):
        aa_boxes, bb_boxes = inputs[0], inputs[1]
        aa_boxes = tf.cast(aa_boxes, tf.float32)
        bb_boxes = tf.cast(bb_boxes, tf.float32)

        # Calculate IOU
        # 1. Calculate Each Box size
        aa_area = bb_boxes[:, 2] * bb_boxes[:, 3]
        bb_area = aa_boxes[:, 2] * aa_boxes[:, 3]
        areas = aa_area[None, :] + bb_area[:, None]

        # 2. Normalize coordinates ( (cx, cy, w, h) -> (y1, x1, y2, x2) )
        aa_norm_boxes = NormalizeBoxes()(aa_boxes[:, :4])
        bb_norm_boxes = NormalizeBoxes()(bb_boxes[:, :4])

        ay1, ax1, ay2, ax2 = tf.unstack(aa_norm_boxes[:, None], axis=-1)
        by1, bx1, by2, bx2 = tf.unstack(bb_norm_boxes[None, :], axis=-1)

        in_ymin = tf.maximum(by1, ay1)
        in_xmin = tf.maximum(bx1, ax1)
        in_ymax = tf.minimum(by2, ay2)
        in_xmax = tf.minimum(bx2, ax2)

        in_width = tf.maximum(0., in_xmax - in_xmin)
        in_height = tf.maximum(0., in_ymax - in_ymin)

        # 3. Calculate intersection size and union size
        intersection = in_width * in_height
        union = areas - intersection

        iou = intersection / (union + 1e-5)
        return iou



"""
Post Process For Detection

모든 Grid Point에서 Classification과 Box Regression을 수행하기 때문에, 수백~수천개의 Box들이 만들어진다.
하나의 사물 주위에 걸쳐 있는 모든 Grid Point가 그 사물에 대한 Bounding Box를 그리기 때문에, 
이렇게 겹친 Bounding Box들을 소거해주는 작업이 필요하다. 이러한 알고리즘을 Non-Maximum Suppression이라 부른다. 
"""


class DetectionProposal(Layer):
    """ Prior Boxes에 대한 위치 정보와 Score 정보를 받아,
    Non-Maximum Suppression을 거쳐 제일 높은 정보를 득표한 것들만 추출

    First, During the bounding box inference we use a confidence threshold of 0.05
    to filter out predictions with low confidence.

    Second, we select the top 1000 scoring boxes from each prediction layer.
    (-> 이 과정은 어떤 의미를 가지는지 파악하지 못하였고, 현재 구현 형태에서는 이렇게 만들기가 어려워 생략하였습니다.)

    Third, we apply non-maximum suppression(nms) with threshold 0.4 for each class separately.

    Finally, the top-100 scoring predictions are selected for each images.

    For mask inference, we use the top 50 bounding box predictions as mask proposals.

    여기에 추가적으로 구현한 것이 post non-maximum-suppression으로,
    이미지 중 다른 라벨이지만 많이 겹치는 경우를 지워줌. 이는 라벨을 오인한 케이스로 파악됨.

    inputs:
        - cls_pred :
            모델의 Classification 정보
            > shape : [batch size, num boxes, num_classes],
        - boxes :
            RestoreBoxes로 복원된 Bounding Box 좌표값
            > shape : [batch size, num boxes, 4(cx, cy, w, h)]
        - images :
            입력 이미지, 좌표를 [0,1]로 Normalization하는 데에 쓰임
            > shape : [batch size, height, width, 3]
    Outputs:
        - proposed_boxes :
            Non-Maximum Suppression을 거쳐 추려진 Object Detection 정보
            > shape : [batch size, num boxes, 6(cx, cy, w, h, class id, class confidence)]
    """
    def __init__(self, min_confidence=0.05,
                 nms_iou_threshold=0.4,
                 post_iou_threshold=0.65,
                 nms_max_output_size=1000,
                 max_batch_size=64,
                 **kwargs):
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.post_iou_threshold = post_iou_threshold
        self.nms_max_output_size = nms_max_output_size
        self.max_batch_size = max_batch_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        cls_pred = inputs[0]
        boxes = inputs[1]

        batch_size = tf.shape(cls_pred)[0]
        num_classes = tf.cast(tf.shape(cls_pred)[-1], tf.int64)
        norm_boxes = NormalizeBoxes()(boxes)

        # 1. filtering out predictions with low cls_pred.
        keep_indices = tf.where(cls_pred >= self.min_confidence)
        keep_image_ids = keep_indices[:, 0]
        keep_class_ids = keep_indices[:, 2]
        keep_confidence = tf.gather_nd(cls_pred, keep_indices)
        keep_norm_boxes = tf.gather_nd(norm_boxes, keep_indices[:, :2])

        # * CAUTION : tf.images.combined_non_max_suppression은 Gradient가 정의되어 있지 않습니다.
        # 위의 코드를 이용하면 보다 간결하게 코드를 짤 수 있지만, 역전파 에러가 발생하여 학습을 시킬 수 없습니다.
        def nms_keep_class_id(img_cls_id):
            """
            이미지 ID와 Class ID 모두 동일한 것 중에서 Non Maximum Suppression을 적용

            :param img_cls_id:
            :return:
            """
            ixs = tf.where(tf.equal(keep_img_cls_ids, img_cls_id))[:, 0]
            nms_keep_ = tf.image.non_max_suppression(tf.gather(keep_norm_boxes, ixs),
                                                     tf.gather(keep_confidence, ixs),
                                                     max_output_size=self.nms_max_output_size,
                                                     iou_threshold=self.nms_iou_threshold)
            nms_keep_ = tf.gather(keep_indices, tf.gather(ixs, nms_keep_))
            gap_ = self.nms_max_output_size - tf.shape(nms_keep_)[0]
            nms_keep_ = tf.pad(nms_keep_, [(0, gap_), (0, 0)], mode='CONSTANT', constant_values=-1)
            nms_keep_.set_shape([self.nms_max_output_size, 3])
            return nms_keep_

        # The unique ID by combining images id and class ids
        # the range of class id : [0, num_classes)
        keep_img_cls_ids = keep_image_ids * (num_classes + 1) + keep_class_ids
        unique_img_cls_ids = tf.unique(keep_img_cls_ids)[0]
        # 2. we apply non-maximum suppression(nms) with threshold 0.4 for each class separately
        per_class_keep = tf.map_fn(nms_keep_class_id,
                                   unique_img_cls_ids,
                                   dtype=tf.int64)
        per_class_keep = tf.gather_nd(per_class_keep,
                                      tf.where(per_class_keep[..., 0] > -1))
        per_class_image_ids = per_class_keep[:, 0]
        per_class_confidence = tf.gather_nd(cls_pred, per_class_keep)
        per_class_norm_boxes = tf.gather_nd(norm_boxes, per_class_keep[:, :2])

        def nms_keep_image_id(image_id):
            """
            이미지 ID가 동일한 것중에서 Non Maximum Suppression을 적용

            Post Processing 과정으로, 다른 Class의 Bounding Box가 있을 경우,
            지나치게 많이 겹쳐있는 경우에 Confidence가 높은 녀석으로 추려내는 역할

            :param image_id:
            :return:
            """
            ixs = tf.where(tf.equal(per_class_image_ids, image_id))[:, 0]
            nms_keep_ = tf.image.non_max_suppression(tf.gather(per_class_norm_boxes, ixs),
                                                     tf.gather(per_class_confidence, ixs),
                                                     max_output_size=self.nms_max_output_size,
                                                     iou_threshold=self.post_iou_threshold)
            nms_keep_ = tf.gather(per_class_keep, tf.gather(ixs, nms_keep_))
            gap_ = self.nms_max_output_size - tf.shape(nms_keep_)[0]
            nms_keep_ = tf.pad(nms_keep_, [(0, gap_), (0, 0)], mode='CONSTANT', constant_values=-1)
            nms_keep_.set_shape([self.nms_max_output_size, 3])
            return nms_keep_

        unique_image_ids = tf.range(0, tf.cast(batch_size, tf.int64), dtype=tf.int64)
        nms_keep = tf.map_fn(nms_keep_image_id, unique_image_ids, dtype=tf.int64)
        nms_keep = tf.gather_nd(nms_keep,
                                tf.where(nms_keep[..., 0] > -1))

        nms_confidence = tf.gather_nd(cls_pred, nms_keep)[:, None]
        nms_boxes = tf.gather_nd(boxes, nms_keep[:, :2])
        nms_class_ids = nms_keep[:, 2][:, None]
        nms_image_ids = tf.cast(nms_keep[:, 0], tf.int32)

        nms_class_ids = tf.cast(nms_class_ids, dtype=tf.float32)
        nms_results = tf.concat([nms_boxes, nms_class_ids, nms_confidence], axis=1)
        results = MoldBatch(max_batch_size=self.max_batch_size)(
            nms_results, batch_indices=nms_image_ids, batch_size=batch_size)

        return tf.stop_gradient(results)

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_confidence": self.min_confidence,
            "nms_iou_threshold": self.nms_iou_threshold,
            "post_iou_threshold": self.post_iou_threshold,
            "nms_max_output_size": self.nms_max_output_size,
            "max_batch_size": self.max_batch_size
        })
        return config


"""
Layers Related On Training

학습할 객체 정보를 Model의 출력 형태에 맞춰 배정하는 Layer

"""


class AssignBoxes(Layer):
    """ 주어진 gt boxes와 prior boxes의 정보를 바탕으로, 모델이 출해야 하는 정답값의 형태로
    정답 값들을 prior boxes의 위치에 할당하는 Layer

    (1) gt boxes가 대응되는 prior boxes에 배정하기
    (2) 배정된 Prior boxes와 GT boxes의 차이를 계산하기
    (3) 차이 만큼을 해당 Prior Boxes의 위치 grid에 값을 배정하기

    inputs:
        - gt_boxes :
            학습할 Ground Truth Boxes 정보.
            배치 데이터 셋 중 가장 사물 정보가 많은 것을 기준으로(max gt boxes), box 정보는
            -1로 padding 처리되어 있음.
            > shape : [batch, max gt boxes, 6(cx, cy, w, h, class id, conf)]
        - pr_boxes :
            모델의 Prior Boxes 정보, 이미지 크기에 따라 결정됨.
            > shape : [batch, num pr boxes, 4(cx, cy, w, h)]
    outputs:
        - cls_true : [batch, num pr boxes, C]
        - loc_true : [batch, num pr boxes, 4]
        - assign_mask : [batch, num pr boxes, 1]
    """

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        gt_boxes = inputs[0]
        pr_boxes = inputs[1]
        batch_size, num_gt, _ = tf.unstack(tf.shape(gt_boxes))
        batch_size, num_pr, _ = tf.unstack(tf.shape(pr_boxes))
        gt_labels = gt_boxes[..., -2]
        gt_confidence = gt_boxes[..., -1]

        iou = CalculateIOU()([tf.reshape(gt_boxes[..., :4], (-1, 4)),
                              pr_boxes[0]])
        iou = tf.reshape(iou, (batch_size, num_gt, num_pr))
        mask = tf.cast(tf.not_equal(gt_boxes[..., 0], -1.),tf.float32)
        iou = iou * mask[..., None]

        match_indices = tf.where(iou >= 0.5)

        gts, bs = tf.meshgrid(tf.range(0, num_gt), tf.range(0, batch_size))
        indices = tf.stack([tf.reshape(bs, (-1,)), tf.reshape(gts, (-1,))], axis=1)
        indices = tf.cast(indices, tf.int64)
        top_indices = tf.argmax(tf.reshape(iou, (-1, num_pr)), axis=1)
        best_indices = tf.concat([indices, top_indices[..., None]], axis=1)
        not_matched = tf.where(tf.reshape(gt_confidence, (-1,)) > 0.)[:, 0]
        best_indices = tf.gather(best_indices, not_matched)
        match_indices = tf.concat([match_indices, best_indices], axis=0)

        batch_indices, gt_indices, pr_indices = tf.unstack(match_indices, axis=1)
        pr_match_indices = tf.stack([batch_indices, pr_indices], axis=1)
        gt_match_indices = tf.stack([batch_indices, gt_indices], axis=1)

        cls_true = tf.ones((batch_size, num_pr)) * -1.
        cls_true = tf.tensor_scatter_nd_update(cls_true, pr_match_indices,
                                               tf.gather_nd(gt_labels, gt_match_indices))
        cls_true = tf.where(tf.not_equal(cls_true, -1.),
                            cls_true,
                            tf.ones_like(cls_true) * self.num_classes)
        cls_true = tf.one_hot(tf.cast(cls_true, tf.int32), self.num_classes + 1)

        assign_mask = cls_true[..., -1]
        ignore_indices = tf.where((iou < 0.5) & (iou >= 0.4))

        ig_batch_indices, ig_gt_indices, ig_pr_indices = tf.unstack(
            tf.transpose(ignore_indices))
        ig_pr_indices = tf.stack([ig_batch_indices,ig_pr_indices], axis=1)
        ignore_mask = tf.scatter_nd(ig_pr_indices,
                                    tf.ones_like(ig_gt_indices),
                                    (batch_size, num_pr,))
        assign_mask = tf.where(ignore_mask > 0,
                               tf.ones_like(assign_mask) * -1,
                               assign_mask)

        p_cx, p_cy, p_w, p_h = tf.unstack(
            tf.gather_nd(tf.cast(pr_boxes, tf.float32),
                         pr_match_indices), axis=1)
        g_cx, g_cy, g_w, g_h = tf.unstack(
            tf.gather_nd(tf.cast(gt_boxes[..., :4], tf.float32),
                         gt_match_indices), axis=1)

        hat_g_cx = (g_cx - p_cx) / p_w
        hat_g_cy = (g_cy - p_cy) / p_h
        hat_g_w = tf.log(g_w / p_w)
        hat_g_h = tf.log(g_h / p_h)

        hat_g_cx = tf.scatter_nd(pr_match_indices, hat_g_cx,
                                 (batch_size, num_pr,))
        hat_g_cy = tf.scatter_nd(pr_match_indices, hat_g_cy,
                                 (batch_size, num_pr,))
        hat_g_w = tf.scatter_nd(pr_match_indices, hat_g_w,
                                (batch_size, num_pr,))
        hat_g_h = tf.scatter_nd(pr_match_indices, hat_g_h,
                                (batch_size, num_pr,))

        cls_true = cls_true[..., :self.num_classes]
        loc_true = tf.stack([hat_g_cx, hat_g_cy, hat_g_w, hat_g_h], axis=2)
        assign_mask = assign_mask[..., None]
        return cls_true, loc_true, assign_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes
        })
        return config


__all__ = [
    "FeaturePyramid",
    "ClassificationSubNet",
    "BoxRegressionSubNet",
    "PriorLayer",
    "RestoreBoxes",
    "NormalizeBoxes",
    "CalculateIOU",
    "DetectionProposal",
    "AssignBoxes"
]

