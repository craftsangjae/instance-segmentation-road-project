"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.python.keras.layers import Layer, GlobalAveragePooling2D, Dense
from tensorflow.python.keras.layers import Conv2D, ReLU, Add, DepthwiseConv2D
from engine.normalization import GroupNormalization
import tensorflow as tf
import numpy as np
"""
Additional Custom Layer For Light Weight and Accurate Model

기존 모델을 개선하기 위해 만들어진 Custom Layer들

SqueezeExcite : 정확도 향상을 위함
<- Squeeze-and-Excitation Networks, 2017

MobileSeparableConv2D : inference time 향상을 위함
<- MobileNets: Efficient Convolutional Neural Networks for mobile vision applications

"""


class SqueezeExcite(Layer):
    """ SqueezeAndExcite Network
    """

    def __init__(self, ratio=16., **kwargs):
        self.ratio = ratio
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_channel = input_shape[-1]
        self.dense1 = Dense(n_channel // self.ratio,
                            activation='relu',
                            kernel_initializer='he_normal',
                            use_bias=False)
        self.dense2 = Dense(n_channel, activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            use_bias=False)

    def call(self, inputs, **kwargs):
        se = GlobalAveragePooling2D()(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = se[:, None, None, :]
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update({
            "ratio": self.ratio,
        })
        return config


class MobileSeparableConv2D(Layer):
    """ Separable Convolution Layer With Inverted Residual Block

        Operation Order:
            Inputs->
            EXPAND Convolution->
            Depthwise Convolution ->
            Squeeze Convolution ->
            [Inputs+Expand]
    """
    def __init__(self, filters, kernel_size=(3, 3), expand_ratio=4., stride=1, groups=16, **kwargs):
        prefix = kwargs.get('name', 'SeparableConv2d')
        super().__init__(**kwargs)

        self.filters = filters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.groups = groups

        self.expand_conv2d = Conv2D(int(self.expand_ratio * filters), (1, 1), use_bias=False,
                                    name=prefix + '_expand_conv')
        self.expand_norm = GroupNormalization(groups=self.groups, name=prefix + '_expand_GN')
        self.expand_relu = ReLU(name=prefix + '_expand_relu')

        self.depth_conv2d = DepthwiseConv2D(self.kernel_size, (stride, stride), padding='same',
                                            use_bias=False, name=prefix + '_depthwise')
        self.depth_norm = GroupNormalization(groups=self.groups, name=prefix + '_depthwise_GN')
        self.depth_relu = ReLU(name=prefix + 'depthwise_relu')

        self.squeeze_conv2d = Conv2D(filters, (1, 1), use_bias=False, name=prefix + '_squeeze_conv')
        self.squeeze_norm = GroupNormalization(groups=self.groups, name=prefix + '_squeeze_GN')
        self.skip_connection = Add(name=prefix + "skip_connection")

    def call(self, inputs, **kwargs):
        x = self.expand_conv2d(inputs)
        x = self.expand_norm(x)
        x = self.expand_relu(x)

        x = self.depth_conv2d(x)
        x = self.depth_norm(x)
        x = self.depth_relu(x)

        x = self.squeeze_conv2d(x)
        x = self.squeeze_norm(x)
        x = self.skip_connection([inputs, x])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "expand_ratio": self.expand_ratio,
            "stride": self.stride,
            "groups": self.groups
        })
        return config


"""
Layers related on Input Resolution

이미지 별로 상이한 Resolution을 가지고 있음. 이를 일괄적으로 통일된 Resolution 하에서 처리하기 위해 구성함
모델 파이프라인에서 제일 첫 단계와 제일 마지막 단계에 붙게 됨

* DownSampleInput -> Target Size로 Input Image를 Resize하는 연산
* UpSampleOutput -> Input Size로 Output 결과를 Restore하는 연산


"""


class DownSampleInput(Layer):
    """
    Input을 특정 해상도 수준(target_height, target_width)으로, 비율에 맞추어 resize하는 Layer

    align_corners=False 이어야, proposed box의 좌표값과 매칭하기가 간단해짐(단순 배수로 매칭이 가능해지기 때문)
    """
    def __init__(self, target_size=(540, 960), **kwargs):
        self.target_size = target_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        _, input_h, input_w, _ = tf.unstack(tf.shape(inputs))
        target_h = tf.cast(self.target_size[0], tf.float32)
        target_w = tf.cast(self.target_size[1], tf.float32)
        input_h, input_w = tf.cast(input_h, tf.float32), tf.cast(input_w, tf.float32)

        ratio = tf.minimum(target_h / input_h, target_w / input_w)
        target_size = tf.cast((ratio * input_h, ratio * input_w), tf.int32)

        return tf.compat.v1.image.resize_bilinear(inputs, size=target_size,
                                                  align_corners=True)

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_size": self.target_size,
        })
        return config


class UpSampleOutput(Layer):
    """
    Ouput을 특정 해상도 수준(target_height, target_width)으로, 비율에 맞추어 resize하는 Layer

    """
    def call(self, inputs, **kwargs):
        target_node = kwargs.get('target')
        roi_box = inputs[0]
        roi_mask = inputs[1]
        semantic_output = inputs[2]

        src_shape = tf.cast(tf.shape(semantic_output)[1:3], tf.float32)
        dst_shape = tf.cast(tf.shape(target_node)[1:3], tf.float32)
        ratio_shape = dst_shape / src_shape

        cx, cy, w, h, label, confs = tf.unstack(roi_box, axis=-1)
        cx = tf.cast(cx * ratio_shape[0], tf.int32)
        cy = tf.cast(cy * ratio_shape[1], tf.int32)
        w = tf.cast(w * ratio_shape[0], tf.int32)
        h = tf.cast(h * ratio_shape[1], tf.int32)
        label = tf.cast(label, tf.int32)
        confs = tf.cast(confs * 100, tf.int32)
        roi_box = tf.stack([cx, cy, w, h, label, confs], axis=-1)

        roi_mask = tf.cast(roi_mask>0.5, tf.int32)

        target_size = tf.cast(dst_shape, tf.int32)
        # Caution why use "align_corners=False"?
        # >>> https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/8
        semantic_output = tf.compat.v1.image.resize_bilinear(semantic_output, target_size,
                                                             align_corners=True)
        semantic_output = tf.cast(semantic_output>0.5, tf.int32)
        return roi_box, roi_mask, semantic_output


"""
기타 연산들

Keras의 동작 메커니즘에 맞추어, 부수적으로 발생하는 Layer들
"""


class Identity(Layer):
    """ Layer의 이름을 지정하기 위한 Custom Layer
    """
    def call(self, inputs, **kwargs):
        return inputs


class MoldBatch(Layer):
    """ 정보를 배치 단위로 따로 묶어주는 Layer

    inputs:
        - inputs :
            > shape : [num boxes, ....]
        - batch_indices :
            > shape : [num boxes, ]
        - batch_size :
            > shape : Scalar Tensor
    Outputs:
        - molded_inputs :
            > shape : [batch size, max num boxes per batch, ....]
    """
    def __init__(self, max_batch_size=None, **kwargs):
        self.max_batch_size = max_batch_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_indices = tf.cast(kwargs.get('batch_indices'), tf.int32)
        batch_size = kwargs.get('batch_size')

        _, _, box_counts = tf.unique_with_counts(batch_indices)
        max_box_counts = tf.maximum(1, tf.reduce_max(box_counts))
        original_shapes = inputs.get_shape().as_list()[1:]

        if self.max_batch_size is None:
            unique_batch_ids = tf.range(0, tf.cast(batch_size, tf.int32), dtype=tf.int32)

            def aggregate_batch_id(batch_id):
                """
                이미지 ID가 동일한 것끼리 nms_results를 묶어줌

                :param batch_id:
                :return:
                """
                ixs = tf.where(tf.equal(batch_indices, batch_id))[:, 0]
                case_results = tf.gather(inputs, ixs)

                case_gap = max_box_counts - tf.shape(case_results)[0]
                case_n_dims = len(case_results.get_shape())
                case_pad_size = tf.zeros((case_n_dims - 1, 2), dtype=tf.int32)
                case_batch_pad = tf.stack([0, case_gap])[None]
                case_pad_size = tf.concat([case_batch_pad, case_pad_size], axis=0)
                case_results = tf.pad(case_results, case_pad_size, constant_values=-1)
                return case_results

            results = tf.map_fn(aggregate_batch_id,
                                unique_batch_ids,
                                dtype=tf.float32)

            # Exception Case Handling Code. NMS 결과로 아무것도 없었을 경우
            # 첫번째 축이 (0, ...)의 형태로 반환하게 되기 때문에, Error가 발생할 수 있음
            # 이 경우, 첫번째 축으로 배치크기만큼 padding을 붙여줌으로써 Error Case Handling
            gap = batch_size - tf.shape(results)[0]
            n_dims = len(results.get_shape())
            pad_size = tf.zeros((n_dims - 1, 2), dtype=tf.int32)
            batch_pad = tf.stack([0, gap])[None]
            pad_size = tf.concat([batch_pad, pad_size], axis=0)
            results = tf.pad(results, pad_size, constant_values=-1.)
        else:
            results = []
            for result in tf.dynamic_partition(inputs, batch_indices, 32):
                case_gap = max_box_counts - tf.shape(result)[0]
                case_n_dims = len(result.get_shape())
                case_pad_size = tf.zeros((case_n_dims - 1, 2), dtype=tf.int32)
                case_batch_pad = tf.stack([0, case_gap])[None]
                case_pad_size = tf.concat([case_batch_pad, case_pad_size], axis=0)
                result = tf.pad(result, case_pad_size, constant_values=-1)
                results.append(result)
            results = tf.stack(results, axis=0)
            results = results[:batch_size]

        return tf.reshape(results, (batch_size, -1, *original_shapes))

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_batch_size": self.max_batch_size
        })
        return config


class ResizeLike(Layer):
    """Change the size of tensor(height & width) to target node"""
    def __init__(self, align_corners=True, **kwargs):
        self.align_corners = align_corners
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        target_node = kwargs.get('target')
        target_shape = tf.shape(target_node)
        target_size = (target_shape[1], target_shape[2])
        return tf.compat.v1.image.resize_bilinear(inputs, target_size,
                                                  align_corners=self.align_corners)

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list)
        input_shape, target_shape = input_shapes
        return input_shape[0], target_shape[1], target_shape[2], input_shape[-1]

    def get_config(self):
        config = super().get_config()
        config.update({
            "align_corners": self.align_corners
        })
        return config

"""
Layers Related on Model Serving

Keras의 동작 메커니즘에 맞추어, 부수적으로 발생하는 Layer들
"""


class DecodeImageContent(Layer):
    """
    Image Format(jpg, jpeg, png)으로 압축된 이미지를 Decode하여
    Tensorflow Array를 반환
    """
    def call(self, inputs, **kwargs):
        inputs = tf.reshape(inputs,shape=()) # From Rank 1 to Rank 0
        image = tf.io.decode_image(inputs, channels=3,
                                   expand_animations=False)
        image.set_shape([None,None,3])
        image = tf.expand_dims(image, axis=0)
        image.set_shape([None, None,None,3])
        return image


class EncodeImageContent(Layer):
    """
    Tensorflow Array를 Image Foramt(jpeg)으로 압축
    """
    def call(self, inputs, **kwargs):
        contents = tf.io.encode_jpeg(inputs[0])
        contents = tf.expand_dims(contents, axis=0)
        contents.set_shape([None, ])
        return contents


class CropAndPadMask(Layer):
    """
    Instance Mask의 크기를 이미지 전체에 맞게 resize -> pad하는 코드
    """
    def call(self, inputs, **kwargs):
        images = inputs[0]
        det_outs = inputs[1]
        ins_outs = inputs[2]

        _, image_h, image_w, _ = tf.unstack(tf.shape(images))
        batch_size, num_box_counts, _ = tf.unstack(tf.shape(det_outs))

        threshold = tf.reduce_max(det_outs[..., -1])
        threshold = tf.cond(threshold > 50,
                            lambda: tf.constant(50),
                            lambda: tf.constant(-100))
        indices = tf.where(det_outs[..., -1] >= threshold)

        def pad_mask_per_batch_id(index):
            box = tf.gather_nd(det_outs, index)
            box = tf.maximum(box, 1)
            mask = tf.gather_nd(ins_outs, index)

            cx, cy, w, h, _, _ = tf.unstack(tf.cast(box, tf.float32))

            xmin = tf.clip_by_value(tf.cast(tf.math.ceil(cx - w / 2), tf.int32),
                                    0, image_w)
            xmax = tf.clip_by_value(tf.cast(tf.math.ceil(cx + w / 2), tf.int32),
                                    0, image_w)
            ymin = tf.clip_by_value(tf.cast(tf.math.ceil(cy - h / 2), tf.int32),
                                    0, image_h)
            ymax = tf.clip_by_value(tf.cast(tf.math.ceil(cy + h / 2), tf.int32),
                                    0, image_h)

            resized_mask = tf.squeeze(
                tf.image.resize(mask[None, ..., None], (ymax-ymin, xmax-xmin),
                                align_corners=True),
                axis=(0, -1))
            return tf.pad(resized_mask, ((ymin, image_h - ymax),
                                         (xmin, image_w - xmax)))

        crop_and_padded_masks = tf.map_fn(pad_mask_per_batch_id,
                                          indices, dtype=tf.float32)
        crop_and_padded_masks = tf.scatter_nd(
            indices, crop_and_padded_masks,
            (batch_size, num_box_counts, image_h, image_w))

        return crop_and_padded_masks


class DrawSegmentation(Layer):
    """
    """
    def __init__(self, colors, alpha=.3, **kwargs):
        self.colors = colors
        self.alpha = alpha
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        images = tf.cast(inputs[0], tf.float32)
        seg_outs = tf.cast(inputs[1], tf.float32)
        colors = tf.constant(self.colors, tf.float32)

        color_seg_outs = tf.reduce_sum(colors * seg_outs[..., None], axis=-2)
        vis = tf.clip_by_value(
            images + color_seg_outs * tf.constant(self.alpha, tf.float32), 0, 255)
        vis = tf.cast(vis, tf.uint8)
        return vis

    def get_config(self):
        config = super().get_config()
        config.update({
            "colors": self.colors,
            "alpha": self.alpha
        })
        return config


class DrawInstance(Layer):
    """
    """
    def __init__(self, colors, alpha=.3, **kwargs):
        self.colors = colors
        self.alpha = alpha
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        images = inputs[0]
        det_outs = inputs[1]
        crop_and_padded_masks = inputs[2]

        batch_size, *_ = tf.unstack(tf.shape(det_outs))
        colors = tf.cast(self.colors, tf.float32)
        num_classes, _ = tf.unstack(tf.shape(colors))

        def aggregate_batch_id(batch_id):
            det_out = det_outs[batch_id]
            mask = crop_and_padded_masks[batch_id]

            def aggregate_class_id(class_id):
                indices = tf.where(tf.equal(det_out[..., -2], class_id))
                class_masks = tf.gather_nd(mask, indices)
                class_masks = tf.reduce_sum(class_masks, axis=0)
                class_masks = tf.cast(class_masks > 0.5, tf.float32)
                return class_masks

            return tf.map_fn(aggregate_class_id, tf.range(0, num_classes), tf.float32)

        masks = tf.map_fn(aggregate_batch_id,
                          tf.range(0, batch_size, dtype=tf.int32),
                          tf.float32)
        masks = tf.transpose(masks, (0, 2, 3, 1))
        vis = DrawSegmentation(self.colors, self.alpha)([images, masks])
        return vis

    def get_config(self):
        config = super().get_config()
        config.update({
            "colors": self.colors,
            "alpha": self.alpha
        })
        return config


class DrawBoxes(Layer):
    """
    """
    def call(self, inputs, **kwargs):
        images = inputs[0]
        det_outs = inputs[1]

        batch_size, image_h, image_w, _ = tf.unstack(tf.shape(images))

        boxes = tf.maximum(det_outs[..., :4], 0)
        cx, cy, w, h = tf.unstack(tf.cast(boxes, tf.float32), axis=-1)

        image_h = tf.cast(image_h, tf.float32)
        image_w = tf.cast(image_w, tf.float32)

        xmin, xmax = (cx - w / 2) / image_w, (cx + w / 2) / image_w
        ymin, ymax = (cy - h / 2) / image_h, (cy + h / 2) / image_h
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        vis = tf.image.draw_bounding_boxes(tf.cast(images, tf.float32),
                                           bboxes,
                                           colors=tf.constant(
                                               [[255, 255, 255, 255]], tf.float32))
        vis = tf.clip_by_value(vis, 0., 255.)
        vis = tf.cast(vis, tf.uint8)
        return vis


class CrackToInstance(Layer):
    """ Crack 정보를 Instance 정보로 변경하는 메소드


    """
    def __init__(self, crack_id=5, **kwargs):
        self.crack_id = crack_id
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        indices = tf.where(inputs)
        indices = tf.cond(tf.size(indices) > 0,
                          lambda: indices,
                          lambda: tf.constant([[0, 0, 0]], dtype=tf.int64))

        ymin, xmin = tf.unstack(tf.reduce_min(indices, axis=0))[1:]
        ymax, xmax = tf.unstack(tf.reduce_max(indices, axis=0))[1:]
        height = tf.cast(ymax - ymin, tf.int32)
        width = tf.cast(xmax - xmin, tf.int32)
        cy = tf.cast(ymin, tf.int32) + tf.cast(height / 2, tf.int32)
        cx = tf.cast(xmin, tf.int32) + tf.cast(width / 2, tf.int32)
        class_id = tf.ones_like(cx) * 5 # CRACK LABEL ID
        conf = tf.clip_by_value(tf.ones_like(cx) * 100 * height * width, 0, 100)

        crack_det_outs = tf.tile(
            tf.stack([cx, cy, width, height, class_id, conf])[None],
            [tf.shape(inputs)[0], 1])
        crack_det_outs = tf.expand_dims(crack_det_outs, axis=1)
        crack_seg_outs = tf.cast(inputs[:, None], tf.float32)

        return crack_det_outs, crack_seg_outs

    def get_config(self):
        config = super().get_config()
        config.update({
            'crack_id': self.crack_id
        })
        return config


class SummaryOutput(Layer):
    """

    """
    def __init__(self, default_road_size=3.25, **kwargs):
        self.default_road_size = default_road_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        det_outs = inputs[0]
        seg_outs = inputs[1]
        crop_ins_outs = inputs[2]

        # crack to instance
        crack_det_outs, crack_seg_outs = CrackToInstance()(seg_outs[..., 2])

        det_outs = tf.cond(tf.reduce_all(crack_det_outs[..., -1] > 0),
                           lambda: tf.concat([det_outs, crack_det_outs], axis=1),
                           lambda: det_outs)

        crop_ins_outs = tf.cond(tf.reduce_all(crack_det_outs[..., -1] > 0),
                                lambda: tf.concat([crop_ins_outs, crack_seg_outs], axis=1),
                                lambda: crop_ins_outs)

        cx, cy, w, h, classes, conf = tf.unstack(
            tf.cast(det_outs[..., :6], tf.float32), axis=-1)  # CX / CY / W / H / CLASSES

        pixel_counts = tf.cast(
            tf.reduce_sum(crop_ins_outs, axis=(2, 3)), tf.float32) # 픽셀 갯수

        sizes = CalculateInstanceSize(default_road_size=self.default_road_size)([
            seg_outs, crop_ins_outs])
        instance_size, horizontal_size, vertical_size = tf.unstack(sizes, axis=-1)

        include_my_road = IncludeMyRoad()([seg_outs, crop_ins_outs])

        return tf.stack([
            classes, cx, cy, w, h, conf, pixel_counts, instance_size,
            horizontal_size, vertical_size, include_my_road], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'default_road_size': self.default_road_size
        })
        return config


class IncludeMyRoad(Layer):
    """ Myroad와 객체가 겹치는지를 확인하는 코드


    """
    def __init__(self, threshold=0.1, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        seg_outs = inputs[0]
        my_road = seg_outs[..., 1]
        my_road = tf.cast(my_road, tf.float32)

        crop_ins_outs = inputs[1]
        crop_ins_outs = tf.cast(crop_ins_outs, tf.float32)

        intersection = tf.logical_and((my_road > 0.5)[:, None],
                                      (crop_ins_outs > 0.5))
        intersection_area = tf.reduce_sum(
            tf.cast(intersection, tf.float32), axis=(2, 3))
        instance_area = tf.reduce_sum(tf.cast(crop_ins_outs > 0.5, tf.float32), axis=(2, 3))

        ioi = intersection_area / (instance_area + 1e-5)
        return tf.cast(ioi > tf.constant(self.threshold, tf.float32), tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            'threshold': self.threshold
        })
        return config


class CalculateInstanceSize(Layer):
    def __init__(self, default_road_size=3.25, **kwargs):
        self.default_road_size = default_road_size
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        seg_outs = inputs[0]
        pad_ins_outs = inputs[1]
        # >>> (batch size, num instance, height, width)

        unit_length_by_vertical = tf.map_fn(
            self._calculate_road_size_by_vertical_per_batch,
            seg_outs[..., 1], dtype=tf.float32)
        # >>> (batch size, height)

        unit_size_by_vertical = unit_length_by_vertical ** 2
        instance_size = tf.reduce_sum(
            unit_size_by_vertical[:, None, :, None] * pad_ins_outs,
            axis=(2, 3))
        # >>> (batch size, num instance)

        vertical_size = tf.reduce_sum(
            (unit_length_by_vertical[:, None, :] *
             tf.cast(tf.reduce_any(pad_ins_outs > 0.5, axis=-1), tf.float32)),
            axis=-1)
        # >>> (batch size, num instance)

        horizontal_size = tf.reduce_max(
            tf.reduce_sum(unit_length_by_vertical[:, None, :, None] *
                          pad_ins_outs, axis=2), axis=-1)
        # >>> (batch size, num instance)

        sizes = tf.stack([instance_size, horizontal_size, vertical_size], axis=-1)
        return sizes

    def _calculate_road_size_by_vertical_per_batch(self, image):
        indices = tf.where(image > 0)
        left_marginal, right_marginal = \
            self._calculate_marginal_x_by_y_axis(indices)

        left_theta = self._calculate_theta(left_marginal)
        right_theta = self._calculate_theta(right_marginal)

        y_arange = tf.cast(tf.range(0, tf.shape(image)[0]), tf.float32)
        pred_left = y_arange * left_theta[0] + left_theta[1]
        pred_right = y_arange * right_theta[0] + right_theta[1]

        width_by_vertical = tf.clip_by_value(
            pred_right - pred_left, 1., np.inf)
        width_by_vertical = tf.cast(width_by_vertical, tf.float32)
        return tf.cast(self.default_road_size, tf.float32) / width_by_vertical

    def _calculate_marginal_x_by_y_axis(self, indices):
        xs = indices[..., 1]
        ys = indices[..., 0]

        x_mins = tf.segment_min(xs, ys)
        x_maxs = tf.segment_max(xs, ys)
        y_pos = tf.range(0, tf.shape(x_mins)[0])
        y_pos = tf.cast(y_pos, tf.int64)
        left_marginal = tf.gather_nd(
            tf.stack([y_pos, x_mins], axis=-1),
            tf.where(tf.not_equal(x_mins, x_maxs)))
        right_marginal = tf.gather_nd(
            tf.stack([y_pos, x_maxs], axis=-1),
            tf.where(tf.not_equal(x_mins, x_maxs)))

        # 노이즈을 줄이기 위해 앞뒤 15% drop
        valid_counts = tf.cast(tf.shape(left_marginal)[0], tf.float32)
        drop_counts = tf.clip_by_value(
            tf.cast(valid_counts * 0.15, tf.int32), 1, 2 ** 31)

        left_marginal = tf.cast(
            left_marginal[drop_counts:-drop_counts], tf.float32)
        right_marginal = tf.cast(
            right_marginal[drop_counts:-drop_counts], tf.float32)
        return left_marginal, right_marginal

    def _calculate_theta(self, pos):
        xs = tf.stack([pos[:, 0], tf.ones_like(pos[:, 0])], axis=1)
        xs = tf.cast(xs, tf.float32)
        ys = tf.expand_dims(pos[:, 1], axis=1)
        ys = tf.cast(ys, tf.float32)

        x_mat = tf.transpose(xs) @ xs
        det_x = tf.linalg.det(x_mat)

        theta = tf.cond(det_x > 0.,
                        lambda: tf.linalg.inv(x_mat) @ (tf.transpose(xs) @ ys),
                        lambda: tf.zeros((2, 1), tf.float32))
        return theta

    def get_config(self):
        config = super().get_config()
        config.update({
            "default_road_size": self.default_road_size
        })
        return config