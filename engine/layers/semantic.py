"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from engine.layers.misc import MobileSeparableConv2D, SqueezeExcite
from engine.layers.misc import ResizeLike
from engine.normalization import GroupNormalization

"""
Refinement Network For Semantic Segmentation

Atrous Spartial Pyramid Pooling, DeepLab V3+

각 Grid 별로 Multi-Scale을 고려할 수 있도록 다양한 크기의 rate로 정보를 읽어들인 후 concat하여 취합함

Reference : https://github.com/tensorflow/tensorflow/issues/6720#issuecomment-298190596
"DeepLab's Four Alignment Rules"
(1) Use of odd-sized kernels in all convolution and pooling ops.
(2) Use of SAME boundary conditions in all convolution and pooling ops.
(3) Use align_corners=True when upsampling feature maps with bilinear interpolation.
(4) Use of inputs with height/width equal to a multiple of the output_stride, plus one 
    (for example, when the CNN output stride is 8, use height or width equal to 8 * n + 1, 
     for some n, e.g., images HxW set to 321x513).
"""


class AtrousSeparableConv2D(Layer):
    """ Atrous Separable Convolution Layer
        Atrous convolution, a powerful tool that allows us to explicitly control the
        resolution of features computed by deep convolutional neural networks
        and adjust filter's field of view in order to capture multi-scale information,
        generalizes standard convolution operation.
        In the case of two-dimensional signals, for each location i on the output feature map y
        and a convolution filter w, atrous convolution is applied over the input feature map x as follows.

        y[i] = sum_k X[i+r*k]W[K]
        where the atrous rate r dtermines the stride with which we sample the input signal.


        Operation Order:
            Atrous & Depthwise Convolution ->
            Normalization ->
            Activation ->
            Pointwise Convolution ->
            Normalization ->
            Activation
    """

    def __init__(self, filters, dilation_rate=3,
                 groups=16, **kwargs):
        prefix = kwargs.get('name', 'AtrousSeparableConv2d')
        super().__init__(**kwargs)

        self.groups = groups
        self.filters = filters
        self.dilation_rate = dilation_rate

        self.depth_conv2d = DepthwiseConv2D((3, 3), dilation_rate=dilation_rate,
                                            padding='same', use_bias=False,
                                            name=prefix + '_depthwise')
        self.point_conv2d = Conv2D(self.filters, (1, 1), use_bias=False,
                                   name=prefix + '_pointwise')

        self.depth_norm = GroupNormalization(groups=self.groups, name=prefix + '_depthwise_GN')
        self.point_norm = GroupNormalization(groups=self.groups, name=prefix + '_pointwise_GN')

        self.depth_relu = ReLU(name=prefix+'_depthwise_relu')
        self.point_relu = ReLU(name=prefix+'_pointwise_relu')

    def call(self, inputs, **kwargs):
        x = self.depth_conv2d(inputs)
        x = self.depth_norm(x)
        x = self.depth_relu(x)

        x = self.point_conv2d(x)
        x = self.point_norm(x)
        x = self.point_relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters,
                       'dilation_rate': self.dilation_rate,
                       'groups': self.groups})
        return config


class ASPPNetwork(Layer):
    """
    Atrous Spartial Pyramid Pooling, or ASPP

    Perform spatial pyramid pooling at several grid scales (including images-level pooling)
    or apply several parallel atrous convolution with different rates.
    These engine have shown promising results on several segmentation benchmarks by
    exploiting the multi-scale information

    """
    def __init__(self, num_features=256,
                 atrous_rate=(6, 12, 18),
                 groups=16, **kwargs):
        self.num_features = num_features
        self.atrous_rate = atrous_rate
        self.groups = groups
        super().__init__(**kwargs)
        # ASPP 1x1 Branch
        self.aspp_1x1_branch = []
        self.aspp_1x1_branch.append(Conv2D(num_features, (1, 1),
                                           use_bias=False, name='aspp_1x1'))
        self.aspp_1x1_branch.append(GroupNormalization(groups=groups,
                                                       name='aspp_1x1_GN'))
        self.aspp_1x1_branch.append(ReLU(name='aspp_1x1_relu'))

        # ASPP Middle Branch
        self.aspp_branches = []
        for rate in atrous_rate:
            self.aspp_branches.append(
                AtrousSeparableConv2D(num_features, dilation_rate=rate,
                                      groups=groups, name=f'aspp_{rate}'))

        # ASPP Pooling Branch
        self.aspp_pool_conv = Conv2D(num_features, (1, 1),
                                     activation='relu',
                                     use_bias=False,
                                     name='aspp_pool')

        # ASPP Concat Layer
        self.concat_branch = []
        self.concat_branch.append(Conv2D(
            num_features, (1, 1), use_bias=False, name='concat_projection'))
        self.concat_branch.append(GroupNormalization(
            groups=groups, name='concat_projection_GN'))
        self.concat_branch.append(ReLU(name='concat_projection_relu'))

    def call(self, inputs, **kwargs):
        aspp_1x1 = inputs

        for layer in self.aspp_1x1_branch:
            aspp_1x1 = layer(aspp_1x1)

        aspp_rate_branches = []
        for layer in self.aspp_branches:
            aspp_rate_branches.append(layer(inputs))

        aspp_pool = tf.reduce_mean(inputs, axis=(1, 2), keepdims=True,
                                   name='global_average_pooling')
        aspp_pool = self.aspp_pool_conv(aspp_pool)
        aspp_pool = ResizeLike()(aspp_pool, target=inputs)

        aspp_concat = tf.concat([aspp_1x1, *aspp_rate_branches, aspp_pool],
                                axis=-1, name='concat_projections')
        for layer in self.concat_branch:
            aspp_concat = layer(aspp_concat)

        return aspp_concat

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "atrous_rate": self.atrous_rate,
            "groups": self.groups
        })
        return config


"""
Head Networks For Semantic Segmentation

DeepLab V3+ 형태로 구성된 모델
"""


class SegmentationSubNet(Layer):
    """ Semantic Segmentation SubNetwork
    DeepLab V3+ Decoder Style

    """
    def __init__(self, num_depth=2, num_features=256,
                 num_skip_features=48, num_classes=3,
                 use_separable_conv=False, expand_ratio=4.,
                 use_squeeze_excite=False, squeeze_ratio=16.,
                 groups=16, **kwargs):
        self.num_depth = num_depth
        self.num_features = num_features
        self.num_skip_features = num_skip_features
        self.num_classes = num_classes
        self.use_separable_conv = use_separable_conv
        self.expand_ratio = expand_ratio
        self.use_squeeze_excite = use_squeeze_excite
        self.squeeze_ratio = squeeze_ratio
        self.groups = groups
        super().__init__(kwargs)
        self.skip_layers = []
        self.skip_layers.append(Conv2D(self.num_skip_features, (1, 1), use_bias=False,
                                       name='skip_projection'))
        self.skip_layers.append(GroupNormalization(groups=groups, name='skip_projection_GN'))
        self.skip_layers.append(ReLU(name='skip_projection_relu'))

        self.block = []
        for i in range(self.num_depth):
            if self.use_squeeze_excite:
                layer = SqueezeExcite(self.squeeze_ratio)
                self.block.append(layer)
            if self.use_separable_conv:
                layer = MobileSeparableConv2D(self.num_features, (3, 3),
                                              expand_ratio=self.expand_ratio)
            else:
                layer = Conv2D(num_features, (3, 3), activation='relu', padding='same',
                               kernel_initializer=RandomNormal(stddev=0.01))
            self.block.append(layer)
            layer = GroupNormalization(self.groups)
            self.block.append(layer)

        self.output_layer = Conv2D(num_classes, (1, 1), activation='sigmoid')

    def call(self, inputs, **kwargs):
        dec_input, skip_dec_input = inputs[0], inputs[1]

        for skip_layer in self.skip_layers:
            skip_dec_input = skip_layer(skip_dec_input)
        upsampled = ResizeLike()(dec_input, target=skip_dec_input)
        dec_input = tf.concat([upsampled, skip_dec_input], axis=-1)

        for layer in self.block:
            dec_input = layer(dec_input)
        return self.output_layer(dec_input)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_depth": self.num_depth,
            "num_features": self.num_features,
            "num_skip_features": self.num_skip_features,
            "num_classes": self.num_classes,
            "use_separable_conv": self.use_separable_conv,
            "expand_ratio": self.expand_ratio,
            "use_squeeze_excite": self.use_squeeze_excite,
            "squeeze_ratio": self.squeeze_ratio,
            "groups": self.groups
        })
        return config


"""
Post Process For Semantic Segmentation

- SemanticSmoothing : 
    Erosion & Dilation 연산을 통해, Semantic Segmentation outputs의 출력 잡음을 보다 Smooth하게 만들어줌
    
- SemanticEnhancement : 
    Semantic Segmentation outputs의 값을 증폭하여, 결과가 보다 선명히 나오도록 도와줌
"""


class SemanticSmoothing(Layer):
    """ Label Prediction 후처리 알고리즘.
    Erosion을 통해 잘못 잡은 것들을 지워주고, Dilation을 통해 메꿔주는 형태.
    이를 통해 결과가 부드러운 형태로 나오게 된다.
    """
    def __init__(self, kernel_size=10, weight=1., **kwargs):
        self.kernel_size = kernel_size
        self.weight = weight
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if self.kernel_size > 0:
            n_classes = inputs.get_shape().as_list()[-1]
            kernel = tf.zeros([self.kernel_size, self.kernel_size, n_classes],
                              tf.float32)
            eroded = tf.nn.erosion2d(inputs, kernel,
                                     strides=[1, 1, 1, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='SAME')
            outputs = tf.nn.dilation2d(eroded, kernel,
                                    strides=[1, 1, 1, 1],
                                    rates=[1, 1, 1, 1],
                                    padding='SAME')
            return outputs * tf.cast(self.weight, tf.float32)
        else:
            return inputs * tf.cast(self.weight, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "weight": self.weight
        })
        return config


"""
Layers Related On Training

학습할 Mask 정보를 Model의 출력 형태에 맞춰 배정하는 Layer

"""


class AssignSeg(Layer):
    """ Semantic Segmentation의 학습 Input Data를 Prediction Shape와 똑같이 맞추어주는 역할
    """
    def call(self, inputs, **kwargs):
        seg_true = inputs[0]
        seg_pred = inputs[1]
        resized_seg_true = ResizeLike()(seg_true, target=seg_pred)
        return tf.round(resized_seg_true)


__all__ = [
    "AtrousSeparableConv2D",
    "ASPPNetwork",
    "SegmentationSubNet",
    "SemanticSmoothing",
    "AssignSeg"]

