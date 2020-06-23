"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNet
from engine.normalization import GroupNormalization
from engine.layers import Identity
import tensorflow as tf
from keras_applications.resnet_v2 import ResNet101V2
from thirdparty import Classifiers
from .ResNext import ResNeXt50
from efficientnet import tfkeras as efn


class BackBonePreProcess(Layer):
    """
    Image Input Preprocess하는 모듈

    BackBone Network마다 입력 형태가 다르다. BGR 채널로 학습된 모델이 있고, 이미지 평균값을 빼주거나,
    normalize를 했냐 안했냐 등 입력 형태가 다르다. 모델에 맞게 이미지를 전처리한다.

    * rgb : True이면 RGB 순서, False이면 BGR 순서
    * mean_shift: True이면 [123.68, 116.779, 103.939] 빼줌
    * normalize :
        - 0 : Un-Normalized
        - 1 : [0,1]
        - 2 : [-1,1]
        - 3 : Standardarzation

    inputs:
        - images : [batch size, height, width, channel]
          RGB 순서로 되어 있고, 값의 범위가 [0,255]인 이미지
    Outputs:
        - images : [batch size, height, width, channel]


    """
    def __init__(self, rgb=True, mean_shift=False, normalize=0, **kwargs):
        self.rgb = rgb
        self.mean_shift = mean_shift
        self.normalize = normalize
        super().__init__(**kwargs)
        if self.rgb:
            self.mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
            self.std = tf.constant([0.225, 0.224, 0.229], dtype=tf.float32)
        else:
            self.mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
            self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def call(self, inputs, **kwargs):
        if not self.rgb:
            inputs = inputs[...,::-1]

        if self.mean_shift:
            inputs = inputs - self.mean

        if self.normalize == 1:
            return inputs / 255.
        elif self.normalize == 2:
            if self.mean_shift:
                return inputs / 127.5
            else:
                return inputs / 127.5 - 1.
        elif self.normalize == 3:
            inputs = inputs / 255.
            return inputs / self.std
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "rgb": self.rgb,
            "mean_shift": self.mean_shift,
            "normalize": self.normalize
        })
        return config


"""
Load Pretrained Model

현재 Pretrained Model 중에서 이용가능한 리스트는 다음과 같습니다.

- resnet50
- resnet50v2
- resnet101v2
- seresnet34
- seresnet50
- seresnext50
- resnext50
- vgg16
- mobilenet
"""


BACKBONE_LAYERS = {
    "resnet50": {
        "C1": 'activation',
        "C2": 'activation_9',
        "C3": "activation_21",
        "C4": "activation_39",
        "C5": "activation_48"
    },
    "resnet50v2": {
        "C1": "conv1_conv",
        "C2": "conv2_block3_preact_relu",
        "C3": "conv3_block4_preact_relu",
        "C4": "conv4_block4_preact_relu",
        "C5": "post_relu"
    },
    "resnet101v2": {
        "C1": "conv1_conv",
        "C2": "conv2_block3_1_relu",
        "C3": "conv3_block4_1_relu",
        "C4": "conv4_block23_1_relu",
        "C5": "post_relu"
    },
    "seresnet34": {
        "C1": "relu0",
        "C2": "stage2_unit1_relu1",
        "C3": "stage3_unit1_relu1",
        "C4": "stage4_unit1_relu1",
        "C5": "relu1",
    },
    "seresnet50": {
        "C1": "activation",
        "C2": "activation_15",
        "C3": "activation_35",
        "C4": "activation_65",
        "C5": "activation_80"
    },
    "seresnext50": {
        "C1": "activation",
        "C2": "activation_16",
        "C3": "activation_36",
        "C4": "activation_66",
        "C5": "activation_80"
    },
    "resnext50": {
        "C1": 'conv1_relu',
        "C2": 'conv2_block3_out',
        "C3": "conv3_block4_out",
        "C4": "conv4_block6_out",
        "C5": "conv5_block3_out"
    },
    "vgg16": {
        "C1": 'block2_conv2',
        "C2": 'block3_conv3',
        "C3": "block4_conv3",
        "C4": "block5_conv3",
        "C5": "block5_pool"
    },
    "mobilenet": {
        "C1": "conv_pw_1_relu",
        "C2": "conv_pw_3_relu",
        "C3": "conv_pw_5_relu",
        "C4": "conv_pw_11_relu",
        "C5": "conv_pw_13_relu"
    },
    "efficientnetb2": {
        "C1": "block2a_expand_activation",
        "C2": "block3a_expand_activation",
        "C3": "block4a_expand_activation",
        "C4": "block6a_expand_activation",
        "C5": "top_activation",
    },
    "efficientnetb3": {
        "C1": "block2a_expand_activation",
        "C2": "block3a_expand_activation",
        "C3": "block4a_expand_activation",
        "C4": "block6a_expand_activation",
        "C5": "top_activation",
    }
}


def load_backbone(backbone_type="resnet50",
                  backbone_outputs=('C3', 'C4', 'C5', 'P6', 'P7'),
                  num_features=256):
    global BACKBONE_LAYERS
    inputs = Input((None, None, 3), name='images')
    if backbone_type.lower() == 'resnet50':
        preprocess = BackBonePreProcess(rgb=False,
                                        mean_shift=True,
                                        normalize=0)(inputs)
        model = ResNet50(input_tensor=preprocess,
                         include_top=False)
    elif backbone_type.lower() == 'resnet50v2':
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=True,
                                        normalize=2)(inputs)
        resnet50v2, _ = Classifiers.get('resnet50v2')
        model = resnet50v2(input_tensor=preprocess,
                           include_top=False,
                           weights='imagenet')
    elif backbone_type.lower() == "resnet101v2":
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=False,
                                        normalize=2)(inputs)
        model = ResNet101V2(input_tensor=preprocess,
                            include_top=False,
                            backend=tf.keras.backend,
                            layers=tf.keras.layers,
                            models=tf.keras.models,
                            utils=tf.keras.utils)
    elif backbone_type.lower() == 'resnext50':
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=True,
                                        normalize=2)(inputs)
        model = ResNeXt50(input_tensor=preprocess,
                          include_top=False)
    elif backbone_type.lower() == "seresnet50":
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=True,
                                        normalize=3)(inputs)
        seresnet50, _ = Classifiers.get('seresnet50')
        model = seresnet50(input_tensor=preprocess,
                           original_input=inputs,
                           include_top=False,
                           weights='imagenet')
    elif backbone_type.lower() == "seresnet34":
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=False,
                                        normalize=0)(inputs)
        seresnet34, _ = Classifiers.get('seresnet34')
        model = seresnet34(input_tensor=preprocess,
                           original_input=inputs,
                           include_top=False,
                           weights='imagenet')
    elif backbone_type.lower() == "seresnext50":
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=True,
                                        normalize=3)(inputs)
        seresnext50, _ = Classifiers.get('seresnext50')
        model = seresnext50(input_tensor=preprocess,
                            original_input=inputs,
                            include_top=False,
                            weights='imagenet')
    elif backbone_type.lower() == "vgg16":
        preprocess = BackBonePreProcess(rgb=False,
                                        mean_shift=True,
                                        normalize=0)(inputs)
        model = VGG16(input_tensor=preprocess,
                      include_top=False)
    elif backbone_type.lower() == "mobilenet":
        preprocess = BackBonePreProcess(rgb=False,
                                        mean_shift=False,
                                        normalize=2)(inputs)
        model = MobileNet(input_tensor=preprocess,
                          include_top=False, alpha=1.0)
    elif backbone_type.lower() == 'efficientnetb2':
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=True,
                                        normalize=3)(inputs)
        model = efn.EfficientNetB2(input_tensor=preprocess,
                                   include_top=False,
                                   weights='imagenet')
    elif backbone_type.lower() == 'efficientnetb3':
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=True,
                                        normalize=3)(inputs)
        model = efn.EfficientNetB3(input_tensor=preprocess,
                                   include_top=False,
                                   weights='imagenet')
    elif backbone_type.lower() == 'efficientnetb4':
        preprocess = BackBonePreProcess(rgb=True,
                                        mean_shift=True,
                                        normalize=3)(inputs)
        model = efn.EfficientNetB4(input_tensor=preprocess,
                                   include_top=False,
                                   weights='imagenet')
    else:
        raise NotImplementedError(
            f"backbone_type은 {BACKBONE_LAYERS.keys()} 중에서 하나가 되어야 합니다.")
    model.trainable = False

    # Block Layer 가져오기
    features = []
    for key, layer_name in BACKBONE_LAYERS[backbone_type.lower()].items():
        if key in backbone_outputs:
            layer_tensor = model.get_layer(layer_name).output
            features.append(Identity(name=key)(layer_tensor))

    if backbone_type.lower() == "mobilenet":
        # Extra Layer for Feature Extracting
        Z6 = ZeroPadding2D(((0, 1), (0, 1)),name=f'P6_zeropadding')(features[-1])
        P6 = Conv2D(num_features, (3, 3), strides=(2, 2),
                    padding='valid', activation='relu', name=f'P6_conv')(Z6)
        if 'P6' in backbone_outputs:
            features.append(Identity(name='P6')(P6))
        G6 = GroupNormalization(name=f'P6_norm')(P6)
        Z7 = ZeroPadding2D(((0, 1), (0, 1)), name=f'P7_zeropadding')(G6)
        P7 = Conv2D(num_features, (3, 3), strides=(2, 2),
                    padding='valid', activation='relu', name=f'P7_conv')(Z7)
        if 'P7' in backbone_outputs:
            features.append(Identity(name=f'P7')(P7))
    else:
        P6 = Conv2D(num_features, (3, 3), strides=(2, 2),
                    padding='same', activation='relu', name=f'P6_conv')(features[-1])
        if 'P6' in backbone_outputs:
            features.append(Identity(name=f'P6')(P6))
        G6 = GroupNormalization(name=f'P6_norm')(P6)
        P7 = Conv2D(num_features, (3, 3), strides=(2, 2),
                    padding='same', activation='relu', name=f'P7_conv')(G6)
        if 'P7' in backbone_outputs:
            features.append(Identity(name=f'P7')(P7))

    return Model(inputs, features, name=backbone_type)


def freeze_backbone(model, model_type, freeze_depth="C5"):
    """
    backbone_network에서 freeze_depth 아래의 Layer가 학습하지 못하도록 지정하는 Method.

    !!Caution!! :
    freeze_backbone 후, backbone_network.compile()을 해주어야 실질적으로 freeze의 동작이 진행됩니다.

    :param model:
        build_base_network() method를 통해 만들어진 backbone_network
    :param freeze_depth:
        Backbone Network에서의 Depth. 작을수록 입력층에 가까워짐.
        ['C0','C1','C2','C3','C4','C5'] 중에서 선택하면 됨.
    :return:
        Backbone Network
    """
    if freeze_depth != "C0":
        layer_blocks = BACKBONE_LAYERS[model_type]
        assert freeze_depth in layer_blocks, f"{layer_blocks}안의 이름으로 지정해 주세요"
        train_flag = False
        for layer in model.layers:
            layer.trainable = train_flag
            if layer.name == layer_blocks[freeze_depth]:
                    train_flag = True
    else:
        for layer in model.layers:
            layer.trainable = True
    return model
