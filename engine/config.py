"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import argparse
import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class ModelConfiguration:
    class _PostProcess:
        """
        학습된 모델에 대해, 마지막 출력값들을 보정하는 Parameters
        """
        # 1. Input Resolution
        # > 학습 시 사용하였던 기준 Resolution
        # > 값이 커질수록 좀 더 정밀하게 처리할 수 있지만, 처리 속도는 느려짐
        # > (432, 768) ~ (648, 1152) 사이의 값 사이를 추천
        resolution = (540, 960)

        # 2. Detection Proposal
        min_confidence = 0.3      # 물체라 인식하는 최소 Confidence
        nms_iou_threshold = 0.4   # 동일 사물의 겹치는 비율(Intersection-Over-Union) 값. 이를 넘으면 동일한 사물로 파악
        post_iou_threshold = 0.6  # 다른 사물의 겹치는 비율(Intersection-Over-Union) 값. 이를 넘으면 같은 사물을 오분류한 것으로 파악
        nms_max_output_size = 100 # 화면 내에서 가능한 최대 물체의 갯수

        # 3. Semantic Segmentation
        smoothing_kernel_sizes = (0, 0, 0)  # 커질수록 경계가 부드럽게 이어지지만, 세밀한 특성을 놓침 (Other Road, My Road, Crack)
        smoothing_weights = (1., 1., 1.)  # Confidence 값을 증폭 혹은 감쇄 (Other Road, My Road, Crack)

        # 4. Visualization
        instance_colors = [[192,  32, 128], # car
                           [160,  96,   0], # bump
                           [ 96,   0, 128], # manhole
                           [ 32,  96, 192], # steel
                           [ 96,  32, 128]] # pothole
        instance_alpha = 0.3 # instance 색상에 대한 투명도

        semantic_colors = [[ 64,   0, 128], # other road
                           [128,  96,   0], # my road
                           [128, 192,   0]] # crack
        semantic_alpha = 0.3 # Semantic 색상에 대한 투명도

        # 5. Summarization
        default_road_size = 3.25

    class _BackBone:
        """
        Hyper Parameters Related On Backbone Network
        """
        backbone_type = 'resnet50'
        num_features = 128
        backbone_outputs = ('C3', 'C4', 'C5', 'P6', 'P7')

    class _Detection:
        """
        Hyper Parameters Related On Detection
        """
        # Prior Boxes의 세팅
        pr_scales = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]
        pr_ratios = [1 / 3, 1 / 2, 1, 2, 3]

        feature_pyramid_inputs = ('C3', 'C4', 'C5')
        num_features = 128
        num_depth = 4

        use_separable_conv = False
        expand_ratio = 4.
        use_squeeze_excite = False
        squeeze_ratio = 16
        groups = 16

        min_confidence = 0.5
        nms_iou_threshold = 0.4
        post_iou_threshold = 0.6
        nms_max_output_size = 100

    class _Instance:
        """
        Hyper Parameters Related On Instance Segmentation
        """
        # Mask Distribute 관련된 Hyper Parameter
        max_k = 2
        base_size = 36

        crop_size = (14, 14)

        num_features = 128
        num_depth = 4

        use_separable_conv = False
        expand_ratio = 4.
        use_squeeze_excite = False
        squeeze_ratio = 16
        groups = 16

    class _Semantic:
        """
        Hyper Parameters Related On Semantic Segmentation
        """
        num_aspp_features = 128
        atrous_rate = (6, 12, 18)
        atrous_groups = 16

        skip_input_name = 'C3'
        aspp_input_name = 'C5'

        num_features = 128
        num_skip_features = 32
        num_depth = 4

        use_separable_conv = False
        expand_ratio = 4.
        use_squeeze_excite = False
        squeeze_ratio = 16
        groups = 16

    class _Loss:
        """
        Hyper Parameters Related On Loss Networks
        """
        cls_loss_weight = 300
        cls_loss_alpha = 0.25
        cls_loss_gamma = 2.

        box_loss_weight = 1.
        box_loss_momentum = .9
        box_loss_beta = .11
        box_loss_use_adjust = True

        mask_loss_weight = 1e-2
        mask_loss_label_smoothing = 0.

        seg_loss_weight = .5
        seg_loss_label_smoothing = 0.

        min_confidence = 5e-2
        nms_iou_threshold = 0.6
        post_iou_threshold = 0.8
        nms_max_output_size = 100

    class _Dataset:
        """
        Hyper Parameters Related On Dataset
        """
        train_cases = [] # train image에 대한 파일이름 리스트
        valid_cases = [] # validation image에 대한 파일이름 리스트

        min_area = 1000.0
        instance_labels = ('car', 'bump', 'manhole', 'steel', 'pothole')
        semantic_labels = ('other_road', 'my_road', 'crack')
        except_semantic_labels = ('car', )

        data_dir = os.path.join(ROOT_DIR, "datasets/")

    class _Train:
        """
        Hyper Parameters Related On Train
        """
        save_dir = os.path.join(ROOT_DIR, "logs/")

        gpu_count = 2
        use_multiprocessing = True

        batch_size = 8
        max_batch_size = 32
        inference_batch_size = 1
        scale_ratio = (0.4, 0.6)

        train_head_tune = True
        train_head_level = 'C5'
        train_head_tune_epoch = 10
        head_base_lr = 1e-4
        head_max_lr = 1e-3
        head_step_size = 700

        train_waist_tune = True
        train_waist_level = 'C2'
        train_waist_tune_epoch = 10
        waist_base_lr = 1e-4
        waist_max_lr = 1e-3
        waist_step_size = 700

        train_all = True
        train_all_epoch = 30
        all_base_lr = 1e-5
        all_max_lr = 1e-4
        all_step_size = 700

    def to_dict(self):
        config = {}
        for attr_group in dir(self):
            config[attr_group] = {}
            attrs = self.__getattribute__(attr_group)
            for attr in dir(attrs):
                if "__" == attr[:2]:
                    continue
                config[attr_group][attr] = attrs.__getattribute__(attr)
        return config

    def from_dict(self, config_dict):
        for attr_group, attr_dict in config_dict.items():
            for key, value in attr_dict.items():
                (self
                 .__getattribute__(attr_group)
                 .__setattr__(key, value))

    def update(self, attr_group, key, value):
        self.__getattribute__(attr_group).__setattr__(key, value)

    def get_arg_parser(self, default_config=None):
        if default_config is None:
            default_config = self
        parser = argparse.ArgumentParser()
        for attr_group in dir(self):
            attrs = self.__getattribute__(attr_group)
            for attr in dir(attrs):
                if "__" == attr[:2]:
                    continue
                default_value = (default_config
                                 .__getattribute__(attr_group)
                                 .__getattribute__(attr))
                if isinstance(default_value,list) or isinstance(default_value,tuple):
                    parser.add_argument(f"-{attr_group}.{attr}",
                                        required=False,
                                        nargs='+',
                                        default=default_value,
                                        type=type(default_value))
                else:
                    parser.add_argument(f"-{attr_group}.{attr}",
                                        required=False,
                                        default=default_value,
                                        type=type(default_value))

        return parser.parse_args()

    postprocess = _PostProcess()
    backbone = _BackBone()
    detection = _Detection()
    instance = _Instance()
    semantic = _Semantic()
    loss = _Loss()
    dataset = _Dataset()
    train = _Train()

    def __dir__(self):
        return ["postprocess", "backbone", "detection", "instance",
                "semantic", "loss", "dataset", "train"]