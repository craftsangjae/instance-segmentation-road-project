"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import time
import re
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Concatenate
from engine.utils import MaskLabDataset
from engine.prior import PriorBoxes
from engine.losses import ClassLoss, BoxLoss, SegLoss, MaskLoss
from engine.metrics import DetectionIOUMetric, ClassBinaryIOU
from engine.layers import *
from engine import backbone
from engine.config import ModelConfiguration


def build_backbone_network(configuration:ModelConfiguration):
    """
    Configuration에 따라 Backbone Network를 구성하는 Method

    :param configuration:
    :return:
    """
    config = configuration.backbone

    model = backbone.load_backbone(config.backbone_type,
                                   config.backbone_outputs,
                                   config.num_features)
    print("Backbone Network Summary")
    print("------------------------")
    print(f"* Backbone Type : {config.backbone_type}")
    print(f"* Backbone outputs : {config.backbone_outputs}")
    print(f"* Num features of Backbone Additional Layers: {config.num_features}")
    print("------------------------\n")
    return model


def build_detection_network(configuration:ModelConfiguration):
    config = configuration.detection
    num_backbone_outputs = len(configuration.backbone.backbone_outputs)
    num_classes = len(configuration.dataset.instance_labels)

    # Output Block의 이름은 Stride와 관련이 있습니다. (stride=2인 Layer의 갯수와 동일)
    prior_strides = [2**int(output_name[-1])
                     for output_name in configuration.backbone.backbone_outputs]
    prior_sizes = [4 * stride for stride in prior_strides]

    prior = PriorBoxes(strides=prior_strides,
                       sizes=prior_sizes,
                       pr_scales=config.pr_scales,
                       pr_ratios=config.pr_ratios)
    prior_subnet = PriorLayer(prior)

    print("Prior Network Summary")
    print("-------------------------------")
    print(f"* Strides of prior : {prior_strides}")
    print(f"* Sizes of prior : {prior_sizes}")
    print(f"* width/height scales of prior : {prior.pr_scales}")
    print(f"* width/height ratios of prior : {prior.pr_ratios}")
    print("-------------------------------\n")

    # C1, C2, C3, C4, C5, P6, P7
    assert len(
        set(config.feature_pyramid_inputs)
        - {'C1', 'C2', 'C3', 'C4', 'C5', 'P6', 'P7'}) == 0, "feature pyramid inputs은 C1~C5, P6, P7으로 이루어져야 합니다."
    feature_pyramid_strides = [
        2 ** int(name[1]) for name in config.feature_pyramid_inputs]
    fpn_subnet = FeaturePyramid(strides=feature_pyramid_strides,
                                num_features=config.num_features)
    print("Feature Pyramid Network Summary")
    print("-------------------------------")
    print(f"* Feature Pyramid Inputs  : {config.feature_pyramid_inputs}")
    print(f"* Num Features of Feature Pyramid : {config.num_features}")
    print("-------------------------------\n")

    cls_subnet = ClassificationSubNet(num_blocks=num_backbone_outputs,
                                      num_classes=num_classes,
                                      num_depth=config.num_depth,
                                      num_features=config.num_features,
                                      num_priors=len(prior),
                                      use_separable_conv=config.use_separable_conv,
                                      expand_ratio=config.expand_ratio,
                                      use_squeeze_excite=config.use_squeeze_excite,
                                      squeeze_ratio=config.squeeze_ratio,
                                      groups=config.groups)

    loc_subnet = BoxRegressionSubNet(num_blocks=num_backbone_outputs,
                                     num_depth=config.num_depth,
                                     num_features=config.num_features,
                                     num_priors=len(prior),
                                     use_separable_conv=config.use_separable_conv,
                                     expand_ratio=config.expand_ratio,
                                     use_squeeze_excite=config.use_separable_conv,
                                     squeeze_ratio=config.squeeze_ratio,
                                     groups=config.groups)
    print("Detection Head Networks Summary")
    print("-------------------------------")
    print(f"* Num Classes of Detection Classes : {num_classes}")
    print(f"* Num Depth of Sub Networks : {config.num_depth}")
    print(f"* Num Features of Sub Networks : {config.num_features}")
    print(f"* Use Separable Conv : {config.use_separable_conv}")
    if config.use_separable_conv:
        print(f"* Expand Ratio in Separable Conv : {config.expand_ratio}")
    print(f"* Use Squeeze Excite : {config.use_squeeze_excite}")
    if config.use_squeeze_excite:
        print(f"* squeeze ratio in Squeeze Excite: {config.squeeze_ratio}")
    print(f"* Num Groups of Group Normalization : {config.groups}")
    print("-------------------------------\n")

    return prior_subnet, fpn_subnet, cls_subnet, loc_subnet


def build_instance_network(configuration:ModelConfiguration):
    num_classes = len(configuration.dataset.instance_labels)
    config = configuration.instance
    restore_subnet = RestoreBoxes()

    distribute_subnet = MaskDistribute(max_k=config.max_k,
                                       base_size=config.base_size)
    print("MaskDistribute Network Summary")
    print("-------------------------------")
    print(f"* max_k      : {config.max_k}")
    print(f"* base_size  : {config.base_size}")
    print("-------------------------------\n")

    pyramid_roi_align = PyramidRoiAlign(crop_size=config.crop_size)
    print("Pyramid Roi Align Network Summary")
    print("-------------------------------")
    print(f"* Crop size : {config.crop_size}")
    print("-------------------------------\n")

    mask_subnet = MaskSubNet(num_blocks=config.max_k+1,
                             num_classes=num_classes,
                             num_depth=config.num_depth,
                             num_features=config.num_features,
                             use_separable_conv=config.use_separable_conv,
                             expand_ratio=config.use_separable_conv,
                             use_squeeze_excite=config.use_squeeze_excite,
                             squeeze_ratio=config.squeeze_ratio,
                             groups=config.groups)
    print("Instance Head Network Summary")
    print("-------------------------------")
    print(f"* Num Classes of Instance Classes : {num_classes}")
    print(f"* Num Depth of Head Network : {config.num_depth}")
    print(f"* Num Features of Head Network : {config.num_features}")
    print(f"* Use Separable Conv : {config.use_separable_conv}")
    if config.use_separable_conv:
        print(f"* Expand Ratio in Separable Conv : {config.expand_ratio}")
    print(f"* Use Squeeze Excite : {config.use_squeeze_excite}")
    if config.use_squeeze_excite:
        print(f"* squeeze ratio in Squeeze Excite: {config.squeeze_ratio}")
    print(f"* Num Groups of Group Normalization : {config.groups}")
    print("-------------------------------\n")

    return restore_subnet, distribute_subnet, pyramid_roi_align, mask_subnet


def build_semantic_network(configuration:ModelConfiguration):
    config = configuration.semantic
    num_classes = len(configuration.dataset.semantic_labels)

    aspp_subnet = ASPPNetwork(num_features=config.num_aspp_features,
                              atrous_rate=config.atrous_rate,
                              groups=config.atrous_groups)
    print("Atrous Spatial Pyramid Pooling Network Summary")
    print("-------------------------------")
    print(f"* num features : {config.num_aspp_features}")
    print(f"* Atrous Rates : {config.atrous_rate}")
    print(f"* groups : {config.atrous_groups}")
    print("-------------------------------\n")

    seg_subnet = SegmentationSubNet(num_depth=config.num_depth,
                                    num_features=config.num_features,
                                    num_skip_features=config.num_skip_features,
                                    num_classes=num_classes,
                                    use_separable_conv=config.use_separable_conv,
                                    expand_ratio=config.use_separable_conv,
                                    use_squeeze_excite=config.use_squeeze_excite,
                                    squeeze_ratio=config.squeeze_ratio,
                                    groups=config.groups)
    print("Semantic Head Networks Summary")
    print("-------------------------------")
    print(f"* Num Classes of Semantic Classes : {num_classes}")
    print(f"* Num Depth of Head Network : {config.num_depth}")
    print(f"* Num Features of Head Network : {config.num_features}")
    print(f"* Num Skip Features of Head Network : {config.num_skip_features}")
    print(f"* Use Separable Conv : {config.use_separable_conv}")
    if config.use_separable_conv:
        print(f"* Expand Ratio in Separable Conv : {config.expand_ratio}")
    print(f"* Use Squeeze Excite : {config.use_squeeze_excite}")
    if config.use_squeeze_excite:
        print(f"* squeeze ratio in Squeeze Excite: {config.squeeze_ratio}")
    print(f"* Num Groups of Group Normalization : {config.groups}")
    print("-------------------------------\n")

    return aspp_subnet, seg_subnet


def construct_masklab_networks(config:ModelConfiguration):
    K.clear_session()
    backbone_network = build_backbone_network(config)
    detection_networks = build_detection_network(config)
    instance_networks = build_instance_network(config)
    semantic_networks = build_semantic_network(config)

    trainer = construct_trainer_network(
        configuration=config, backbone_network=backbone_network,
        detection_networks=detection_networks,
        semantic_networks=semantic_networks,
        instance_networks=instance_networks)

    inference = construct_inference_network(
        configuration=config, backbone_network=backbone_network,
        detection_networks=detection_networks,
        semantic_networks=semantic_networks,
        instance_networks=instance_networks)

    return trainer, inference


def construct_trainer_network(configuration:ModelConfiguration,
                              backbone_network,
                              detection_networks=None,
                              semantic_networks=None,
                              instance_networks=None):
    config = configuration.loss

    trainer_inputs = []
    trainer_outputs = []
    # (1) Build Backbone Networks
    inputs = backbone_network.input
    trainer_inputs.append(inputs)

    # (2) Build Detection Network
    if detection_networks is not None:
        det_config = configuration.detection
        det_num_classes = len(configuration.dataset.instance_labels)
        prior_subnet, fpn_subnet, cls_subnet, loc_subnet = detection_networks
        pr_boxes = prior_subnet(inputs)

        ## 1) Connect to Feature Pyramid Network
        fpn_inputs = [backbone_network.outputs[idx]
                      for idx, name in enumerate(backbone_network.output_names)
                      if name in det_config.feature_pyramid_inputs]
        without_fpn_outputs = [backbone_network.outputs[idx]
                               for idx, name in enumerate(backbone_network.output_names)
                               if name not in det_config.feature_pyramid_inputs]
        fpn_outputs = fpn_subnet(fpn_inputs)
        feature_outputs = fpn_outputs + without_fpn_outputs

        ## 2) Connect to two Head Networks
        cls_pred = cls_subnet(feature_outputs)
        loc_pred = loc_subnet(feature_outputs)

        ## 3) assign ground truth
        gt_boxes = Input((None, 6,), name='gt_boxes')
        trainer_inputs.append(gt_boxes)
        gt_boxes_exist = Input((det_num_classes,),
                               name='gt_boxes_exist')
        trainer_inputs.append(gt_boxes_exist)
        cls_true, loc_true, assign_mask = AssignBoxes(
            num_classes=det_num_classes)([gt_boxes, pr_boxes])

        ## 4) Calculate Loss
        cls_loss = ClassLoss(weight=config.cls_loss_weight,
                             alpha=config.cls_loss_alpha,
                             gamma=config.cls_loss_gamma,
                             name='class_loss')(
            [cls_true, cls_pred, assign_mask, gt_boxes_exist])
        box_loss = BoxLoss(weight=config.box_loss_weight,
                           momentum=config.box_loss_momentum,
                           beta=config.box_loss_beta,
                           use_adjust=config.box_loss_use_adjust,
                           name='box_loss')(
            [loc_true, loc_pred, assign_mask])
        print("Detection Losses Summary")
        print("-------------------------------")
        print(f"* Classification Loss Weight: {config.cls_loss_weight}")
        print(f"* Classification Loss alpha: {config.cls_loss_alpha}")
        print(f"* Classification Loss gamma: {config.cls_loss_gamma}")

        print(f"* Localization Loss Weight: {config.box_loss_weight}")
        print(f"* Localization Use adjusted Smoothing L1: {config.box_loss_use_adjust}")
        if config.box_loss_use_adjust:
            print(f"* Localization Loss momentum: {config.box_loss_momentum}")
            print(f"* Localization Loss beta: {config.box_loss_beta}")
        print("-------------------------------\n")
        trainer_outputs.extend([cls_loss, box_loss])

        ## 5) Calculate Metric
        det_config = configuration.detection
        restored_boxes = RestoreBoxes()([loc_pred, pr_boxes])
        proposed_boxes = DetectionProposal(min_confidence=det_config.min_confidence,
                                           nms_iou_threshold=det_config.nms_iou_threshold,
                                           post_iou_threshold=det_config.post_iou_threshold,
                                           nms_max_output_size=det_config.nms_max_output_size,
                                           max_batch_size=configuration.train.max_batch_size)(
            [cls_pred, restored_boxes, inputs])
        precision, recall, fmeasure = DetectionIOUMetric()([
            proposed_boxes, gt_boxes])
        precision = Identity(name='detection_precision_metric')(precision)
        recall = Identity(name='detection_recall_metric')(recall)
        fmeasure = Identity(name='detection_fmeasure_metric')(fmeasure)
        trainer_outputs.extend([precision, recall, fmeasure])

        # (3) Build Instance Segmentation Network
        if instance_networks is not None:
            restore_layer, distribute_layer, pyramid_roi_align, mask_subnet = instance_networks

            ## 1) propose Available boxes
            restored_boxes = restore_layer([loc_pred, pr_boxes])
            proposed_boxes = DetectionProposal(min_confidence=config.min_confidence,
                                               nms_iou_threshold=config.nms_iou_threshold,
                                               post_iou_threshold=config.post_iou_threshold,
                                               nms_max_output_size=config.nms_max_output_size,
                                               max_batch_size=configuration.train.max_batch_size)(
                [cls_pred, restored_boxes, inputs])
            ## 2) Distribute to proper Feature Map
            chosen_boxes = Concatenate(axis=1)([gt_boxes, proposed_boxes])
            dist_boxes = distribute_layer(chosen_boxes)
            ## 3) Crop feature Map
            roi_fmaps, roi_boxes = pyramid_roi_align(
                [feature_outputs[:configuration.instance.max_k+1], dist_boxes, inputs])
            ## 4) Connect to Head Network
            roi_masks = mask_subnet(roi_fmaps)
            ## 5) Assign Ground Truth
            gt_masks = Input(shape=(None, None, None,), name='gt_masks')
            trainer_inputs.append(gt_masks)
            match_gt_masks = AssignMasks()([
                roi_boxes, roi_masks, gt_boxes, gt_masks])
            ## 6) Calculate Loss
            mask_loss = MaskLoss(weight=config.mask_loss_weight,
                                 label_smoothing=config.mask_loss_label_smoothing,
                                 name='mask_loss')(
                [match_gt_masks, roi_masks])
            print("Instance Loss Summary")
            print("-------------------------------")
            print(f"* Instance Loss Weight: {config.mask_loss_weight}")
            print(f"* Instance Loss Label Smoothing: {config.mask_loss_label_smoothing}")
            print("-------------------------------\n")

            trainer_outputs.append(mask_loss)

    # (4) Build Semantic Segmentation Network
    if semantic_networks is not None:
        sem_config = configuration.semantic
        sem_num_classes = len(configuration.dataset.semantic_labels)
        aspp_subnet, seg_subnet = semantic_networks

        ## 1) Connect to ASPP Networks
        skip_input = [backbone_network.outputs[idx]
                      for idx, name in enumerate(backbone_network.output_names)
                      if name == sem_config.skip_input_name][0]
        aspp_input = [backbone_network.outputs[idx]
                      for idx, name in enumerate(backbone_network.output_names)
                      if name == sem_config.aspp_input_name][0]
        aspp_outputs = aspp_subnet(aspp_input)

        ## 2) Connect to Head Networks
        seg_pred = seg_subnet([aspp_outputs, skip_input])

        ## 3) Assign Ground truth
        gt_seg = Input((None, None, sem_num_classes),
                       name='gt_seg')
        trainer_inputs.append(gt_seg)
        gt_seg_exist = Input((sem_num_classes, ),
                             name='gt_seg_exist')
        trainer_inputs.append(gt_seg_exist)
        seg_assigned = AssignSeg()([gt_seg, seg_pred])

        ## 4) Calculate Loss
        seg_loss = SegLoss(weight=config.seg_loss_weight,
                           label_smoothing=config.seg_loss_label_smoothing,
                           name='seg_loss')([seg_assigned, seg_pred, gt_seg_exist])
        print("Semantic Loss Summary")
        print("-------------------------------")
        print(f"* Semantic Loss Weight: {config.seg_loss_weight}")
        print(f"* Semantic Loss Label Smoothing: {config.seg_loss_label_smoothing}")
        print("-------------------------------\n")
        trainer_outputs.append(seg_loss)

        ## 5) Calculate Metric
        other_road_iou, my_road_iou, crack_iou = \
            ClassBinaryIOU(0.5, name='class_iou_metric')(
                [seg_assigned, seg_pred])
        other_road_iou = Identity(name='other_road_iou_metric')(other_road_iou)
        my_road_iou = Identity(name='my_road_metric')(my_road_iou)
        crack_iou = Identity(name='crack_iou_metric')(crack_iou)
        trainer_outputs.extend([other_road_iou, my_road_iou, crack_iou])

    return Model(trainer_inputs,
                 trainer_outputs,
                 name='trainer')


def construct_masklabdataset(configuration:ModelConfiguration):
    d_config = configuration.dataset
    trainset = MaskLabDataset(d_config.train_cases,
                              min_area=d_config.min_area,
                              data_dir=d_config.data_dir,
                              instance_labels=d_config.instance_labels,
                              semantic_labels=d_config.semantic_labels,)
    validset = MaskLabDataset(d_config.valid_cases,
                              min_area=d_config.min_area,
                              data_dir=d_config.data_dir,
                              instance_labels=d_config.instance_labels,
                              semantic_labels=d_config.semantic_labels)
    print("Dataset Summary")
    print("-------------------------------")
    print(f"* Num of Train Images : {len(d_config.train_cases)}")
    print(f"* Num of Valid Images : {len(d_config.valid_cases)}")
    print(f"* Num of Images : {len(d_config.train_cases)+len(d_config.valid_cases)}")
    print("-------------------------------\n")

    return trainset, validset


def construct_inference_network(configuration:ModelConfiguration,
                                backbone_network,
                                detection_networks=None,
                                semantic_networks=None,
                                instance_networks=None):
    inference_outputs = []
    # (1) Build Backbone Networks
    inputs = backbone_network.input
    inference_inputs = inputs

    # (2) Build Detection Network
    if detection_networks is not None:
        det_config = configuration.detection
        prior_subnet, fpn_subnet, cls_subnet, loc_subnet = detection_networks
        pr_boxes = prior_subnet(inputs)

        ## 1) Connect to Feature Pyramid Network
        fpn_inputs = [backbone_network.outputs[idx]
                      for idx, name in enumerate(backbone_network.output_names)
                      if name in det_config.feature_pyramid_inputs]
        without_fpn_outputs = [backbone_network.outputs[idx]
                               for idx, name in enumerate(backbone_network.output_names)
                               if name not in det_config.feature_pyramid_inputs]
        fpn_outputs = fpn_subnet(fpn_inputs)
        feature_outputs = fpn_outputs + without_fpn_outputs

        ## 2) Connect to two Head Networks
        cls_pred = cls_subnet(feature_outputs)
        loc_pred = loc_subnet(feature_outputs)
        inference_outputs.append(cls_pred)
        inference_outputs.append(loc_pred)

        # (3) Build Instance Segmentation Network
        # Instance Segmentation은 Detection Network가 구성되어 있어야만 가능
        if instance_networks is not None:

            restore_subnet, distribute_subnet, pyramid_roi_align, mask_subnet = instance_networks

            restored_boxes = restore_subnet([loc_pred, pr_boxes])
            proposed_boxes = DetectionProposal(
                min_confidence=det_config.min_confidence,
                nms_iou_threshold=det_config.nms_iou_threshold,
                post_iou_threshold=det_config.post_iou_threshold,
                nms_max_output_size=det_config.nms_max_output_size,
                max_batch_size=configuration.train.inference_batch_size
                )(
                [cls_pred, restored_boxes, inputs])
            dist_boxes = distribute_subnet(proposed_boxes)
            roi_fmaps, roi_boxes = pyramid_roi_align(
                [feature_outputs[:configuration.instance.max_k+1], dist_boxes, inputs])
            roi_masks = mask_subnet(roi_fmaps)

            inference_outputs.append(roi_boxes)
            inference_outputs.append(roi_masks)

    # (4) Build Semantic Segmentation Network
    if semantic_networks is not None:
        seg_config = configuration.semantic
        aspp_subnet, seg_subnet = semantic_networks

        ## 1) Connect to ASPP Networks
        skip_input = [backbone_network.outputs[idx]
                      for idx, name in enumerate(backbone_network.output_names)
                      if name == seg_config.skip_input_name][0]
        aspp_input = [backbone_network.outputs[idx]
                      for idx, name in enumerate(backbone_network.output_names)
                      if name == seg_config.aspp_input_name][0]
        aspp_outputs = aspp_subnet(aspp_input)

        ## 2) Connect to Head Networks
        seg_pred = seg_subnet([aspp_outputs, skip_input])
        inference_outputs.append(seg_pred)

    return Model(inference_inputs,
                 inference_outputs,
                 name='inference')


def load_masklab_inference_model_from_h5(save_path, config:ModelConfiguration,
                                         serving=False):
    """
    h5 file 포맷으로 저장된 Weigth들을 불러와서 RetinaMaskLab Model을 Inference할 수 있도록 수정한 모델

    """
    K.clear_session()
    s = time.time()
    print("weight을 .h5로부터 불러오는 중...")
    model = load_model(save_path, compile=False)
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    # (1) Get Backbone Network
    s = time.time()
    print("backbone Network를 재구성하는 중...")
    backbone_inputs = model.input
    backbone_output_names = find_layer_name(re.compile('^[PC][1-9]$'),
                                            model)
    backbone_outputs = [model.get_layer(name).output
                        for name in backbone_output_names]
    backbone_network = Model(backbone_inputs, backbone_outputs,
                             name='backbone')
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")


    # (2) Get Sub Networks related on Detection
    s = time.time()
    print("Detection Network를 재구성하는 중...")
    prior_subnet_name = find_layer_name(re.compile('^prior_layer*'),
                                        model)
    fpn_subnet_name = find_layer_name(re.compile('^feature_pyramid*'),
                                      model)
    cls_subnet_name = find_layer_name(re.compile("^classification_sub_net*"),
                                      model)
    loc_subnet_name = find_layer_name(re.compile("^box_regression_sub_net*"),
                                      model)
    if (len(prior_subnet_name)
        * len(fpn_subnet_name)
        * len(cls_subnet_name)
        * len(loc_subnet_name)) > 0:
        detection_networks = [
            model.get_layer(prior_subnet_name[0]),
            model.get_layer(fpn_subnet_name[0]),
            model.get_layer(cls_subnet_name[0]),
            model.get_layer(loc_subnet_name[0])]
    else:
        detection_networks = None
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    s = time.time()
    print("Instance Segmentation Network를 재구성하는 중...")
    # (3) Get Sub Networks related on Instance Segmentation
    restore_subnet_name = find_layer_name(re.compile("^restore_boxes*"),
                                          model)
    distribute_subnet_name = find_layer_name(re.compile("^mask_distribute*"),
                                             model)
    roi_align_subnet_name = find_layer_name(re.compile("^pyramid_roi_align*"),
                                            model)
    mask_subnet_name = find_layer_name(re.compile("^mask_sub_net*"),
                                       model)
    if len(mask_subnet_name) > 0:
        instance_networks = [
            model.get_layer(restore_subnet_name[0]),
            model.get_layer(distribute_subnet_name[0]),
            model.get_layer(roi_align_subnet_name[0]),
            model.get_layer(mask_subnet_name[0])]
    else:
        instance_networks = None
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    s = time.time()
    print("Semantic Segmentation Network를 재구성하는 중...")
    # (5) Get Sub Networks related on Semantic Segmentation
    aspp_subnet_name = find_layer_name(re.compile("^aspp*"),
                                       model)
    seg_subnet_name = find_layer_name(re.compile("^segmentation_sub_net*"),
                                      model)
    if len(aspp_subnet_name) * len(seg_subnet_name) > 0:
        semantic_networks = [
            model.get_layer(aspp_subnet_name[0]),
            model.get_layer(seg_subnet_name[0])]
    else:
        semantic_networks = None
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    s = time.time()
    print("각 Backbone Network와 Sub Network를 잇는 중...")
    model = construct_inference_network(
        config, backbone_network,
        semantic_networks=semantic_networks,
        detection_networks=detection_networks,
        instance_networks=instance_networks)
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    s = time.time()
    print("DownSampling 적용 중...")
    if serving:
        raw_inputs = Input((), dtype=tf.string, name='contents')
        inputs = DecodeImageContent()(raw_inputs)
    else:
        inputs = Input((None, None, 3),  dtype=tf.uint8, name='images')
        raw_inputs = inputs
    # DownSampling For Model Input Resolution
    downsampled = DownSampleInput(config.postprocess.resolution)(inputs)
    (_, _, box_pred, mask_pred, seg_pred) = model(downsampled)
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    s = time.time()
    print("후처리 코드 적용 중...")
    # Post-Process Instance Segmentation
    detection_pred, instance_pred = TrimInstances(
        mold=True)([box_pred, mask_pred])

    # Post-Process Instance Segmentation
    semantics = tf.split(seg_pred,
                     len(config.postprocess.smoothing_kernel_sizes),axis=-1)
    posts = []
    for tensor, kernel, weight in zip(
        semantics, config.postprocess.smoothing_kernel_sizes,
        config.postprocess.smoothing_weights):
        posts.append(SemanticSmoothing(kernel_size=kernel, weight=weight)(
            tensor))
    post_semantics = tf.concat(posts,axis=-1)
    semantic_pred = ResizeLike()(post_semantics,target=downsampled)
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    s = time.time()
    print("Upsampling 적용 중...")
    # Upsampling For Original Resolution
    outputs = UpSampleOutput()(
        [detection_pred, instance_pred, semantic_pred], target=inputs)
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    if serving:
        return Model(raw_inputs, [inputs]+list(outputs), name='serving')
    else:
        return Model(raw_inputs, outputs, name='inference')


def find_layer_name(re_format, model):
    return sorted([layer.name
                   for layer in model.layers
                   if re_format.match(layer.name)])

