"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras.utils import get_custom_objects
from .losses import *
from .layers import *
from .backbone import *
from .backbone.ResNext import *
from .metrics import *
from .normalization import *
from .optimizers import *
from .prior import *
from .callbacks import *

# Base Model 전처리에 필요한 Custom Keras Object
get_custom_objects().update({'BackBonePreProcess': BackBonePreProcess})

# ResNext 구성에 필요한 Custom Keras Object
get_custom_objects().update({'MergeGroups': MergeGroups,
                             'ReduceGroups': ReduceGroups,
                             'SplitGroups': SplitGroups})


# RetinaMask & DeepLabV3+ Modeling에 필요한 Custom Keras Object
get_custom_objects().update({'RestoreBoxes': RestoreBoxes,
                             'PriorLayer': PriorLayer,
                             "Identity": Identity,
                             "FeaturePyramid": FeaturePyramid,
                             "ClassificationSubNet": ClassificationSubNet,
                             "BoxRegressionSubNet": BoxRegressionSubNet,
                             "MaskSubNet": MaskSubNet,
                             "MobileSeparableConv2D": MobileSeparableConv2D,
                             "NormalizeBoxes": NormalizeBoxes,
                             "DetectionProposal": DetectionProposal,
                             "MoldBatch": MoldBatch,
                             "MaskDistribute": MaskDistribute,
                             "PyramidRoiAlign": PyramidRoiAlign,
                             "AssignMasks": AssignMasks,
                             "CalculateIOU": CalculateIOU,
                             "AssignBoxes": AssignBoxes,
                             "ResizeLike": ResizeLike,
                             "AtrousSeparableConv2D": AtrousSeparableConv2D,
                             "ASPPNetwork": ASPPNetwork,
                             "SegmentationSubNet": SegmentationSubNet,
                             "AssignSeg": AssignSeg,
                             "SemanticSmoothing": SemanticSmoothing,
                             "TrimInstances": TrimInstances,
                             "DecodeImageContent": DecodeImageContent,
                             "CropAndPadMask": CropAndPadMask,
                             "DrawSegmentation": DrawSegmentation,
                             "DrawInstance": DrawInstance,
                             "SummaryOutput": SummaryOutput,
                             "CalculateInstanceSize": CalculateInstanceSize,
                             "IncludeMyRoad": IncludeMyRoad})

# Loss에 관련된 Custom Keras Object
get_custom_objects().update({'ClassLoss': ClassLoss,
                             'BoxLoss': BoxLoss,
                             'MaskLoss': MaskLoss,
                             "SegLoss": SegLoss})

# Normalization에 관련된 Custom Keras Object
get_custom_objects().update({'GroupNormalization': GroupNormalization})

# Metric 관련된 Custom Keras Object
get_custom_objects().update({"ClassBinaryIOU": ClassBinaryIOU,
                             'ConfusionMatrixMetric': ConfusionMatrixMetric,
                             'DetectionIOUMetric': DetectionIOUMetric})

# Optimzier에 관련된 Custom Keras Object
get_custom_objects().update({'AdamW': AdamW,
                             'RectifiedAdam': RectifiedAdam})
