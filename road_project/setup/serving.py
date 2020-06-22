"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from engine.retinamasklab import load_masklab_inference_model_from_h5
from engine.layers import DrawInstance, DrawSegmentation, DrawBoxes, CropAndPadMask
from engine.layers import EncodeImageContent
from engine.config import ModelConfiguration
from engine.layers import SummaryOutput
from tensorflow.python.keras.models import Model
import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
import time


def load_serving_model_from_h5(weight_path, config:ModelConfiguration):
    """
    h5 file로부터, Serving 용 모델을 구성하는 메소드

    :param weight_path: h5 file이 저장된 폴더
    :param config: Model Configuration 파일
    :return:
    """
    K.clear_session()
    model = load_masklab_inference_model_from_h5(weight_path, config,
                                                 serving=True)

    images, det_outs, ins_outs, seg_outs = model.outputs
    crop_and_pad_masks = CropAndPadMask()(model.outputs)

    s = time.time()
    print("Output Visualization Network를 구성 중...")
    vis_boxes = DrawBoxes()([images, det_outs])
    vis_instance = DrawInstance(config.postprocess.instance_colors,
                                config.postprocess.instance_alpha)(
        [vis_boxes, det_outs, crop_and_pad_masks])
    vis_semantic = DrawSegmentation(config.postprocess.semantic_colors,
                                    config.postprocess.semantic_alpha)(
        [vis_instance, seg_outs])
    contents = EncodeImageContent()(vis_semantic)
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    s = time.time()
    print("Output Summary Network를 구성 중...")
    summary = SummaryOutput(default_road_size=config.postprocess.default_road_size)([
        det_outs, seg_outs, crop_and_pad_masks])
    c = time.time() - s
    print(f"완료....(소요시간  :{c:.3f}s)")

    serving_model = Model(model.input, [contents, summary], name='serving')
    return serving_model


def save_serving_model(serving, serving_dir):
    """
    keras.model을 savedModel 포맷으로 정하는 메소드

    :param serving:
    :param serving_dir:
    :return:
    """
    version_dir = get_latest_version_dir(serving_dir)

    K.set_learning_phase(0)
    session = K.get_session()

    tf.saved_model.simple_save(session, version_dir,
        inputs={'image': serving.input},
        outputs={'visualize': serving.output[0],
                 'summarize': serving.output[1]})


def get_latest_version_dir(export_dir):
    """
    현재 export_dir 내 새 version directory의 경로를 가져오기

    :param export_dir:
    :return:
    """
    os.makedirs(export_dir,exist_ok=True)
    curr_version = max([0] + [int(version) for version in os.listdir(export_dir)]) + 1
    return os.path.join(export_dir, str(curr_version))


__all__ = [
    "load_serving_model_from_h5", "save_serving_model", "get_latest_version_dir"
]