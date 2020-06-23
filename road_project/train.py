"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from mlflow import log_param, log_metric, log_artifact
import os
import numpy as np
import sys
import cv2
from tqdm import tqdm
import pandas as pd
from datetime import datetime
sys.path.append("../")
from road_project import DATA_DIR, LOG_DIR, PROCESS_DIR

from engine.train import train_masklab_model, construct_masklabdataset
from engine.config import ModelConfiguration
from engine.retinamasklab import load_masklab_inference_model_from_h5

# Default Configuration For road project
config = ModelConfiguration()

# Dataset 관련 Configuration
config.dataset.train_cases = pd.read_csv(os.path.join(PROCESS_DIR, 'train.csv'),
                                         header=None)[0].tolist()
config.dataset.valid_cases = pd.read_csv(os.path.join(PROCESS_DIR, 'valid.csv'),
                                         header=None)[0].tolist()
config.dataset.min_area = 200.0
config.dataset.instance_labels = ('car', 'bump', 'manhole', 'steel', 'pothole')
config.dataset.semantic_labels = ('other_road', 'my_road', 'crack')
config.dataset.except_semantic_labels = ('car',)

config.dataset.data_dir = DATA_DIR

# Backbone 관련 Configuration
config.backbone.backbone_type = 'seresnet34'
config.backbone.backbone_outputs = ('C3', 'C4', 'C5', 'P6')

# Detection 관련 Configuration
config.detection.num_features = 128
config.detection.num_depth = 3
config.detection.use_squeeze_excite = True

config.detection.pr_scales = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]
config.detection.pr_ratios = [1 / 2, 1, 2, 5, 8]

# Instance Segmentation 관련 Configuration
config.instance.crop_size = (14, 14)
config.instance.max_k = 2
config.instance.num_features = 128
config.instance.num_depth = 4
config.instance.use_squeeze_excite = True

# Semantic Segmentation 관련 Configuration
config.semantic.num_features = 128
config.semantic.num_depth = 3
config.semantic.use_squeeze_excite = True

# Training 관련
config.train.gpu_count = 2
config.train.use_multiprocessing = False
config.train.head_max_lr = 3e-4
config.train.waist_max_lr = 3e-4


if __name__ == "__main__":
    # ArgParsing
    args = config.get_arg_parser()
    for key, value in args._get_kwargs():
        attr_group, attr = key.split('.')
        config.update(attr_group, attr, value=value)
        log_param(key, value)
    dt = datetime.strftime(datetime.now(), '%m-%d-%H')
    save_dir = os.path.join(LOG_DIR, f"{config.backbone.backbone_type}/{dt}/")
    os.makedirs(save_dir, exist_ok=True)
    config.train.save_dir = save_dir
    config_path = os.path.join(config.train.save_dir, "config.json")

    # Train Model
    train_masklab_model(config)

    # Load Best Model
    model_dir = os.path.join(config.train.save_dir, 'engine/')
    best_filename = sorted(os.listdir(model_dir))[0]
    best_filepath = os.path.join(model_dir, best_filename)
    log_artifact(config_path)
    log_artifact(best_filepath)

    model = load_masklab_inference_model_from_h5(best_filepath, config)

    # Evaluate Model
    _, validset = construct_masklabdataset(config)

    batch_size = 1
    semantic_labels = list(config.dataset.semantic_labels)
    instance_labels = list(config.dataset.instance_labels)

    result_df = pd.DataFrame(columns=['iou', 'counts'],
                             index=semantic_labels + instance_labels)
    result_df[:] = 0
    for idx in tqdm(range(len(validset) // batch_size)):
        # 데이터 가져오기
        targets = validset[idx * batch_size:(idx + 1) * batch_size]
        images = targets['images']
        gt_detections = targets['detection']
        gt_semantics = targets['semantic']
        gt_instances = targets['instance']

        # 추론하기
        (pr_detections, pr_instances, pr_semantics) = model.predict(images)

        #####
        # Instance Segmentation을 평가
        #####
        for idx in range(batch_size):
            gt_detection = gt_detections[idx]
            pr_detection = pr_detections[idx]
            gt_instance = gt_instances[idx]
            pr_instance = pr_instances[idx]
            gt_semantic = gt_semantics[idx]
            pr_semantic = pr_semantics[idx]

            image_h, image_w = images.shape[1:3]
            masks = []
            for j, box in enumerate(pr_detection):
                if box[-1] < 0:
                    continue
                xmin = np.clip(box[0] - box[2] / 2, 0, image_w)
                xmax = np.clip(box[0] + box[2] / 2, 0, image_w)
                ymin = np.clip(box[1] - box[3] / 2, 0, image_h)
                ymax = np.clip(box[1] + box[3] / 2, 0, image_h)

                start = tuple(np.array((xmin, ymin), dtype=np.int32))
                end = tuple(np.array((xmax, ymax), dtype=np.int32))
                w = end[0] - start[0]
                h = end[1] - start[1]
                mask = cv2.resize(np.maximum(pr_instance[j].astype(np.float), 0.), (w, h))
                masks.append(np.pad((mask > 0.5).astype(np.int8),
                                    ((start[1], image_h - end[1]),
                                     (start[0], image_w - end[0])),
                                    'constant'))
            pr_instance = masks

            gt_area = (gt_detection[:, 2] * gt_detection[:, 3])
            pr_area = (pr_detection[:, 2] * pr_detection[:, 3])
            areas = gt_area[None, :] + pr_area[:, None]

            gt_norm_boxes = np.stack(
                (gt_detection[:, 0] - gt_detection[:, 2] / 2,
                 gt_detection[:, 0] + gt_detection[:, 2] / 2,
                 gt_detection[:, 1] - gt_detection[:, 3] / 2,
                 gt_detection[:, 1] + gt_detection[:, 3] / 2),
                axis=1)
            pr_norm_boxes = np.stack(
                (pr_detection[:, 0] - pr_detection[:, 2] / 2,
                 pr_detection[:, 0] + pr_detection[:, 2] / 2,
                 pr_detection[:, 1] - pr_detection[:, 3] / 2,
                 pr_detection[:, 1] + pr_detection[:, 3] / 2),
                axis=1)

            px1, px2, py1, py2 = np.transpose(pr_norm_boxes[:, None], (2, 0, 1))
            gx1, gx2, gy1, gy2 = np.transpose(gt_norm_boxes[None, :], (2, 0, 1))

            in_ymin = np.maximum(gy1, py1)
            in_xmin = np.maximum(gx1, px1)
            in_ymax = np.minimum(gy2, py2)
            in_xmax = np.minimum(gx2, px2)

            in_width = np.maximum(0., in_xmax - in_xmin)
            in_height = np.maximum(0., in_ymax - in_ymin)

            # 3. Calculate intersection size and union size
            intersection = in_width * in_height
            union = areas - intersection

            # 4. Masking if pred and gt's label is different
            iou = intersection / (union)
            iou = iou * np.equal(gt_detection[None, :, -2],
                                 pr_detection[:, None, -2])

            # 5. calculate iou
            for pr_i, gt_i in zip(*np.where(iou > 0.5)):
                label = pr_detection[pr_i, -2]
                mask_intersect = np.logical_and(
                    pr_instance[pr_i], gt_instance[gt_i])
                mask_union = np.logical_or(
                    pr_instance[pr_i], gt_instance[gt_i])
                mask_iou = np.sum(mask_intersect) / np.sum(mask_union)
                result_df.loc[instance_labels[int(label)], "iou"] += mask_iou
                result_df.loc[instance_labels[int(label)], "counts"] += 1

            #######
            # Semantic Segmentation을 평가
            #######
            mask_intersect = np.logical_and(gt_semantic > 0.5, pr_semantic > 0.5)
            mask_union = np.logical_or(gt_semantic > 0.5, pr_semantic > 0.5)
            ious = (np.sum(mask_intersect, axis=(0, 1)) /
                    (np.sum(mask_union, axis=(0, 1)) + 1e-7))

            result_df.loc['other_road', 'iou'] += ious[0]
            result_df.loc['other_road', 'counts'] += 1

            result_df.loc['my_road', 'iou'] += ious[1]
            result_df.loc['my_road', 'counts'] += 1

            if np.any(gt_instance[..., -1] != -1):
                result_df.loc['crack', 'iou'] += ious[2]
                result_df.loc['crack', 'counts'] += 1
        result_df.loc[:, "miou"] = result_df.iou / (result_df.counts + 1e-7)

    for key, value in result_df.miou.items():
        log_metric(key, value)
