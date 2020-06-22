"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.python.keras.utils import Sequence
from ..dataset import MaskLabDataset
from ..dataset import Dataset
import cv2
import numpy as np


class MaskLabGenerator(Sequence):
    'Generates Instance & Semantic Segmentation dataset for Keras'
    def __init__(self, dataset:Dataset,
                 scale_ratio=(0.4, 0.6),
                 batch_size=8,
                 shuffle=True):
        'Initialization'
        # Dictionary로 받았을 때에만 Multiprocessing이 동작가능함.
        # Keras fit_generator에서 Multiprocessing으로 동작시키기 위함
        if isinstance(dataset, dict):
            self.dataset = MaskLabDataset(**dataset)
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise ValueError('dataset은 dict혹은 DetectionDataset Class로 이루어져 있어야 합니다.')

        self.scale_ratio = scale_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        data = self.dataset[self.batch_size * index:
                            self.batch_size * (index + 1)]

        images = data['images']
        gt_seg = data['semantic']
        gt_seg = gt_seg.astype(np.float)
        gt_seg_exist = data['semantic_exist'].astype(np.float)
        gt_boxes = data['detection']
        gt_masks = data['instance']
        gt_boxes_exist = data['instance_exist'].astype(np.float)

        batch_images = []
        batch_seg = []
        if isinstance(self.scale_ratio, tuple) or isinstance(self.scale_ratio, list):
            scale_ratio = np.random.uniform(*self.scale_ratio)
        else:
            scale_ratio = self.scale_ratio

        height, width = images.shape[1:3]
        target_h = np.int(height * scale_ratio)
        target_w = np.int(width * scale_ratio)
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32

        for seg, image in zip(gt_seg, images):
            batch_images.append(cv2.resize(image, (target_w, target_h)))
            batch_seg.append(cv2.resize(seg, (target_w, target_h)))
        batch_images = np.stack(batch_images)
        batch_seg = np.stack(batch_seg)
        batch_seg = np.round(batch_seg)

        batch_size, max_instances, _, _ = gt_masks.shape
        batch_masks = np.full((batch_size, max_instances,target_h,target_w),-1,np.int8)
        for i, masks in enumerate(gt_masks):
            for j, mask in enumerate(masks):
                if mask[0,0] == -1.:
                    continue
                batch_masks[i, j] = cv2.resize(mask.astype(np.uint8), (target_w, target_h))

        not_ignore = gt_boxes[..., 5] > 0
        gt_boxes[not_ignore, 0] *= target_w / width
        gt_boxes[not_ignore, 1] *= target_h / height
        gt_boxes[not_ignore, 2] *= target_w / width
        gt_boxes[not_ignore, 3] *= target_h / height

        X = {"images": batch_images,
             "gt_seg": batch_seg,
             'gt_seg_exist': gt_seg_exist,
             'gt_boxes': gt_boxes,
             'gt_boxes_exist': gt_boxes_exist,
             'gt_masks': batch_masks}
        return (X, )

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.dataset.shuffle()

