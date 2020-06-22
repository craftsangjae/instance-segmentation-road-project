"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import pandas as pd
import numpy as np


class PriorBoxes:
    """
    Default Box Configuration Class
    """
    boxes = pd.DataFrame()

    def __init__(self, strides, sizes, pr_scales, pr_ratios):
        """
        :param strides: 각 Feature Map에서의 Stride 크기
        :param sizes: 각 Feature Map에서의 기준 Prior box 크기
        :param pr_scales: Feature Map에서의 width & height의 Scale
        :param pr_ratios: Feature Map에서의 Width & Height의 Shape
        """
        if isinstance(strides, np.ndarray):
            self.strides = strides.tolist()
        else:
            self.strides = strides
        if isinstance(sizes, np.ndarray):
            self.sizes = sizes.tolist()
        else:
            self.sizes = sizes
        if isinstance(pr_scales, np.ndarray):
            self.pr_scales = pr_scales.tolist()
        else:
            self.pr_scales = pr_scales
        if isinstance(pr_ratios, np.ndarray):
            self.pr_ratios = pr_ratios.tolist()
        else:
            self.pr_ratios = pr_ratios

        self.setup()
        assert len(self.strides) == len(self.sizes), "stride의 갯수와 size의 갯수는 동일해야 합니다."
        self.config = {
            "strides": self.strides,
            "sizes": self.sizes,
            "pr_scales": self.pr_scales,
            "pr_ratios": self.pr_ratios
        }

    def __len__(self):
        """
        Return the Number of Anchors(Prior shape)
        :return:
        """
        return len(self.pr_scales) * len(self.pr_ratios)

    def setup(self):
        boxes = pd.DataFrame(columns=['stride', 'w', 'h'])
        for size, stride in zip(self.sizes, self.strides):
            for wh_size in self.pr_scales:
                for wh_ratio in self.pr_ratios:
                    w = np.round(size * wh_size * np.sqrt(wh_ratio)).astype(np.int)
                    h = np.round(size * wh_size / np.sqrt(wh_ratio)).astype(np.int)
                    boxes.loc[len(boxes) + 1] = [stride, w, h]

        boxes.stride = boxes.stride.astype(np.int)
        boxes.w = boxes.w.astype(np.int)
        boxes.h = boxes.h.astype(np.int)
        self.boxes = boxes

    def get_config(self):
        return self.config

