"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np
import cv2
import os
import glob
from .dataset import Dataset
import pandas as pd


class MaskLabDataset(Dataset):
    def __init__(self, cases=None,
                 instance_labels=('car', 'bump', 'manhole', 'steel', 'pothole'),
                 semantic_labels=('other_road', 'my_road', 'crack'),
                 data_dir="./datasets/",
                 min_area=1000.,
                 **kwargs):
        """
        generate data for Hyundai Dataset
        """
        if cases is None:
            image_dir = os.path.join(data_dir, 'images/')
            self.cases = np.array(get_image_cases(image_dir))
        else:
            self.cases = np.array(cases)

        self.instance_labels = instance_labels
        self.semantic_labels = semantic_labels
        self.min_area = min_area

        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, "images/")
        self.label_dir = os.path.join(self.data_dir, 'labels/')
        self.process_dir = os.path.join(self.data_dir, 'processed/')
        self.semantic_dir = os.path.join(self.process_dir, 'semantic/')
        self.instance_dir = os.path.join(self.process_dir, 'instance/')

        self.label_exists_df = read_dataframe(
            os.path.join(self.process_dir, "label_exists.tsv"))
        self.label_exists_df[self.label_exists_df.columns[1:]] =\
            self.label_exists_df[self.label_exists_df.columns[1:]].astype(np.float)
        self.boxes_df = read_dataframe(
            os.path.join(self.process_dir, "boxes.tsv"))
        self.boxes_df = self.boxes_df[self.boxes_df.label.isin(instance_labels)]
        self.boxes_df = self.boxes_df[(self.boxes_df.w * self.boxes_df.h)>self.min_area]
        self.boxes_df.file_name = self.boxes_df.file_name.astype('category')
        self.boxes_df.label = self.boxes_df.label.map(
            lambda x: self.instance_labels.index(x))
        self.boxes_df.loc[:, 'confidence'] = 1.
        self.config = {
            "cases": list(self.cases),
            "instance_labels": instance_labels,
            "semantic_labels": semantic_labels,
            "data_dir": data_dir,
            "min_area": min_area,
        }
        self.config.update(kwargs)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        if isinstance(index, int):
            case_name = self.cases[index]
            image = read_image(os.path.join(self.image_dir, case_name))

            semantic_mask = self.get_semantic_mask(case_name, image.shape[:2])
            semantic_exist = self.get_semantic_exist(case_name)
            detection = self.get_detection(case_name)
            instance = self.get_instance(case_name, image.shape[:2])
            instance_exist = self.get_instance_exist(case_name)

            return {
                "images": image,
                "semantic": semantic_mask,
                "semantic_exist": semantic_exist,
                "detection": detection,
                "instance": instance,
                "instance_exist": instance_exist
            }
        elif isinstance(index, str):
            case_name = index
            image = read_image(os.path.join(self.image_dir, case_name))

            semantic_mask = self.get_semantic_mask(case_name, image.shape[:2])
            semantic_exist = self.get_semantic_exist(case_name)
            detection = self.get_detection(case_name)
            instance = self.get_instance(case_name, image.shape[:2])
            instance_exist = self.get_instance_exist(case_name)

            return {
                "images": image,
                "semantic": semantic_mask,
                "semantic_exist": semantic_exist,
                "detection": detection,
                "instance": instance,
                "instance_exist": instance_exist
            }
        else:
            cases = self.cases[index]
            images = None
            semantic_masks = None
            semantic_exists = None
            detection = None
            instances = None
            instance_exists = None

            image = read_image(os.path.join(self.image_dir, cases[0]))
            height, width = image.shape[:2]

            max_instances = np.max(
                self.boxes_df.loc[self.boxes_df.file_name.isin(cases), "file_name"].value_counts())
            max_instances = int(max_instances)

            for index, case_name in enumerate(cases):
                if images is None:
                    images = np.zeros((len(cases), height, width, 3), np.uint8)
                    semantic_masks = np.zeros((len(cases), height, width, len(self.semantic_labels)),
                                              np.uint8)
                    semantic_exists = np.zeros((len(cases), len(self.semantic_labels)))
                    instance_exists = np.zeros((len(cases), len(self.instance_labels)))
                    detection = np.ones((len(cases), max_instances, 6)) * -1
                    instances = np.full((len(cases), max_instances, height, width), -1, np.int8)

                images[index] = cv2.resize(
                    read_image(os.path.join(self.image_dir, case_name)), (width, height))
                semantic_masks[index] = cv2.resize(
                    self.get_semantic_mask(case_name, (height, width)), (width, height))
                semantic_exists[index] = self.get_semantic_exist(case_name)
                instance_exists[index] = self.get_instance_exist(case_name)

                one_detection = self.get_detection(case_name)
                detection[index, :len(one_detection)] = one_detection
                one_instances = self.get_instance(case_name, (height, width))
                instances[index, :len(one_instances)] = one_instances

            return {
                "images": images,
                "semantic": semantic_masks,
                "semantic_exist": semantic_exists,
                "detection": detection,
                "instance": instances,
                "instance_exist": instance_exists
            }

    def shuffle(self):
        np.random.shuffle(self.cases)

    def get_config(self):
        return self.config

    def get_semantic_mask(self, case_name, image_size):
        height, width = image_size[:2]
        semantic_mask = np.zeros((height, width, len(self.semantic_labels)), np.uint8)
        for idx, semantic_label in enumerate(self.semantic_labels):
            semantic_path = os.path.join(
                self.semantic_dir, f"{case_name}/{semantic_label}.png")
            if os.path.exists(semantic_path):
                semantic_mask[..., idx] = read_mask(semantic_path)
        return semantic_mask

    def get_semantic_exist(self, case_name):
        semantic_exist = self.label_exists_df.loc[
            self.label_exists_df.file_name == case_name, self.semantic_labels].values
        if semantic_exist.size == 0:
            semantic_exist = np.array([0., ])
        return semantic_exist.ravel()


    def get_detection(self, case_name):
        detection_df = self.boxes_df[self.boxes_df.file_name == case_name]
        return detection_df[['cx', 'cy', 'w', 'h', 'label', 'confidence']].values

    def get_instance(self, case_name, image_size):
        instance_dir = os.path.join(self.instance_dir, case_name)
        height, width = image_size[:2]

        df = self.boxes_df[self.boxes_df.file_name == case_name]

        instance = np.zeros((len(df), height, width), np.uint8)
        for idx, (_, row) in enumerate(df.iterrows()):
            bbox = (
                row.cx - row.w / 2, row.cy - row.h / 2,
                row.cx + row.w / 2, row.cy + row.h / 2)
            bbox = np.array(bbox, np.int)
            bbox = np.maximum(bbox, 0)
            x1, y1, x2, y2 = bbox

            mask_path = os.path.join(instance_dir, str(row.mask_index)+".png")
            shape = instance[idx, y1:y2 + 1, x1:x2 + 1].shape
            instance[idx, y1:y2 + 1, x1:x2 + 1] = cv2.resize(read_mask(mask_path),
                                                             shape[::-1])
        return instance

    def get_instance_exist(self, case_name):
        instance_exist = self.label_exists_df.loc[
            self.label_exists_df.file_name == case_name, self.instance_labels].values
        if instance_exist.size == 0:
            instance_exist = np.array([0., ])
        return instance_exist.ravel()


def read_image(filepath):
    image = cv2.imread(filepath)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(filepath):
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)


def read_dataframe(filepath):
    if os.path.splitext(filepath)[1] == '.tsv':
        return pd.read_csv(filepath, sep='\t')
    else:
        return pd.read_csv(filepath)


def get_image_cases(image_dir):
    file_paths = glob.glob(os.path.join(image_dir, "**/*"), recursive=True)
    image_formats = (".jpg", '.jpeg', '.png')

    file_paths = [
        file_path for file_path in file_paths
        if os.path.splitext(file_path)[1].lower() in image_formats]

    return [os.path.split(file_path)[1] for file_path in file_paths]