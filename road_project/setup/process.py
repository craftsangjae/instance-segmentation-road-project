"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import cv2
import json
import numpy as np
import pandas as pd
from road_project.setup.imglab import imglabformat_to_dataframe
from skimage.draw import polygon
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from engine.utils import get_image_cases
import time
from tqdm import tqdm
import random


def load_label_dataframes_from_imglab_files(label_dir):
    """
    imglab의 coco 포맷으로 되어 있는 라벨링 정보를 모아서
    pd.Dataframe으로 가져오는 메소드

    :param label_dir:
    :return:
    """
    df_dict = {}
    for label_name in os.listdir(label_dir):
        if os.path.isdir(os.path.join(label_dir, label_name)):
            dfs = []
            file_dir = os.path.join(label_dir, label_name)
            for file_name in os.listdir(file_dir):
                if os.path.splitext(file_name)[1].lower() != '.json':
                    continue
                file_path = os.path.join(file_dir, file_name)
                with open(file_path, 'r') as f:
                    coco = json.load(f)
                    dfs.append(imglabformat_to_dataframe(coco))

            file_names = set()
            for df in dfs:
                file_names |= set(df.file_name.cat.categories.values)
            df = pd.concat(dfs)

            df.file_name = pd.Categorical(df.file_name, categories=file_names)
            df.name = label_name
            df_dict[label_name] = df

    file_exists = {}
    for key, value in df_dict.items():
        file_exists[key] = value.file_name.unique()

    filenames = set()
    for value in file_exists.values():
        filenames = filenames | set(value)

    label_exists = pd.DataFrame(data=list(filenames),
                                columns=["file_name"])
    for key, value in df_dict.items():
        label_exists[key] = (
            label_exists.file_name.isin(value.file_name.cat.categories))

    df = pd.concat(df_dict.values())
    file_name = df.file_name
    label = df.name
    cx = df.bbox.apply(lambda x: x[0])
    cy = df.bbox.apply(lambda x: x[1])
    w = df.bbox.apply(lambda x: x[2])
    h = df.bbox.apply(lambda x: x[3])
    annotation = df.segmentation.apply(lambda x: np.array(x).reshape(-1, 2))

    annotations = pd.concat(
        [file_name, cx, cy, w, h, label, annotation], axis=1)
    annotations.columns = ['file_name', 'cx', 'cy', 'w', 'h', 'label', 'annotation']
    annotations = annotations[(annotations.w * annotations.h) > 0]
    annotations = annotations.sort_values('file_name')

    return label_exists, annotations


def process_semantic(inputs, data_dir, semantic_labels, except_semantic_labels):
    """
    Semantic Mask를 그려서 저장하는 메소드
    """
    image_dir = os.path.join(data_dir, 'images')
    processed_dir = os.path.join(data_dir, 'processed')

    file_name, df = inputs

    semantic_save_dir = \
        os.path.join(processed_dir, f"semantic/{file_name}")
    os.makedirs(semantic_save_dir, exist_ok=True)

    image_size = None
    except_mask = None

    for label in semantic_labels:
        save_path = os.path.join(semantic_save_dir, f"{label}.png")
        if os.path.exists(save_path) or np.sum(df.label==label)==0:
            continue
        if image_size is None and except_mask is None:
            image_size = cv2.imread(
                os.path.join(image_dir, file_name)).shape[:2]
            height, width = image_size[:2]
            except_mask = np.zeros((height, width))
            for except_label in except_semantic_labels:
                for point in df.loc[df.label == except_label, 'annotation']:
                    point[:, 0] = np.clip(point[:, 0], 0, width - 1)
                    point[:, 1] = np.clip(point[:, 1], 0, height - 1)
                    rr, cc = polygon(point[:, 1], point[:, 0])
                    except_mask[rr, cc] = 1

        mask = np.zeros((height, width))
        for point in df.loc[df.label == label, 'annotation']:
            point[:, 0] = np.clip(point[:, 0], 0, width - 1)
            point[:, 1] = np.clip(point[:, 1], 0, height - 1)
            rr, cc = polygon(point[:, 1], point[:, 0])
            mask[rr, cc] = 1
        label_mask = ((mask - except_mask)>0).astype(np.uint8)

        cv2.imwrite(save_path, label_mask,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])


def process_instance(inputs, data_dir, instance_labels):
    """
    Instance Mask를 그려서 저장하는 메소드
    """
    file_name, df = inputs

    image_dir = os.path.join(data_dir, 'images')
    processed_dir = os.path.join(data_dir, 'processed')

    instance_save_dir = os.path.join(processed_dir,
                                     f'instance/{file_name}')
    os.makedirs(instance_save_dir, exist_ok=True)

    target_df = df[df.label.isin(instance_labels)]
    image_path = os.path.join(image_dir, file_name)

    height = None
    width = None
    for idx, row in target_df.iterrows():
        save_path = os.path.join(instance_save_dir, f'{row.mask_index}.png')
        if os.path.exists(save_path):
            continue
        if height is None and width is None:
            height, width = cv2.imread(image_path).shape[:2]

        point = row.annotation
        point[:, 0] = np.clip(point[:, 0], 0, width - 1)
        point[:, 1] = np.clip(point[:, 1], 0, height - 1)
        rr, cc = polygon(point[:, 1], point[:, 0])

        blank = np.zeros((height, width))
        blank[rr, cc] = 1
        bbox = (
            row.cx - row.w / 2, row.cy - row.h / 2,
            row.cx + row.w / 2, row.cy + row.h / 2)
        bbox = np.array(bbox, np.int)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, np.inf)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, np.inf)
        x1, y1, x2, y2 = bbox

        mask = blank[y1:y2 + 1, x1:x2 + 1]
        cv2.imwrite(save_path,
                    mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def process_dataset(data_dir,
                    except_semantic_labels=('car',)):
    image_dir = os.path.join(data_dir, 'images/')
    label_dir = os.path.join(data_dir, 'labels/')
    processed_dir = os.path.join(data_dir, 'processed/')
    os.makedirs(processed_dir, exist_ok=True)

    label_exists, annotations =\
    load_label_dataframes_from_imglab_files(label_dir)

    labels = list(label_exists.columns[1:])

    s = time.time()
    print("Start Calculating Table about the Existence of Labels per images")
    label_exists = label_exists[
        label_exists.file_name.isin(set(os.listdir(image_dir)))]
    label_exists.to_csv(
        os.path.join(processed_dir, 'label_exists.tsv'),
        sep='\t', index=False)
    print(f"Finish --- consumed time : {time.time()-s:.3f}")

    s = time.time()
    print("Start spliting train and validation dataset by file names")
    fnames = get_image_cases(image_dir)
    random.seed(777)
    random.shuffle(fnames)

    valid_nums = int(len(fnames) * 0.1)
    valid_fnames = fnames[:valid_nums]
    train_fnames = fnames[valid_nums:]
    train_df = pd.DataFrame(train_fnames, columns=['file_name'])
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False, header=None)

    valid_df = pd.DataFrame(valid_fnames, columns=['file_name'])
    valid_df.to_csv(os.path.join(processed_dir, 'valid.csv'), index=False, header=None)
    print(f"Finish --- consumed time : {time.time()-s:.3f}")

    s = time.time()
    print("Start Calculating Table about the bounding boxes of Labels per images")
    annotations = annotations[
        annotations.file_name.isin(set(os.listdir(image_dir)))]
    dfs = []
    for file_name, sub_df in annotations.groupby('file_name'):
        dfs.append(sub_df.reset_index(drop=True))
    annotations = pd.concat(dfs)
    annotations = annotations.reset_index().rename({'index': 'mask_index'}, axis=1)
    annotations = annotations[
        ['file_name', 'cx', 'cy', 'w', 'h', 'label', 'mask_index', 'annotation']]

    annotations[['file_name', 'cx', 'cy', 'w', 'h', 'label', 'mask_index']].to_csv(
        os.path.join(processed_dir, 'boxes.tsv'), sep='\t', index=False)
    print(f"Finish --- consumed time : {time.time()-s:.3f}")

    total = len(annotations.file_name.unique())

    s = time.time()
    print("Start Drawing Semantic Labels per Images")
    do_work = partial(process_semantic,
                      data_dir=data_dir,
                      semantic_labels=labels,
                      except_semantic_labels=except_semantic_labels)
    pool = Pool(cpu_count())
    for _ in tqdm(pool.imap_unordered(do_work, annotations.groupby('file_name')),
                  total=total):
        pass
    print(f"Finish --- consumed time : {time.time()-s:.3f}")

    s = time.time()
    print("Start Drawing Instance Labels per Images")
    do_work = partial(process_instance,
                      data_dir=data_dir,
                      instance_labels=labels)
    pool = Pool(cpu_count())
    for _ in tqdm(pool.imap_unordered(do_work, annotations.groupby('file_name')),
                  total=total):
        pass
    print(f"Finish --- consumed time : {time.time()-s:.3f}")
