"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import pandas as pd


def dataframe_to_imglabformat(df):
    df = df.copy()
    df.file_name = df.file_name.astype('category')
    df.name = df.name.astype('category')

    file_df = pd.DataFrame(df.file_name.cat.categories,
                           columns=['file_name'])
    file_df = file_df.reset_index()
    file_df['index'] += 1
    file_df = file_df.rename({'index': 'id'}, axis=1)
    file_df['width'] = 1920
    file_df['height'] = 1080

    cate_df = pd.DataFrame(df.name.cat.categories,
                           columns=['name'])
    cate_df = cate_df.reset_index()
    cate_df['index'] += 1
    cate_df = cate_df.rename({'index': 'id'}, axis=1)
    cate_df['supercategory'] = 'none'

    df['image_id'] = df.file_name.cat.codes + 1
    anno_df = df.drop('file_name', axis=1)
    anno_df['category_id'] = df.name.cat.codes + 1
    anno_df = anno_df.drop('name', axis=1)

    min_x = anno_df.segmentation.apply(lambda x: min(x[0][::2]))
    min_y = anno_df.segmentation.apply(lambda x: min(x[0][1::2]))
    max_x = anno_df.segmentation.apply(lambda x: max(x[0][::2]))
    max_y = anno_df.segmentation.apply(lambda x: max(x[0][1::2]))

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y

    anno_df['ignore'] = 0
    anno_df['iscrowd'] = 0
    anno_df['bbox'] = list(list(p) for p in zip(center_x, center_y, w, h))
    anno_df['area'] = (w * h).astype(float)

    a = []
    for _, x in anno_df.groupby('image_id'):
        a.append(x.sort_values('area',ascending=False).reset_index(drop=True))
    a = pd.concat(a)
    anno_df = a.reset_index()
    anno_df = anno_df.rename({'index': 'id'}, axis=1)
    anno_df['id'] += 1

    return {"images": file_df.to_dict(orient='row'),
            "annotations": anno_df.to_dict(orient='row'),
            "categories": cate_df.to_dict(orient='row'),
            "type": "instances"}


def imglabformat_to_dataframe(imglab):
    file_df = pd.DataFrame(imglab['images'])
    file_df = file_df[['file_name', 'id']]
    anno_df = pd.DataFrame(imglab['annotations'])
    anno_df = anno_df[['image_id', 'category_id', 'id', 'bbox', 'segmentation']]
    cate_df = pd.DataFrame(imglab['categories'])
    cate_df = cate_df[['name', 'id']]
    merge_df = (anno_df
                .merge(cate_df,
                       left_on='category_id',
                       right_on='id',
                       suffixes=('', '_y'))
                .merge(file_df,
                       left_on='image_id',
                       right_on='id',
                       suffixes=('', '_y')))
    merge_df['file_name'] = pd.Categorical(
        merge_df['file_name'], categories=file_df.file_name.unique())
    return merge_df[['file_name', 'segmentation', 'name', 'bbox']]

