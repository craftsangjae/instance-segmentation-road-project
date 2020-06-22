"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from IPython.display import display, HTML
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2


def color_map(seed=40):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap[1:]
    np.random.seed(seed)
    np.random.shuffle(cmap)
    return cmap


def draw_semantics(image, segs, color_map=color_map()):
    """ 주어진 좌표값 Dataframe에 따라, image에 사각형을 그리는 메소드
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.copy().astype(np.uint8)

    blank = np.zeros_like(image)
    for color, seg in zip(color_map, np.transpose(segs,(2, 0, 1))):
        seg = (seg > 0.5).astype(np.uint8)
        seg = (seg[..., None] * color).astype(np.uint8)
        blank = cv2.addWeighted(blank, 1., seg, .3, 1.)
    return cv2.addWeighted(image, 1., blank, .9, 1.)


def draw_instances(image, boxes, masks, color_map=color_map(), thickness=3):
    """ 주어진 좌표값 Dataframe과 Mask에 따라 그림을 그리는 모델
    """
    masks = masks[boxes[:, -2] != -1]
    if isinstance(boxes, np.ndarray):
        if boxes.shape[1] == 4:
            boxes = pd.DataFrame(boxes, columns=['cx', 'cy', 'w', 'h'])
        elif boxes.shape[1] == 5:
            boxes = pd.DataFrame(boxes, columns=['cx', 'cy', 'w', 'h', 'label'])
        elif boxes.shape[1] == 6:
            boxes = pd.DataFrame(boxes, columns=['cx', 'cy', 'w', 'h', 'label', 'confidence'])
    elif isinstance(boxes, pd.DataFrame):
        pass
    else:
        raise TypeError("digits은 numpy.ndarray 혹은 pandas.Dataframe으로 이루어져 있어야 합니다.")
    if 'label' in boxes:
        boxes.label = boxes.label.astype(np.int)
        boxes = boxes[boxes.label >= 0]  # drop blank label
        boxes.reset_index(drop=True, inplace=True)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()

    for label, boxes_per_label in boxes.groupby('label'):
        color = color_map[label]
        blank = np.zeros(image.shape[:2], np.uint8)
        image_h, image_w = blank.shape[:2]
        for idx, row in boxes_per_label.iterrows():
            xmin = np.clip(row.cx - row.w / 2, 0, image_w)
            xmax = np.clip(row.cx + row.w / 2, 0, image_w)
            ymin = np.clip(row.cy - row.h / 2, 0, image_h)
            ymax = np.clip(row.cy + row.h / 2, 0, image_h)

            start = tuple(np.array((xmin, ymin), dtype=np.int32))
            end = tuple(np.array((xmax, ymax), dtype=np.int32))
            image = cv2.rectangle(image, start, end, color.tolist(), thickness)
            w = end[0] - start[0]
            h = end[1] - start[1]
            mask = masks[idx].astype(np.uint8)
            if not np.all(image.shape[:2] == mask.shape[:2]):
                mask = cv2.resize(mask, (w, h))
                mask = np.pad(mask, ((start[1], image_h - end[1]),
                                     (start[0], image_w - end[0])),
                              'constant')
                mask = (mask > 0.5).astype(np.uint8)
            blank = cv2.add(blank, mask)
        blank = (blank[..., None] * color)
        image = cv2.addWeighted(image, 1., blank, 0.3, 1.)
    return image


"""
Tensorflow Graph Visualization
"""


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

