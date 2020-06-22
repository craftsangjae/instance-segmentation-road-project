"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import docker
import os
import cv2
import fire
import numpy as np
from tqdm import tqdm
import glob
import json
from road_project.setup.serving import load_serving_model_from_h5, save_serving_model
from engine.config import ModelConfiguration
import tensorflow as tf
import time
import pandas as pd
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework.tensor_util import MakeNdarray

# Error Log Message Disable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TAG_NAME = "tensorflow/serving:latest" # Tensorflow Serving Image NAME
ROOT_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(ROOT_DIR, "test")
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

INSTANCE_LABELS = {
    0: 'car',
    1: 'bump',
    2: 'manhole',
    3: 'steel',
    4: 'pothole',
    5: 'crack'
}


def run_tensorflow_serving_container(model_dir=os.path.join(MODEL_DIR, 'v0.0'),
                                     grpc_port=8500, rest_port=8501):
    """
    :param model_dir:
    :return:
    """
    global TAG_NAME

    s = time.time()
    print("Docker Client Set-up ...")
    client = docker.from_env()
    if not client.ping():
        raise ValueError(
            "")
    c = time.time() - s
    print(f"Finished ----- {c:.3f}s")

    s = time.time()
    print("Docker Tensorflow Serving Image creating ...")
    tags = sum([image.tags
                for image in client.images.list()], [])

    if not TAG_NAME in tags:
        client.images.pull(TAG_NAME)
    c = time.time() - s
    print(f"Finished ----- {c:.3f}s")

    serving_dir = os.path.abspath(os.path.join(model_dir, "serving"))
    if not os.path.exists(serving_dir) or len(os.listdir(serving_dir)) == 0:
        s = time.time()
        print("SavedModel Format for Tensorflow Serving set-up ...")
        save_hyundai_model_with_visualization(model_dir)
        c = time.time() - s
        print(f"Finished ----- {c:.3f}s")

    s = time.time()
    print("Tensorflow/serving Container creating...")
    for container in client.containers.list():
        if TAG_NAME in container.image.tags:
            container.kill()

    client.containers.run("tensorflow/serving:latest",
                          detach=True,
                          ports={
                              '8500/tcp': str(grpc_port),
                              "8501/tcp": str(rest_port)},
                          environment=['MODEL_NAME=serving'],
                          volumes={
                              serving_dir: {
                                  "bind": "/models/serving",
                                  'mode': 'rw'}})
    c = time.time() - s
    print(f"Finished ----- {c:.3f}s")

    time.sleep(10) #
    s = time.time()
    print("Tensorflow/serving conatiner Checking...")
    image_path = os.path.join(TEST_DIR, "test_input.jpg")
    with open(image_path, 'rb') as f:
        content = f.read()

    content = send_image_to_serving(content, port=grpc_port, verbose=True)
    image = content_to_array(content)
    c = time.time() - s
    print(f"Finished ----- {c:.3f}s")

    print("Launching Model Server.\n"
          f"Connection PORT(GRPC) >>> http://localhost:{grpc_port} \n")


def save_hyundai_model_with_visualization(model_dir,
                                          car_color=(192, 32, 128),
                                          bump_color=(160, 96, 0),
                                          manhole_color=(96, 0, 128),
                                          steel_color=(32, 96, 192),
                                          pothole_color=(96, 32, 128),
                                          other_road_color=(64, 0, 128),
                                          my_road_color=(128, 96, 0),
                                          crack_color=(128, 192, 0),
                                          instance_alpha=0.3,
                                          semantic_alpha=0.3,
                                          default_road_size=3.25,
                                          smoothing_kernel_sizes=(0, 0, 0),
                                          smoothing_weights=(1., 1., 1.)):
    serving_dir = os.path.abspath(os.path.join(model_dir, "serving"))
    weight_path = os.path.join(model_dir, "weights.h5")
    if not os.path.exists(weight_path):
        raise FileExistsError(f"{model_dir} should contain weights.h5")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileExistsError(f'{model_dir} should contain config.json')

    config = ModelConfiguration()
    with open(config_path, 'r') as f:
        config.from_dict(json.load(f))

    config.postprocess.instance_colors = np.array([
        car_color, bump_color, manhole_color,
        steel_color, pothole_color])
    config.postprocess.instance_alpha = instance_alpha

    config.postprocess.semantic_colors = np.array([
        other_road_color, my_road_color, crack_color])
    config.postprocess.semantic_alpha = semantic_alpha

    config.postprocess.default_road_size = default_road_size

    config.postprocess.smoothing_kernel_sizes = smoothing_kernel_sizes
    config.postprocess.smoothing_weights = smoothing_weights

    serving = load_serving_model_from_h5(weight_path, config)
    save_serving_model(serving, serving_dir)


def send_image_to_serving(input_content, url="localhost:8500", verbose=True):
    """
    입력 이미지를 Tensorflow Serving으로 송신하는 Function

    :param input_content: bytes of jpg or png
    :param url: GRPC server URL
    :param verbose:
    :return:
    """
    # (1) Serving의 입력값을 구성하기
    channel = grpc.insecure_channel(url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'serving'
    request.model_spec.signature_name = 'serving_default'

    tensor_proto = tf.make_tensor_proto(input_content) # 이미지 byte 파일을 tensor proto로 변경
    request.inputs['image'].CopyFrom(tensor_proto)

    s = time.time()

    # (2) Serving에 입력값 전달하기
    result_future = stub.Predict.future(request, 10.25)
    response = result_future.result() # Response를 받을 때까지 Waiting

    if verbose:
        print(f"서버에서 처리하는 데 걸린 시간 : {time.time() - s:.3f}")

    # (3) Serving의 출력값 받아오기
    visualize_content = response.outputs['visualize'].string_val[0]
    summary = response.outputs['summarize']

    # (4) Serving의 출력값을 Json Format으로 변경하기
    summary_arr = MakeNdarray(summary)[0]
    summary_df = pd.DataFrame(summary_arr,
                              columns=['name', 'x', 'y', 'w', 'h', "confidence",
                                       'pixelSize', 'estimatedSize',
                                       'estimatedHorizontalLength',
                                       'estimatedVerticalLength',
                                       'includeMyRoad'])
    # Drop NaN Cases
    summary_df = summary_df[summary_df.pixelSize > 0]

    # name을 숫자(0,1,2,3,4,5)에서 이름(car,bump,manhole,steel,pothole,crack)로 변경
    summary_df.loc[:, 'name'] = summary_df.loc[:, 'name'].astype(np.int)
    summary_df.loc[:, 'name'] = summary_df.loc[:, 'name'].map(INSTANCE_LABELS)

    summary_df.loc[:, ['x','y','w','h']] = summary_df.loc[:, ['x','y','w','h']].astype(np.int)
    summary_df.loc[:, 'pixelSize'] = summary_df.loc[:, 'pixelSize'].astype(np.int)
    summary_df.loc[:, 'includeMyRoad'] =summary_df.loc[:, 'includeMyRoad'].astype(np.bool)

    summary_content = {
        "objs": summary_df.to_dict('row')
    }
    return visualize_content, summary_content


def content_to_array(content):
    """
    convert from image to numpy.ndarray

    :param content:
    :return:
    """

    output_image = cv2.imdecode(
        np.fromstring(content, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


def process_folder(input_dir, output_dir, verbose=True):
    """
    input_dir 내 이미지들을 처리하여 output_dir에 처리된 결과들을 저장하는 코드

    :param input_dir: 처리할 이미지가 저장된 폴더
    :param output_dir: 결과가 저장될 폴더
    :param verbose: 진행상황 시각화
    :return:
    """
    # 디렉토리 내 이미지 리스트 가져오기 (CAUTION : jpg, png, jpeg만 모델이 처리가능합니다.)
    fpath_list = glob.glob(os.path.join(input_dir, "*"))
    image_path_list = [file for file in fpath_list
                       if os.path.splitext(file)[-1].lower()
                       in ['.jpg', '.png', '.jpeg']]
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        # Progress Bar을 적용하는 코드
        image_path_list = tqdm(image_path_list)

    for image_path in image_path_list:
        # 1. 이미지 읽어오기
        with open(image_path, 'rb') as f:
            req_content = f.read()
        # 2. 모델을 통해 결과값 수신하기
        visualize_content, summary_content = send_image_to_serving(req_content)

        # 3. 저장할 폴더에 결과를 저장하기
        fname = os.path.split(image_path)[1]

        # 3.1) 시각화된 이미지 정보 저장하기
        save_image_path = os.path.join(output_dir, fname)
        with open(save_image_path, 'wb') as f:
            f.write(visualize_content)
        # 3.2) Summary 정보 저장하기
        save_json_path = os.path.join(output_dir, fname.split('.')[0]+'.json')
        with open(save_json_path, 'w') as f:
            json.dump(summary_content, f)

    return True


__all__ = [
    "run_tensorflow_serving_container",
    "send_image_to_serving",
    "content_to_array",
    "process_folder"
]


if __name__ == "__main__":
    fire.Fire({
        "docker": run_tensorflow_serving_container,
        "save": save_hyundai_model_with_visualization,
        "folder": process_folder})
