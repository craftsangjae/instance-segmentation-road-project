"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import json
import os
import time
from engine.backbone import freeze_backbone
from engine.config import ModelConfiguration
from engine.utils.generator import MaskLabGenerator
from engine.parallel import ParallelModel
from engine.optimizers import RectifiedAdam
from engine.callbacks import SaveInferenceModel, CyclicLR
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend as K
from multiprocessing import cpu_count
from engine.retinamasklab import construct_masklabdataset
from engine.retinamasklab import construct_masklab_networks


def train_masklab_model(config:ModelConfiguration):
    train_config = config.train
    save_dir = train_config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    """
    설정 정보 저장하기
    """
    s = time.time()
    print("--- Configuration 정보를 저장 ---")
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f)
    print(f"--- 소요시간 :{time.time()-s:.3f}s")

    """
    데이터 파이프라인 구성하기
    """
    s = time.time()
    print("--- 데이터 파이프라인 구성하기 ---")
    trainset, validset = construct_masklabdataset(config)
    scale_ratio = train_config.scale_ratio
    batch_size = train_config.batch_size
    traingen = MaskLabGenerator(trainset.config,
                                scale_ratio=scale_ratio,
                                batch_size=batch_size)
    if isinstance(scale_ratio,float) or isinstance(scale_ratio,int):
        valid_scale_ratio = scale_ratio
    else:
        valid_scale_ratio = sum(scale_ratio) / len(scale_ratio)
    valid_batch_size = batch_size * 4
    validgen = MaskLabGenerator(validset.config,
                                scale_ratio=valid_scale_ratio,
                                batch_size=valid_batch_size)
    print(f" --- 소요시간 :{time.time()-s:.3f}s")

    """
    모델 구성하기
    """
    s = time.time()
    print("--- MaskLab 모델 구성하기 ---")
    trainer, inferencer = construct_masklab_networks(config)
    print(f" --- 소요시간 :{time.time()-s:.3f}s")

    """
    CallBack 함수 구성하기
    """
    s = time.time()
    print("--- 콜백 함수 구성 ---")
    model_dir = os.path.join(save_dir, "engine/")
    save_cb = SaveInferenceModel(model_dir, inferencer)
    tensorboard_cb = TensorBoard(save_dir)
    print(f"--- 소요시간 :{time.time()-s:.3f}s")

    """
    학습가능한지 확인하기
    """
    s = time.time()
    print("Check Able To Fit Dataset ")
    if isinstance(scale_ratio, float) or isinstance(scale_ratio, int):
        max_scale_ratio = scale_ratio
    else:
        max_scale_ratio = scale_ratio[1]

    checkgen = MaskLabGenerator(trainset.config,
                                scale_ratio=max_scale_ratio,
                                batch_size=batch_size,)
    # Full Trainable
    freeze_backbone(trainer,
                    model_type=config.backbone.backbone_type,
                    freeze_depth="C0")

    if train_config.gpu_count > 1:
        parallel = ParallelModel(trainer, train_config.gpu_count)
    else:
        parallel = trainer

    # Set Loss & Metric
    for tensor, name in zip(parallel.output, parallel.output_names):
        if 'loss' in name:
            parallel.add_loss(K.mean(tensor))
        parallel.add_metric(tensor, aggregation='mean', name=name)
    parallel.compile(RectifiedAdam(1e-10))
    parallel.summary()

    parallel.fit_generator(checkgen, steps_per_epoch=1, verbose=0)
    parallel.fit_generator(checkgen, steps_per_epoch=10, verbose=1,
                           use_multiprocessing=train_config.use_multiprocessing,
                           workers=cpu_count(),
                           callbacks=[tensorboard_cb, save_cb])
    print(f"Finish Checking Network --- consumed time : {time.time()-s:.3f}s")

    """
    5. Train Head-Tune Network
    """
    epoch_index = 0

    if train_config.train_head_tune:
        s = time.time()
        print("Start Compiling Network for Head Tune")
        freeze_backbone(trainer,
                        model_type=config.backbone.backbone_type,
                        freeze_depth=train_config.train_head_level)
        if train_config.gpu_count > 1:
            parallel = ParallelModel(trainer, train_config.gpu_count)
        else:
            parallel = trainer

        # Set Loss & Metric
        for tensor, name in zip(parallel.output, parallel.output_names):
            if 'loss' in name:
                parallel.add_loss(K.mean(tensor))
            parallel.add_metric(tensor, aggregation='mean', name=name)

        clr_cb = CyclicLR(base_lr=train_config.head_base_lr,
                          max_lr=train_config.head_max_lr,
                          step_size=train_config.head_step_size)
        callbacks = [save_cb, tensorboard_cb, clr_cb]

        # Compile Model
        parallel.compile(RectifiedAdam(1e-4))
        parallel.summary()
        print(f"Finish Compiling Network for Head Tune --- consumed time : {time.time()-s:.3f}s")
        print("Start Training Network for Head Tune")
        parallel.fit_generator(checkgen, steps_per_epoch=1, verbose=0)
        hist = parallel.fit_generator(traingen, epochs=epoch_index+train_config.train_head_tune_epoch,
                                      initial_epoch=epoch_index,
                                      steps_per_epoch=train_config.head_step_size//2,
                                      validation_data=validgen,
                                      use_multiprocessing=train_config.use_multiprocessing,
                                      workers=cpu_count(),
                                      max_queue_size=100,
                                      callbacks=callbacks)
        epoch_index = max(hist.epoch)
        print(f"Finish Training Network for Head Tune --- consumed time : {time.time()-s:.3f}s")

    """
    6. Train Waist-Tune Network
    """
    if train_config.train_waist_tune:
        s = time.time()
        print("Start Compiling Network for Waist Tune")
        freeze_backbone(trainer,
                        model_type=config.backbone.backbone_type,
                        freeze_depth=train_config.train_waist_level)
        if train_config.gpu_count > 1:
            parallel = ParallelModel(trainer, train_config.gpu_count)
        else:
            parallel = trainer

        # Set Loss & Metric
        for tensor, name in zip(parallel.output, parallel.output_names):
            if 'loss' in name:
                parallel.add_loss(K.mean(tensor))
            parallel.add_metric(tensor, aggregation='mean', name=name)

        clr_cb = CyclicLR(base_lr=train_config.waist_base_lr,
                          max_lr=train_config.waist_max_lr,
                          step_size=train_config.waist_step_size)
        callbacks = [save_cb, tensorboard_cb, clr_cb]

        # Compile Model
        parallel.compile(RectifiedAdam(1e-4))
        parallel.summary()
        print(f"Finish Compiling Network for Waist Tune --- consumed time : {time.time()-s:.3f}s")
        print("Start Training Network for Waist Tune")
        parallel.fit_generator(checkgen, steps_per_epoch=1, verbose=0)
        hist = parallel.fit_generator(traingen, epochs=epoch_index+train_config.train_waist_tune_epoch,
                                      initial_epoch=epoch_index,
                                      steps_per_epoch=train_config.waist_step_size//2,
                                      validation_data=validgen,
                                      use_multiprocessing=train_config.use_multiprocessing,
                                      workers=cpu_count(),
                                      max_queue_size=100,
                                      callbacks=callbacks)
        epoch_index = max(hist.epoch)
        print(f"Finish Training Network for Waist Tune --- consumed time : {time.time()-s:.3f}s")

    """
    7. Train All-Tune Network
    """
    if train_config.train_all:
        s = time.time()
        print("Start Compiling Network for All Tune")
        freeze_backbone(trainer,
                        model_type=config.backbone.backbone_type,
                        freeze_depth="C0")

        if train_config.gpu_count > 1:
            parallel = ParallelModel(trainer, train_config.gpu_count)
        else:
            parallel = trainer

        # Set Loss & Metric
        for tensor, name in zip(parallel.output, parallel.output_names):
            if 'loss' in name:
                parallel.add_loss(K.mean(tensor))
            parallel.add_metric(tensor, aggregation='mean', name=name)

        clr_cb = CyclicLR(base_lr=train_config.all_base_lr,
                          max_lr=train_config.all_max_lr,
                          step_size=train_config.all_step_size)
        callbacks = [save_cb, tensorboard_cb, clr_cb]

        # Compile Model
        parallel.compile(RectifiedAdam(1e-4))
        parallel.summary()
        print(f"Finish Compiling Network for All Tune --- consumed time : {time.time()-s:.3f}s")
        print("Start Training Network for All Tune")
        parallel.fit_generator(checkgen, steps_per_epoch=1, verbose=0)
        parallel.fit_generator(traingen, epochs=epoch_index+train_config.train_all_epoch,
                               initial_epoch=epoch_index,
                               steps_per_epoch=train_config.all_step_size//2,
                               validation_data=validgen,
                               use_multiprocessing=train_config.use_multiprocessing,
                               workers=cpu_count(),
                               max_queue_size=100,
                               callbacks=callbacks)
        print(f"Finish Training Network for All Tune --- consumed time : {time.time()-s:.3f}s")

