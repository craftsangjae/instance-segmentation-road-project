# RetinaMask 구현하기

## Objective

> 주행 중인 자동차의 영상에서 도로, 자동차, 맨홀 등을 실시간으로 파악하는 프로젝트. 실시간으로 파악할 수 있게 하기 위해서 우리는 Single-Shot Multi Detector 계통이면서, Instance Segmentation이 되는 `Retina-Mask`을 구현

## 디렉토리 구성

```` markdown
|- models/ : RetinaMask & DeepLab V3+ Model에 관련된 정보들 
     |- layers/ : RetinaMask & DeepLab V3+ 모델을 구성하는 Keras Custom Layers
          |- detection.py : RetinaMask에 관련된 Keras Custom Layer
          |- segmentation.py : DeepLab V3+에 관련된 Keras Custom Layer
     |- backbone/ : ResNet, VGG, MobileNEt 등 pretrained Model을 불러오는 Method.
          |- base.py : Pretrained Model을 Load하고 weight들을 freeze하는 메소드들.
          |- MobileNetv2.py : MobileNetV2을 호출하는 메소드
          |- ResNext.py : ResNext을 호출하는 메소드
     |- callbacks.py : Keras Custom Callbacks, CycleLR이 구현되어 있음
     |- losses.py : RetinaMaks & DeepLab V3+ 모델에 관련된 Custom Loss Function이 구현되어 있음
     |- metrics.py : Keras Custom Metrics, Layer형태로 구현되어 있음
     |- normalization.py : GroupNormalization이 정의되어 있음 
     |- optimizers.py : AdamW & SGDW가 정의되어 있음
     |- prior.py : RetinMask의 Prior box(Anchor)에 대한 Configuration Class
|- datasets/ : MNIST, FashionMnist, Project 데이터셋이 저장되는 공간
|- utils/ : 모델의 학습 데이터셋 및 시각화에 관련된 데이터 공간
     |- dataset/ : 모델의 학습 데이터셋을 불러오는 클래스가 구현
     |- generator/ : 모델의 학습 데이터셋을 Keras Model에 feed해주는 Generator 클래스
     |- visualize.py : 모델 결과의 시각화를 담당
     |- noise.py : 모델 학습에 노이즈를 주는 메소드들
|- misc/ : 모델을 디버깅할 때 이용하는 잡다한 데이터들
|- scripts/ : DeepLab V3+ & RetinaMask를 통해 작업을 수행하는 것을 기록한 Jupyter Notebook Scripts
|- examples/ : 모델 동작에 대한 여러 Sample Scripts 
````

## Reference

#### Paper

1. [retinamask](https://arxiv.org/pdf/1901.03353.pdf)
2. [retinanet](https://arxiv.org/abs/1708.02002)
3. [SSD](https://arxiv.org/abs/1512.02325)
4. [Mask R-CNN](https://arxiv.org/abs/1703.06870)

