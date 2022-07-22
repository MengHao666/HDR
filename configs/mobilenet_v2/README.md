# MobileNetV2: Inverted Residuals and Linear Bottlenecks

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/tensorflow/models/tree/master/research/deeplab">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/mobilenet_v2.py#L14">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/1801.04381">MobileNetV2 (CVPR'2018)</a></summary>

```latex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

</details>

## Results and models

### Cityscapes

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                   | download                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN        | M-V2-D8  | 512x1024  |   80000 |      3.4 | 14.2           | 61.54 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes/fcn_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-d24c28c1.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x1024_80k_cityscapes/fcn_m-v2-d8_512x1024_80k_cityscapes-20200825_124817.log.json)                                         |
| PSPNet     | M-V2-D8  | 512x1024  |   80000 |      3.6 | 11.2           | 70.23 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes/pspnet_m-v2-d8_512x1024_80k_cityscapes_20200825_124817-19e81d51.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x1024_80k_cityscapes/pspnet_m-v2-d8_512x1024_80k_cityscapes-20200825_124817.log.json)                             |
| DeepLabV3  | M-V2-D8  | 512x1024  |   80000 |      3.9 | 8.4            | 73.84 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes/deeplabv3_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-bef03590.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x1024_80k_cityscapes/deeplabv3_m-v2-d8_512x1024_80k_cityscapes-20200825_124836.log.json)                 |
| DeepLabV3+ | M-V2-D8  | 512x1024  |   80000 |      5.1 | 8.4            | 75.20 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes_20200825_124836-d256dd4b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes/deeplabv3plus_m-v2-d8_512x1024_80k_cityscapes-20200825_124836.log.json) |

### ADE20k

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                               | download                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | -------- | --------- | ------: | -------: | -------------- | ----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FCN        | M-V2-D8  | 512x512   |  160000 |      6.5 | 64.4           | 19.71 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k/fcn_m-v2-d8_512x512_160k_ade20k_20200825_214953-c40e1095.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k/fcn_m-v2-d8_512x512_160k_ade20k-20200825_214953.log.json)                                         |
| PSPNet     | M-V2-D8  | 512x512   |  160000 |      6.5 | 57.7           | 29.68 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/pspnet_m-v2-d8_512x512_160k_ade20k.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x512_160k_ade20k/pspnet_m-v2-d8_512x512_160k_ade20k_20200825_214953-f5942f7a.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/pspnet_m-v2-d8_512x512_160k_ade20k/pspnet_m-v2-d8_512x512_160k_ade20k-20200825_214953.log.json)                             |
| DeepLabV3  | M-V2-D8  | 512x512   |  160000 |      6.8 | 39.9           | 34.08 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k/deeplabv3_m-v2-d8_512x512_160k_ade20k-20200825_223255.log.json)                 |
| DeepLabV3+ | M-V2-D8  | 512x512   |  160000 |      8.2 | 43.1           | 34.02 | -             | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3plus_m-v2-d8_512x512_160k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x512_160k_ade20k/deeplabv3plus_m-v2-d8_512x512_160k_ade20k_20200825_223255-465a01d4.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/mobilenet_v2/deeplabv3plus_m-v2-d8_512x512_160k_ade20k/deeplabv3plus_m-v2-d8_512x512_160k_ade20k-20200825_223255.log.json) |
