# Mixed Precision Training

## Introduction

<!-- [OTHERS] -->

<a href="https://github.com/baidu-research/DeepBench">Official Repo</a>

<a href="https://github.com/open-mmlab/mmcv/blob/v1.3.14/mmcv/runner/hooks/optimizer.py#L134">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/1710.03740">Mixed Precision (FP16) Training (ArXiv'2017)</a></summary>

```latex
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
}
```

</details>

## Results and models

### Cityscapes

| Method     | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                                                                | download                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | -------- | --------- | ------: | -------- | -------------- | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN        | R-101-D8 | 512x1024  |   80000 | 5.37     | 8.64           | 76.80 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fp16/fcn_r101-d8_512x1024_80k_fp16_cityscapes.py)           | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/fcn_r101-d8_512x1024_80k_fp16_cityscapes/fcn_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230921-50245227.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/fcn_r101-d8_512x1024_80k_fp16_cityscapes/fcn_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230921.log.json)                                         |
| PSPNet     | R-101-D8 | 512x1024  |   80000 | 5.34     | 8.77           | 79.46 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fp16/pspnet_r101-d8_512x1024_80k_fp16_cityscapes.py)        | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/pspnet_r101-d8_512x1024_80k_fp16_cityscapes/pspnet_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230919-ade37931.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/pspnet_r101-d8_512x1024_80k_fp16_cityscapes/pspnet_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230919.log.json)                             |
| DeepLabV3  | R-101-D8 | 512x1024  |   80000 | 5.75     | 3.86           | 80.48 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fp16/deeplabv3_r101-d8_512x1024_80k_fp16_cityscapes.py)     | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/deeplabv3_r101-d8_512x1024_80k_fp16_cityscapes/deeplabv3_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230920-bc86dc84.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/deeplabv3_r101-d8_512x1024_80k_fp16_cityscapes/deeplabv3_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230920.log.json)                 |
| DeepLabV3+ | R-101-D8 | 512x1024  |   80000 | 6.35     | 7.87           | 80.46 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fp16/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230920-cc58bc8d.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/fp16/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes_20200717_230920.log.json) |
