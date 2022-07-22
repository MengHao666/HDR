# BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/ycszen/TorchSeg/tree/master/model/bisenet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/backbones/bisenetv1.py#L266">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/1808.00897">BiSeNetV1 (ECCV'2018)</a></summary>

```latex
@inproceedings{yu2018bisenet,
  title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={325--341},
  year={2018}
}
```

</details>

## Results and models

### Cityscapes

| Method    | Backbone  | Crop Size | Lr schd | Mem (GB) | Inf time (fps) |  mIoU | mIoU(ms+flip) | config                                                                                  | download                                                                                                                                                                                                                                                       |
| --------- | --------- | --------- | ------: | -------- | -------------- | ----: | ------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BiSeNetV1 (No Pretrain) | R-18-D32 | 1024x1024  | 160000 | 5.69 | 31.77 | 74.44 | 77.05 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes_20210922_172239-c55e78e2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes_20210922_172239.log.json) |
| BiSeNetV1| R-18-D32 | 1024x1024  | 160000 | 5.69 | 31.77 | 74.37 | 76.91 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210905_220251-8ba80eff.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210905_220251.log.json) |
| BiSeNetV1 (4x8) | R-18-D32 | 1024x1024  | 160000 | 11.17 | 31.77 | 75.16 | 77.24 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes_20210905_220322-bb8db75f.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes/bisenetv1_r18-d32_in1k-pre_4x8_1024x1024_160k_cityscapes_20210905_220322.log.json) |
| BiSeNetV1 (No Pretrain) | R-50-D32 | 1024x1024  | 160000 | 15.39 | 7.71 | 76.92 | 78.87 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes_20210923_222639-7b28a2a6.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_4x4_1024x1024_160k_cityscapes_20210923_222639.log.json) |
| BiSeNetV1 | R-50-D32 | 1024x1024  | 160000 | 15.39 | 7.71 | 77.68 | 79.57 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628-8b304447.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628.log.json) |

Note:

- `4x8`: Using 4 GPUs with 8 samples per GPU in training.
- Default setting is 4 GPUs with 4 samples per GPU in training.
- `No Pretrain` means the model is trained from scratch.
