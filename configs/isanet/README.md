# Interlaced Sparse Self-Attention for Semantic Segmentation

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/openseg-group/openseg.pytorch">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.18.0/mmseg/models/decode_heads/isa_head.py#L58">Code Snippet</a>

<details>
<summary align="right"><a href="https://arxiv.org/abs/1907.12273">ISANet (ArXiv'2019/IJCV'2021)</a></summary>

```
@article{huang2019isa,
  title={Interlaced Sparse Self-Attention for Semantic Segmentation},
  author={Huang, Lang and Yuan, Yuhui and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={arXiv preprint arXiv:1907.12273},
  year={2019}
}

The technical report above is also presented at:
@article{yuan2021ocnet,
  title={OCNet: Object Context for Semantic Segmentation},
  author={Yuan, Yuhui and Huang, Lang and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={International Journal of Computer Vision},
  pages={1--24},
  year={2021},
  publisher={Springer}
}
```

</details>

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config |download |
| --------|----------|-----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ISANet | R-50-D8  | 512x1024  |  40000  |  5.869   |  2.91   | 78.49 |   79.44   |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_512x1024_40k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_40k_cityscapes/isanet_r50-d8_512x1024_40k_cityscapes_20210901_054739-981bd763.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_40k_cityscapes/isanet_r50-d8_512x1024_40k_cityscapes_20210901_054739.log.json)                                         |
| ISANet | R-50-D8  | 512x1024  |  80000  |  5.869  |  2.91          | 78.68 |   80.25   |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_512x1024_80k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_80k_cityscapes/isanet_r50-d8_512x1024_80k_cityscapes_20210901_074202-89384497.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x1024_80k_cityscapes/isanet_r50-d8_512x1024_80k_cityscapes_20210901_074202.log.json)                                         |
| ISANet | R-50-D8  | 769x769   |  40000  |  6.759   |  1.54          | 78.70 |   80.28      |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_769x769_40k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_40k_cityscapes/isanet_r50-d8_769x769_40k_cityscapes_20210903_050200-4ae7e65b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_40k_cityscapes_20210903_050200.log.json)                                         |
| ISANet | R-50-D8  | 769x769   |  80000  |  6.759   |  1.54          | 79.29 |   80.53    |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_769x769_80k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_80k_cityscapes/isanet_r50-d8_769x769_80k_cityscapes_20210903_101126-99b54519.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_769x769_80k_cityscapes/isanet_r50-d8_769x769_80k_cityscapes_20210903_101126.log.json)                                         |
| ISANet | R-101-D8 | 512x1024  |  40000  |  9.425   |  2.35         | 79.58 |   81.05  |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_512x1024_40k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x1024_40k_cityscapes/isanet_r101-d8_512x1024_40k_cityscapes_20210901_145553-293e6bd6.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet_r101-d8_512x1024_40k_cityscapes/isanet_r101-d8_512x1024_40k_cityscapes_20210901_145553.log.json)                                         |
| ISANet | R-101-D8 | 512x1024  |  80000  |  9.425   |  2.35          | 80.32 |  81.58  |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_512x1024_80k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x1024_80k_cityscapes/isanet_r101-d8_512x1024_80k_cityscapes_20210901_145243-5b99c9b2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x1024_80k_cityscapes/isanet_r101-d8_512x1024_80k_cityscapes_20210901_145243.log.json)                                         |
| ISANet | R-101-D8 | 769x769   |  40000  |  10.815  |  0.92         | 79.68 |    80.95     |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_769x769_40k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_40k_cityscapes/isanet_r101-d8_769x769_40k_cityscapes_20210903_111320-509e7224.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_40k_cityscapes/isanet_r101-d8_769x769_40k_cityscapes_20210903_111320.log.json)                                         |
| ISANet | R-101-D8 | 769x769   |  80000  |  10.815  |  0.92          | 80.61 |   81.59  |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_769x769_80k_cityscapes.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_80k_cityscapes/isanet_r101-d8_769x769_80k_cityscapes_20210903_111319-24f71dfa.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_769x769_80k_cityscapes/isanet_r101-d8_769x769_80k_cityscapes_20210903_111319.log.json)                                         |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config |download |
| --------|----------|-----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ISANet | R-50-D8  | 512x512  |  80000  |  9.0    |    22.55      | 41.12 |   42.35     |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_512x512_80k_ade20k.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_80k_ade20k/isanet_r50-d8_512x512_80k_ade20k_20210903_124557-6ed83a0c.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_80k_ade20k/isanet_r50-d8_512x512_80k_ade20k_20210903_124557.log.json)|
| ISANet | R-50-D8  | 512x512  |  160000 |  9.0     |    22.55        | 42.59 |  43.07      |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_512x512_160k_ade20k.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_160k_ade20k/isanet_r50-d8_512x512_160k_ade20k_20210903_104850-f752d0a3.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_160k_ade20k/isanet_r50-d8_512x512_160k_ade20k_20210903_104850.log.json)|
| ISANet | R-101-D8 | 512x512  |  80000  |  12.562   |    10.56       | 43.51 |  44.38        |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_512x512_80k_ade20k.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_80k_ade20k/isanet_r101-d8_512x512_80k_ade20k_20210903_162056-68b235c2.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_80k_ade20k/isanet_r101-d8_512x512_80k_ade20k_20210903_162056.log.json)|
| ISANet | R-101-D8 | 512x512  |  160000 |  12.562  |    10.56       | 43.80 |   45.4       |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_512x512_160k_ade20k.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_160k_ade20k/isanet_r101-d8_512x512_160k_ade20k_20210903_211431-a7879dcd.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_160k_ade20k/isanet_r101-d8_512x512_160k_ade20k_20210903_211431.log.json)|

### Pascal VOC 2012 + Aug

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU | mIoU(ms+flip) | config |download |
| --------|----------|-----------|-----------|--------:|----------|----------------|------:|--------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ISANet | R-50-D8  | 512x512  |  20000  |   5.9  |   23.08       | 76.78 |   77.79    |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_512x512_20k_voc12aug.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_20k_voc12aug/isanet_r50-d8_512x512_20k_voc12aug_20210901_164838-79d59b80.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_20k_voc12aug/isanet_r50-d8_512x512_20k_voc12aug_20210901_164838.log.json)|
| ISANet | R-50-D8  | 512x512  |  40000  |   5.9  |   23.08        | 76.20 |    77.22       |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r50-d8_512x512_40k_voc12aug.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_40k_voc12aug/isanet_r50-d8_512x512_40k_voc12aug_20210901_151349-7d08a54e.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r50-d8_512x512_40k_voc12aug/isanet_r50-d8_512x512_40k_voc12aug_20210901_151349.log.json)|
| ISANet | R-101-D8 | 512x512  |  20000  |   9.465  |   7.42        | 78.46 |     79.16    |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_512x512_20k_voc12aug.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_20k_voc12aug/isanet_r101-d8_512x512_20k_voc12aug_20210901_115805-3ccbf355.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_20k_voc12aug/isanet_r101-d8_512x512_20k_voc12aug_20210901_115805.log.json)|
| ISANet | R-101-D8 | 512x512  |  40000  |   9.465  |   7.42         | 78.12 |     79.04    |[config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/isanet/isanet_r101-d8_512x512_40k_voc12aug.py)|[model](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_40k_voc12aug/isanet_r101-d8_512x512_40k_voc12aug_20210901_145814-bc71233b.pth) &#124; [log](https://download.openmmlab.com/mmsegmentation/v0.5/isanet/isanet_r101-d8_512x512_40k_voc12aug/isanet_r101-d8_512x512_40k_voc12aug_20210901_145814.log.json)|
