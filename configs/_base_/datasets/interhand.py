# dataset settings
dataset_type = 'InterHand_Dataset'

data_root_H = '/mnt/lustre/menghao/dataset/InterHand2.6M/5fps/1030_SEG_DATA_Interhand_GTbbx/human_annot'
data_root_M = '/mnt/lustre/menghao/dataset/InterHand2.6M/5fps/1030_SEG_DATA_Interhand_GTbbx/machine_annot'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='My_LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0),
    dict(type='My_RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(256, 256),
#         # img_scale=None,
#         img_ratios=[1.0],
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip', prob=0),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]



### modified on 2021/11/23, for offline inference only , not for train or test
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # img_scale=None,
        img_ratios=[1.0],
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip', prob=0),
            # dict(type='Normalize', **img_norm_cfg),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

### train,val,test in right hand center crop images
dataset_train_H = dict(
    type=dataset_type,
    data_root=data_root_H,
    img_dir='train/images',
    ann_dir=['train/labels/l_amodal', 'train/labels/l_visible', 'train/labels/r_amodal', 'train/labels/r_visible'],
    pipeline=train_pipeline)

dataset_train_M = dict(
    type=dataset_type,
    data_root=data_root_M,
    img_dir='train/images',
    ann_dir=['train/labels/l_amodal', 'train/labels/l_visible', 'train/labels/r_amodal', 'train/labels/r_visible'],
    pipeline=train_pipeline)

dataset_val_M = dict(
    type=dataset_type,
    data_root=data_root_M,
    img_dir='val/mini_images',
    ann_dir=['val/labels/l_amodal', 'val/labels/l_visible', 'val/labels/r_amodal', 'val/labels/r_visible'],
    # split=['val/labels/mini_M_val.txt','val/labels/mini_M_val.txt','val/labels/mini_M_val.txt','val/labels/mini_M_val.txt'],
    pipeline=test_pipeline)

dataset_test_H = dict(
    type=dataset_type,
    data_root=data_root_H,
    img_dir='test/images',
    ann_dir=['test/labels/l_amodal', 'test/labels/l_visible', 'test/labels/r_amodal', 'test/labels/r_visible'],
    pipeline=test_pipeline)

dataset_test_M = dict(
    type=dataset_type,
    data_root=data_root_M,
    img_dir='test/images',
    ann_dir=['test/labels/l_amodal', 'test/labels/l_visible', 'test/labels/r_amodal', 'test/labels/r_visible'],
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=48,
    workers_per_gpu=4,
    train=[
        dataset_train_H,
        dataset_train_M
    ],
    val=dataset_val_M,
    test=dataset_test_M)
