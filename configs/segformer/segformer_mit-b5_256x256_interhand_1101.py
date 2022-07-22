_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/interhand.py',
]

# model settings
model = dict(
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=2))

# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.00006 * 48,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
optimizer_config = dict()

lr_config = dict(
    # _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=250000)
runner = dict(type='IterBasedRunner', max_iters=500000)
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=2500, metric='mIoU', pre_eval=True)
# evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
