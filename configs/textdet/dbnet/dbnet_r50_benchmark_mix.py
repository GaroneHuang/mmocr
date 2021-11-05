_base_ = []

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=True)
total_epochs = 10

model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPNC', in_channels=[256, 512, 1024, 2048], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        text_repr_type='quad',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True)),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'TextDetDataset'
data_root = '../mmocr_datasets/det'
# from official dbnet code
img_norm_cfg = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[255, 255, 255],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # img aug
    dict(
        type='ImgAug',
        args=[['Resize', {'shorter-side':[600, 672, 704, 736, 768, 800, 832, 864, 896], 'longer-side':1600}],
              dict(cls='Affine', rotate=[-10, 10])]),
    # random crop
    dict(type='EastRandomCrop', target_size=(736, 736)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    # for visualizing img and gts, pls set visualize = True
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1600, 1000), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/mix_labels.json',
        img_prefix=data_root + '/images',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])
        ),
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/val_labels.json',
        img_prefix=data_root + '/images',
        # select_first_k=100,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/test_labels.json',
        img_prefix=data_root + '/images',
        # select_first_k=100,
        pipeline=test_pipeline,
        test_mode=True)
    )
evaluation = dict(interval=2, metric='hmean-iou')

cudnn_benchmark = True
