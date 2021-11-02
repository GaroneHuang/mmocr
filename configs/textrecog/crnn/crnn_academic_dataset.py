_base_ = []
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# model
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT_BENCHMARK', with_unknown=True, lower=False)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=3),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)

train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 5

# data
img_norm_cfg = dict(mean=[127], std=[127])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=128,
        max_width=128,
        keep_aspect_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'valid_ratio']),
]

dataset_type = 'OCRDataset'

benchmark_pretrain = dict(
    type='OCRDataset',
    img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
    ann_file='data/mixture/Syn90k/label.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

benchmark_train = dict(
    type='OCRDataset',
    img_prefix='data/mixture/Syn90k/mnt/ramdisk/max/90kDICT32px',
    ann_file='data/mixture/Syn90k/label.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

benchmark_val = dict(
    type=dataset_type,
    img_prefix='',
    ann_file='',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'])),
    pipeline=None,
    test_mode=True)

benchmark_test = dict(
    type=dataset_type,
    img_prefix='',
    ann_file='',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'])),
    pipeline=None,
    test_mode=True)


data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[benchmark_pretrain, benchmark_train],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=[benchmark_val],
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=[benchmark_test],
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True
