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

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT_BENCHMARK', with_unknown=True, lower=False)

model = dict(
    type='SARNet',
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=3),
    encoder=dict(
        type='SAREncoder',
        enc_bi_rnn=False,
        enc_do_rnn=0.1,
        enc_gru=False,
    ),
    decoder=dict(
        type='ParallelSARDecoder',
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True),
    loss=dict(type='SARLoss'),
    label_convertor=label_convertor,
    max_seq_len=30)

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'valid_ratio',
            'img_norm_cfg', 'ori_filename'
        ])
]

dataset_type = 'OCRDataset'

benchmark_pretrain = dict(
    type='OCRDataset',
    img_prefix='data/recg/images/pretrain',
    ann_file='data/recg/annotations/pretrain_labels.lmdb',
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
    img_prefix='data/recg/images/train',
    ann_file='data/recg/annotations/train_labels.lmdb',
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
    img_prefix='data/recg/images/val',
    ann_file='data/recg/annotations/val_labels.lmdb',
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
    img_prefix='data/recg/images/test',
    ann_file='data/recg/annotations/test.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['filename', 'text'])),
    pipeline=None,
    test_mode=True)

data = dict(
    workers_per_gpu=4,
    samples_per_gpu=64,
    train=dict(
        type='UniformConcatDataset',
        datasets=[benchmark_pretrain, benchmark_train],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset', datasets=[benchmark_val], pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset', datasets=[benchmark_test], pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True
