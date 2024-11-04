# dataset settings
dataset_type = 'SeaPersonDataset'  # CocoDataset
data_root = '../data/tiny_set_v2/'  # 数据集路径
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_sp'),  # change by hui
    dict(type='LoadAnnotations_sp', with_bbox=True),  # change by hui
    dict(type='Resize_sp', scale_factor=[1.0], keep_ratio=True),  # add by hui
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle_sp'),  # change by hui
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),  # change by hui
]
test_pipeline = [
    dict(type='LoadImageFromFile_sp'),  # change by hui
    dict(
        type='CroppedTilesFlipAug',  # add by hui
        tile_shape=(640, 640),  # sub image size by cropped
        tile_overlap=(100, 100),
        scale_factor=[1.0],

        flip=False,
        transforms=[
            dict(type='Resize_sp', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        min_gt_size=2,  # add
        type=dataset_type,
        ann_file=data_root + 'anns_realease_rgb/release/corner/rgb_trainvalid_w640h640ow100oh100.json',  # 训练json
        img_prefix=data_root + 'imgs_rgb',  # 图片路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'anns_realease_rgb/release/rgb_test.json',  # 验证json
        img_prefix=data_root + 'imgs_rgb',  # 图片路径
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'anns_realease_rgb/release/rgb_test.json',  # 测试json
        img_prefix=data_root + 'imgs_rgb',  # 图片路径
        pipeline=test_pipeline))

# evaluation = dict(interval=1, metric='bbox')  #change
# tiny bbox eval with IOD
evaluation = dict(
    interval=3, metric='bbox',
    iou_thrs=[0.25, 0.5, 0.75],  # set None mean use 0.5:1.0::0.05
    proposal_nums=[1000],
    cocofmt_kwargs=dict(
        ignore_uncertain=True,
        use_ignore_attr=True,
        use_iod_for_ignore=True,
        iod_th_of_iou_f="lambda iou: iou",  #"lambda iou: (2*iou)/(1+iou)",
        cocofmt_param=dict(
            evaluate_standard='tiny',  # or 'coco'
            # iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
            # maxDets=[200],              # set this same as set evaluation.proposal_nums
        )
    )
)
