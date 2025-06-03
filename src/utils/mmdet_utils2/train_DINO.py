from .. import mmdet_utils2

import numpy as np
from pathlib import Path
from typing import Union

import kwcoco_v2
from kwcoco_v2 import COCO_dataset
from kwcoco_v2.visualizer_matplot import VisConfig
from kwcoco_v2.visualizer_matplot import COCO_visualizer

def create_DINO_train_config(
    base_cfg,
    data_root,
    batch_size,
    max_epochs,
    stage2_num_epochs,
    base_lr,
    classes,
    palette,
    train_pipeline,
    train_coco,
    val_coco,
    test_coco,
    input_img_size,
    base_model_checkpoint_path,
    
    repeat_train_dataset:bool,
    seed=0,
):
    
    train_pipeline = [
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(prob=0.5, type='RandomFlip'),
                            
        # Augmentation
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomResize', ratio_range=(1.0, 1.5), scale=input_img_size, keep_ratio=True),
        dict(
            type='AutoAugment',
            policies=[
                [
                    dict(type='Rotate', level=3, img_border_value=(124, 116, 104), prob=1.0),
                ],
                [
                    dict(type='Rotate', level=5, img_border_value=(124, 116, 104), prob=1.0),
                ],
                [
                    dict(type='Rotate', level=7, img_border_value=(124, 116, 104), prob=1.0),
                ],
                [
                    dict(type='Rotate', level=9, img_border_value=(124, 116, 104), prob=1.0),
                ],
            ]),
        dict(type='TranslateX', level=3, prob=0.5, img_border_value=(124, 116, 104)),
        dict(type='TranslateX', level=3, prob=0.5, img_border_value=(124, 116, 104)),
        dict(type='RandomCrop', crop_size=input_img_size),
        
        dict(type='PackDetInputs'),
    ]
    
    val_test_pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=input_img_size, type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ]
    
    
    cfg = base_cfg.copy()
    cfg.data_root = str(data_root)
    cfg.train_batch_size_per_gpu = batch_size
    cfg.train_num_workers = 1
    cfg.max_epochs        = {max_epochs}
    cfg.stage2_num_epochs = {stage2_num_epochs}
    cfg.base_lr           = {base_lr}
    cfg.metainfo = {
        'classes': classes,
        'palette': palette,
    }    
    cfg.randomness = dict(seed=seed)
    
    cfg.train_dataloader.batch_size               = cfg.train_batch_size_per_gpu
    cfg.train_dataloader.num_workers              = cfg.train_num_workers
    cfg.train_dataloader.dataset.data_root        = cfg.data_root
    cfg.train_dataloader.dataset.metainfo         = cfg.metainfo
    cfg.train_dataloader.dataset.data_prefix.img  = f'{Path(train_coco.dataset_dir).relative_to(data_root)}/'
    cfg.train_dataloader.dataset.ann_file         = str(Path(train_coco.json_path).relative_to(data_root))
    cfg.train_dataloader.dataset.pipeline         = train_pipeline
    if repeat_train_dataset:
        cfg.train_dataloader.dataset.dataset = cfg.train_dataloader.dataset
    
    cfg.val_dataloader.batch_size                 = cfg.train_batch_size_per_gpu
    cfg.val_dataloader.num_workers                = cfg.train_num_workers
    cfg.val_dataloader.dataset.data_root          = cfg.data_root
    cfg.val_dataloader.dataset.metainfo           = cfg.metainfo
    cfg.val_dataloader.dataset.data_prefix.img    = f'{Path(val_coco.dataset_dir).relative_to(data_root)}/'
    cfg.val_dataloader.dataset.ann_file           = str(Path(val_coco.json_path).relative_to(data_root))
    cfg.val_dataloader.dataset.pipeline           = val_test_pipeline
    
    cfg.test_dataloader.batch_size                 = cfg.train_batch_size_per_gpu
    cfg.test_dataloader.num_workers                = cfg.train_num_workers
    cfg.test_dataloader.dataset.data_root          = cfg.data_root
    cfg.test_dataloader.dataset.metainfo           = cfg.metainfo
    cfg.test_dataloader.dataset.data_prefix.img    = f'{Path(test_coco.dataset_dir).relative_to(data_root)}/'
    cfg.test_dataloader.dataset.ann_file           = str(Path(test_coco.json_path).relative_to(data_root))
    cfg.test_dataloader.dataset.pipeline           = val_test_pipeline

    cfg.val_evaluator.ann_file  = str(Path(val_coco.json_path))
    cfg.test_evaluator.ann_file = str(Path(test_coco.json_path))

    cfg.model.bbox_head.num_classes = len(classes)
    
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=5, max_keep_ckpts=2, save_best='auto')
    cfg.default_hooks.logger     = dict(type='LoggerHook',     interval=5)

    # learning rate
    # cfg.param_scheduler = [
    #     dict(
    #         type='LinearLR',
    #         start_factor=1.0e-5,
    #         by_epoch=False,
    #         begin=0,
    #         end=10),
    #     dict(
    #         # use cosine lr from 10 to 20 epoch
    #         type='CosineAnnealingLR',
    #         eta_min=base_lr * 0.05,
    #         begin=max_epochs // 2,
    #         end=max_epochs,
    #         T_max=max_epochs // 2,
    #         by_epoch=True,
    #         convert_to_iter_based=True),
    # ]
    
    # pipeline
    cfg.train_pipeline = train_pipeline
    # cfg.train_pipeline_stage2 = train_pipeline
    cfg.test_pipeline = val_test_pipeline

    # # optimizer
    # cfg.optim_wrapper = dict(
    #     type='OptimWrapper',
    #     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    #     paramwise_cfg=dict(
    #         norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
    # )

    # cfg.default_hooks = dict(
    #     checkpoint=dict(
    #         type='CheckpointHook',
    #         interval=5,
    #         max_keep_ckpts=2,  # only keep latest 2 checkpoints
    #         save_best='auto'
    #     ),
    #     logger=dict(type='LoggerHook', interval=5))

    # cfg.custom_hooks = [
    #     dict(
    #         type='PipelineSwitchHook',
    #         switch_epoch=max_epochs - stage2_num_epochs,
    #         switch_pipeline=cfg.train_pipeline_stage2)
    # ]

    # load COCO pre-trained weight
    cfg.load_from = str(base_model_checkpoint_path)

    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
    cfg.visualizer = dict(
        name='visualizer',
        type='DetLocalVisualizer',
        vis_backends = [
            dict(type='LocalVisBackend'),
            dict(type='TensorboardVisBackend')
        ]
    )
    
    cfg.auto_scale_lr = dict(
        enable=False,
        base_batch_size=batch_size,
    )
    
    cfg.log_processor.by_epoch = True

    return cfg