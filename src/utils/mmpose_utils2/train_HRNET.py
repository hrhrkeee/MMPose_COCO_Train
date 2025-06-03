from .. import mmpose_utils2

import numpy as np
from pathlib import Path
from typing import Union

import kwcoco_v2
from kwcoco_v2 import COCO_dataset
from kwcoco_v2.visualizer_matplot import VisConfig
from kwcoco_v2.visualizer_matplot import COCO_visualizer

def create_hrnet_train_config(
    base_cfg,
    data_root,
    batch_size,
    max_epochs,
    train_coco,
    val_coco,
    test_coco,
    input_img_size,
    base_pose_model_checkpoint_path,
    output_config_path,
    kp_class_name,
    seed=0,
):
    
    cfg = base_cfg.copy()
    
    # general config
    cfg.data_mode = "topdown"
    cfg.data_root = str(data_root)
    cfg.dataset_type = 'CocoDataset'
    
    cfg.randomness = dict(seed=seed)

    cfg.default_hooks.checkpoint.interval = 5
    cfg.default_hooks.checkpoint.max_keep_ckpts = 2
    cfg.default_hooks.checkpoint.save_best = 'auto'
    cfg.default_hooks.logger.type = 'LoggerHook'
    cfg.default_hooks.logger.interval = 5
    
    cfg.train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

    # train dataset config
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.dataset.data_root = str(data_root)
    cfg.train_dataloader.dataset.data_prefix.img = str(Path(train_coco.dataset_dir).relative_to(data_root))
    cfg.train_dataloader.dataset.ann_file = str(Path(train_coco.json_path).relative_to(data_root))
    cfg.train_dataloader.dataset.type = 'CocoDataset'
    cfg.train_dataloader.dataset.metainfo = dict(from_file=str(output_config_path))

    # val dataset config
    cfg.val_dataloader.batch_size = batch_size
    cfg.val_dataloader.dataset.data_root = str(data_root)
    cfg.val_dataloader.dataset.data_prefix.img = str(Path(val_coco.dataset_dir).relative_to(data_root))
    cfg.val_dataloader.dataset.ann_file = str(Path(val_coco.json_path).relative_to(data_root))
    cfg.val_dataloader.dataset.bbox_file = None
    cfg.val_dataloader.dataset.type = 'CocoDataset'
    cfg.val_dataloader.dataset.metainfo = dict(from_file=str(output_config_path))

    # test dataset config
    cfg.test_dataloader.batch_size = batch_size
    cfg.test_dataloader.dataset.data_root = str(data_root)
    cfg.test_dataloader.dataset.data_prefix.img = str(Path(test_coco.dataset_dir).relative_to(data_root))
    cfg.test_dataloader.dataset.ann_file = str(Path(test_coco.json_path).relative_to(data_root))
    cfg.test_dataloader.dataset.bbox_file = None
    cfg.test_dataloader.dataset.type = 'CocoDataset'
    cfg.test_dataloader.dataset.metainfo = dict(from_file=str(output_config_path))
    
    # val evaluator config
    cfg.val_evaluator.ann_file = str(Path(val_coco.json_path))
    cfg.val_evaluator.type     = 'CocoMetric'
    
    # test evaluator config
    cfg.test_evaluator.ann_file = str(Path(test_coco.json_path))
    cfg.test_evaluator.type     = 'CocoMetric'
    
    # load COCO pre-trained weight
    cfg.load_from = str(base_pose_model_checkpoint_path)
    
    # model_info fix
    classes = tuple( cat["name"] for cat in train_coco.kw_dataset.cats.values())
    if kp_class_name is None:
        kp_class_name = [cat for cat in train_coco.kw_dataset.cats.values() if "keypoints" in cat.keys()][0]["name"]
    if kp_class_name not in classes:
        raise ValueError(f"kp_class_name must be in {classes}")
    keypoint_cat_info = train_coco.kw_dataset.name_to_cat[kp_class_name]
    
    cfg.model.head.out_channels = len(keypoint_cat_info["keypoints"])
    
    # tensorboard
    cfg.visualizer.vis_backends += [dict(type='TensorboardVisBackend')]
    
    cfg = mmpose_utils2.create_original_dataset_info(cfg, train_coco, kp_class_name=kp_class_name)
    
    return cfg