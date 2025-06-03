# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import os
import os.path as osp
from pathlib import Path

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


from mmengine.config import Config, DictAction
from mmpose.datasets.datasets.base import BaseCocoStyleDataset

import util
import kwcoco_v2

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')

    parser.add_argument('-train_coco',  '--train_coco_path',        type=str, required=True)
    parser.add_argument('-val_coco'  ,  '--validation_coco_path',   type=str, required=True)
    parser.add_argument('-eval_coco',   '--evaluation_coco_path',   type=str, required=False, default=None)
    
    parser.add_argument(                '--det_model_root',         type=str, required=False, default="../__model__/03_mmdetection/")
    parser.add_argument(                '--pose_model_root',        type=str, required=False, default="../__model__/04_mmpose/")
    parser.add_argument('-wd',          '--work-dir',               type=str, required=False, default="../__output__/04_mmpose/")
    
    parser.add_argument('-det_model',   '--base_det_model_name',    type=str, required=False, default=None)
    parser.add_argument(                '--det_model_config_path',  type=str, required=False, default=None)
    parser.add_argument(                '--det_model_model_path',   type=str, required=False, default=None)
    parser.add_argument('-det_epoch',   '--det_model_epoch',        type=int, required=False, default=10)
    parser.add_argument('-det_batch',   '--det_batch_size',         type=int, required=False, default=16)
    parser.add_argument('-det_select',  '--select_det_model',       type=str, required=False, default="best") # "best" or "last"
    parser.add_argument(                '--not_det_model_train',    action='store_true',      default=False)
    
    
    parser.add_argument('-pose_model',  '--base_pose_model_name',   type=str, required=False, default=None)
    parser.add_argument(                '--pose_model_config_path', type=str, required=False, default=None)
    parser.add_argument(                '--pose_model_model_path',  type=str, required=False, default=None)
    parser.add_argument('-pose_epoch',  '--pose_model_epoch',       type=int, required=False, default=10)
    parser.add_argument('-pose_batch',  '--pose_batch_size',        type=int, required=False, default=16)
    parser.add_argument(                '--val_interval',           type=int, required=False, default=10)
    parser.add_argument('-pose_select', '--select_pose_model',      type=str, required=False, default="best") # "best" or "last"s
    parser.add_argument(                '--not_pose_model_train',   action='store_true',      default=False)


    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)


    if args.evaluation_coco_path is None:
        args.evaluation_coco_path = args.validation_coco_path
        
    if args.base_det_model_name is None and (args.det_model_config_path is None and args.det_model_model_path is None):
        raise ValueError("Please specify --base_det_model_name and --det_model_config_path or --det_model_model_path")
        
    if args.base_pose_model_name is None and (args.pose_model_config_path is None and args.pose_model_model_path is None):
        raise ValueError("Please specify --base_pose_model_name and --pose_model_config_path or --pose_model_model_path")
        
        
    if args.base_det_model_name is None and (args.det_model_config_path is not None):
        args.base_det_model_name = Path(args.det_model_config_path).stem
        
    if args.base_pose_model_name is None and (args.pose_model_config_path is not None):
        args.base_pose_model_name = Path(args.pose_model_config_path).stem

    return args

def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    cfg.launcher = args.launcher

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True
        
    # set train_cfg
    cfg.train_cfg.max_epochs   = args.pose_model_epoch
    cfg.train_cfg.val_interval = args.val_interval

    return cfg

def train_mmdet_model(args, base_work_dir):
    
    print()
    print()
    print("===========================================")
    print("===========================================")
    print("train detection model")
    print("===========================================")
    print("===========================================")
    print()
    
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()
    
    det_work_dir = base_work_dir / "det_model"
    (det_work_dir/"custom_config").mkdir(parents=True, exist_ok=True)
    
    det_config_path = util.create_original_mmdetection_config(
                        train_coco = kwcoco_v2.COCO_dataset(Path(args.train_coco_path)),    
                        val_coco   = kwcoco_v2.COCO_dataset(Path(args.validation_coco_path)),    
                        test_coco  = kwcoco_v2.COCO_dataset(Path(args.evaluation_coco_path)),

                        base_model_name = args.base_det_model_name,

                        max_epochs=args.det_model_epoch,
                        output_config_dir=det_work_dir/"custom_config",
                        model_root=args.det_model_root,
                        dataloader_batch_size = args.det_batch_size
                    )
    
    # load config
    det_cfg = Config.fromfile(str(det_config_path))
    det_cfg.launcher = args.launcher

    # set work_dir 
    det_cfg.work_dir = str(det_work_dir)

    # enable automatically scaling LR
    if args.auto_scale_lr:
        det_cfg.auto_scale_lr.base_batch_size = args.base_batch_size

        if 'auto_scale_lr' in det_cfg and \
                'enable' in det_cfg.auto_scale_lr and \
                'base_batch_size' in det_cfg.auto_scale_lr:
            det_cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                            '"auto_scale_lr.enable" or '
                            '"auto_scale_lr.base_batch_size" in your'
                            ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        det_cfg.resume = True
        det_cfg.load_from = None
    elif args.resume is not None:
        det_cfg.resume = True
        det_cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in det_cfg:
        # build the default runner
        runner = Runner.from_cfg(det_cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(det_cfg)

    # start training
    runner.train()
    
    model_path, config_path = util.get_train_model_and_config(det_work_dir, model_type=args.select_det_model)
    
    return model_path, config_path

def train_mmpose_model(args, base_work_dir):
    
    print()
    print()
    print("===========================================")
    print("===========================================")
    print("train pose model")
    print("===========================================")
    print("===========================================")
    print()
        
    
    pose_work_dir = base_work_dir / "pose_model"
    (pose_work_dir/"custom_config").mkdir(parents=True, exist_ok=True)
    
    
    config_path = util.create_original_mmpose_config(
                    train_coco = kwcoco_v2.COCO_dataset(Path(args.train_coco_path)),    
                    val_coco   = kwcoco_v2.COCO_dataset(Path(args.validation_coco_path)),    
                    test_coco  = kwcoco_v2.COCO_dataset(Path(args.evaluation_coco_path)),

                    base_pose_model_name = args.base_pose_model_name,

                    output_config_dir=pose_work_dir/"custom_config",
                    pose_model_root=args.pose_model_root,
                    dataloader_batch_size = args.pose_batch_size
                )


    # load config
    cfg = Config.fromfile(str(config_path))

    # merge CLI arguments to config
    cfg = merge_args(cfg, args)
    cfg.work_dir = str(pose_work_dir)
    
    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault(   
                                'data_preprocessor',
                                cfg.get('preprocess_cfg', {})
                            )

    cfg.dump(config_path)
    
    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
    
    model_path, config_path = util.get_train_model_and_config(pose_work_dir, model_type=args.select_pose_model)
    
    return model_path, config_path

def train_mmdet_and_mmpose_model(args):
    # create work_dir subdir
    dt_now = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    det_model_name  = "-".join(Path(args.base_det_model_name).stem.split('_')[:2])
    pose_model_name = "-".join(Path(args.base_pose_model_name).stem.split('_')[:2])
    dataset_name = Path(args.train_coco_path).stem
    
    subdir_name = f"{dt_now}_{det_model_name}_{pose_model_name}_{dataset_name}"
    work_dir = Path(args.work_dir) / subdir_name
    
    
    ## train detection model
    if args.base_det_model_name is not None and ((args.det_model_config_path is None) or (args.det_model_model_path is None)):
        if not args.not_det_model_train:
            det_model_path, det_config_path = train_mmdet_model(args, work_dir)
    else:
        det_model_path  = args.det_model_model_path
        det_config_path = args.det_model_config_path

    
    ## train pose model
    if not args.not_pose_model_train:
        pose_model_path, pose_config_path = train_mmpose_model(args, work_dir)
    
    return work_dir


def main():
    args = parse_args()
    
    train_mmdet_and_mmpose_model(args)
    
    return


if __name__ == '__main__':
    main()
