# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import logging
import os
import os.path as osp
import kwcoco_v2
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

import util

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('-train_coco', '--train_coco_path',       type=str, required=True)
    parser.add_argument('-val_coco'  , '--validation_coco_path',  type=str, required=True)
    parser.add_argument('-base_model', '--base_mmdet_model_name', type=str, required=True)

    parser.add_argument('-eval_coco', '--evaluation_coco_path',   type=str, required=False, default=None)
    parser.add_argument('-epoch',     '--epoch_num',              type=int, required=False, default=10)
    parser.add_argument('-batch',     '--base_batch_size',        type=int, required=False, default=16)
    parser.add_argument('-wd',        '--work-dir',               type=str, required=False, default="./__output__/03_mmdetection/")
    parser.add_argument('-model_root','--model_root',             type=str, required=False, default="./__model__/03_mmdetection/")
    parser.add_argument(              '--seed',                   type=int, required=False, default=None)
    parser.add_argument('-aug',       '--augmentation',           action='store_true', default=False)

    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
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

    return args

def train_mmdet_model(args):
    
    if args.seed is not None:
        from mmengine.runner.utils import set_random_seed
        set_random_seed(args.seed, deterministic=True)
    
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # create work_dir subdir
    dt_now = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    subdir_name = f"{dt_now}_{Path(args.base_mmdet_model_name).stem}_{Path(args.train_coco_path).stem}"
    work_dir = Path(args.work_dir) / subdir_name

    # create config file
    (work_dir/"custom_config").mkdir(parents=True, exist_ok=True)
    config_path = util.create_original_mmdetection_config(
                    train_coco = kwcoco_v2.COCO_dataset(Path(args.train_coco_path)),    
                    val_coco   = kwcoco_v2.COCO_dataset(Path(args.validation_coco_path)),    
                    test_coco  = kwcoco_v2.COCO_dataset(Path(args.evaluation_coco_path)),

                    base_model_name = args.base_mmdet_model_name,

                    max_epochs=args.epoch_num,
                    output_config_dir=work_dir/"custom_config",
                    model_root=args.model_root,
                    dataloader_batch_size = args.base_batch_size,
                    augmentation=args.augmentation,
                    seed=args.seed
                )
    print(config_path)

    # load config
    cfg = Config.fromfile(str(config_path))
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set work_dir 
    cfg.work_dir = str(work_dir)
    
    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        cfg.auto_scale_lr.base_batch_size = args.base_batch_size

        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()
    
    return work_dir

def main():
    args = parse_args()

    train_mmdet_model(args)
    
    return

if __name__ == '__main__':
    main()
