# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
import os
import os.path as osp
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmpose.datasets.datasets.base import BaseCocoStyleDataset

import util
import kwcoco_v2

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')

    parser.add_argument('-train_coco', '--train_coco_path',       type=str, required=True)
    parser.add_argument('-val_coco'  , '--validation_coco_path',  type=str, required=True)
    parser.add_argument('-det_model',  '--base_det_model',        type=str, required=True)
    parser.add_argument('-pose_model', '--base_pose_model',       type=str, required=True)

    parser.add_argument('-eval_coco', '--evaluation_coco_path',   type=str, required=False, default=None)
    parser.add_argument('-epoch',     '--epoch_num',              type=int, required=False, default=10)
    parser.add_argument('-batch',     '--base_batch_size',        type=int, required=False, default=16)
    parser.add_argument('-wd',        '--work-dir',               type=str, required=False, default="./__output__/04_mmpose/")
    parser.add_argument(              '--det_model_root',         type=str, required=False, default="./__model__/03_mmdetection/")
    parser.add_argument(              '--pose_model_root',        type=str, required=False, default="./__model__/04_mmpose/")
    parser.add_argument(              '--val_interval',           type=int, required=False, default=10)
    
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
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

def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # enable automatic-mixed-precision training
    if args.amp is True:
        from mmengine.optim import AmpOptimWrapper, OptimWrapper
        optim_wrapper = cfg.optim_wrapper.get('type', OptimWrapper)
        assert optim_wrapper in (OptimWrapper, AmpOptimWrapper), \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

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

    # visualization
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    # set train_cfg
    cfg.train_cfg.max_epochs   = args.epoch_num
    cfg.train_cfg.val_interval = args.val_interval

    return cfg

def train_mmpose_model(args):
    # create work_dir subdir
    dt_now = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    det_model_name  = "-".join(Path(args.base_det_model).stem.split('_')[:2])
    pose_model_name = "-".join(Path(args.base_pose_model).stem.split('_')[:2])
    dataset_name = Path(args.train_coco_path).stem
    
    subdir_name = f"{dt_now}_{det_model_name}_{pose_model_name}_{dataset_name}"
    work_dir = Path(args.work_dir) / subdir_name
    
    (work_dir/"custom_config").mkdir(parents=True, exist_ok=True)
    
    
    config_path = util.create_original_mmpose_config(
                    train_coco = kwcoco_v2.COCO_dataset(Path(args.train_coco_path)),    
                    val_coco   = kwcoco_v2.COCO_dataset(Path(args.validation_coco_path)),    
                    test_coco  = kwcoco_v2.COCO_dataset(Path(args.evaluation_coco_path)),

                    base_pose_model_name = args.base_pose_model,

                    output_config_dir=work_dir/"custom_config",
                    pose_model_root=args.pose_model_root,
                    dataloader_batch_size = args.base_batch_size
                )
    
    

    # load config
    cfg = Config.fromfile(str(config_path))

    # merge CLI arguments to config
    cfg = merge_args(cfg, args)
    cfg.work_dir = str(work_dir)
    
    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor',
                             cfg.get('preprocess_cfg', {}))

    cfg.dump(config_path)
    
    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
    
    
    return work_dir


def main():
    args = parse_args()
    
    train_mmpose_model(args)
    
    return


if __name__ == '__main__':
    main()
