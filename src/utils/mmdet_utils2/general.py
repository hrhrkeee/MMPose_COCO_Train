from ..general_utils import *
from ..cocodataset_utils import *

# Docs : https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/inference.html

def get_local_public_model(model_name:str, model_root="./models/", scope="mmdet"):

    if ".py" in model_name:
        model_search = list((Path(model_root).glob(f"{Path(model_name).stem}*.pth")))
    else:
        model_search = list((Path(model_root).glob(f"{model_name}*.pth")))

    # MMDetectionのモデルについて
    if len(model_search) > 0:
        checkpoint_path = model_search[0]
    else:
        checkpoint_path = download(package=scope, configs=[model_name], dest_root=model_root)[0]
        checkpoint_path = Path(model_root)/checkpoint_path

    if not checkpoint_path.exists():
        raise("Checkpoint not found.")
    

    # MMDetectionのConfigファイルについて
    config_name = f"{model_name}.py"
    config_path = Path(checkpoint_path).parent / config_name
    
    if not config_path.exists():
        raise("Config not found.")

    return checkpoint_path, config_path

def get_MM_inferencer(  model_name, 
                        checkpoint_path=None, 
                        config_path=None, 
                        device = 'cuda', 
                        model_root = "../__model__/03_mmdetection/",
                        palette = 'none',
                    ):
    
    if not torch.cuda.is_available():
        print("Chenge device to CPU")
        device = 'cpu' 

    if checkpoint_path is None and config_path is None:
        checkpoint_path, config_path = get_local_public_model(model_name, model_root=model_root)

    inferencer = DetInferencer(
                    model         = str(config_path),
                    weights       = str(checkpoint_path),
                    device        = device,
                    scope         = 'mmdet',
                    palette       = palette,
                    show_progress = False
            )
    
    return inferencer

def check_MMconfig_dataset_in_dataset(config_path):

    cfg = Config.fromfile(str(config_path))

    if "dataset" in cfg["train_dataloader"]["dataset"].keys():
        return True

    return False

def get_base_dataset_pipeline(input_img_size:tuple):
    
    dataset_pipeline = [
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
    
    return dataset_pipeline

def get_augment_dataset_pipeline(input_img_size:tuple):
    
    dataset_pipeline = [
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(keep_ratio=True, scale=input_img_size, type='Resize'),
                            dict(type='LoadAnnotations', with_bbox=True),
                            
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
    
    return dataset_pipeline

def get_train_model_and_config(train_work_dir, model_type="best"):

    if model_type not in ["best", "last"]:
        raise ValueError("model_type must be 'best' or 'last'")

    train_work_dir = Path(train_work_dir)

    model_path = None
    config_path = list(train_work_dir.glob("*.py"))[0]

    if model_type == "best":
        model_path = list(train_work_dir.glob("best_*.pth"))[0]

    elif model_type == "last":
        with open(list(train_work_dir.glob("last_checkpoint"))[0], "r") as f:
            last_checkpoint = Path(f.read())

        model_path = list(train_work_dir.glob(last_checkpoint.name))[0]

    return model_path, config_path

def create_original_mmdetection_config(
        base_model_name : str,

        train_coco : kwcoco_v2.COCO_dataset,
        val_coco   : kwcoco_v2.COCO_dataset = None,
        test_coco  : kwcoco_v2.COCO_dataset = None,

        max_epochs : int        = 20,
        stage2_num_epochs : int = 1,
        base_lr : float         = 0.00008,

        model_root : Union[str, Path]         = "./models/",
        output_config_name : Union[str, Path] = None,
        output_config_dir  : Union[str, Path] = Path("./custom_config"),

        input_img_size = (512, 512), # 256, 256
        dataloader_batch_size = 32,
        augmentation = False,
        seed:int = None
    ):

    if output_config_name is None:
        output_config_name = f"{base_model_name}_{Path(train_coco.json_path).stem}.py"

    if isinstance(output_config_name, str):
        output_config_name = Path(output_config_name)
    output_config_path = Path(output_config_dir) / output_config_name

    if val_coco is None:
        val_coco = train_coco
    if test_coco is None:
        test_coco = val_coco
    
    train_dir_parents = set(map(str, list(Path(train_coco.json_path).parents)))
    val_dir_parents   = set(map(str, list(Path(val_coco.json_path).parents)))
    test_dir_parents  = set(map(str, list(Path(test_coco.json_path).parents)))
    data_root = Path(sorted(list(train_dir_parents & val_dir_parents & test_dir_parents), key=len)[-1])

    classes = tuple( cat["name"] for cat in train_coco.kw_dataset.cats.values())
    palette = [ (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(classes)) ]

    base_model_checkpoint_path, base_model_config_path = get_local_public_model(base_model_name, model_root=model_root)

    repeat_train_dataset = check_MMconfig_dataset_in_dataset(base_model_config_path)
    
    if augmentation:
        train_pipeline = get_augment_dataset_pipeline(input_img_size=input_img_size)
    else:
        train_pipeline = get_base_dataset_pipeline(input_img_size=input_img_size)
    
    # create config
    cfg = Config()
    cfg = Config.fromfile(str(base_model_config_path))
    
    # --------------------------------------------------------------------------- # 
    
    if False:
        pass

    ### 精度が下がってしまった...
    # if base_model_name.split("_")[0] == "faster-rcnn":
    #     from .train_FasterRCNN import create_FasterRCNN_train_config
    #     cfg = create_FasterRCNN_train_config(
    #         base_cfg                   = cfg,
    #         data_root                  = data_root,
    #         batch_size                 = dataloader_batch_size,
    #         max_epochs                 = max_epochs,
    #         stage2_num_epochs          = stage2_num_epochs,
    #         base_lr                    = base_lr,
    #         classes                    = classes,
    #         palette                    = palette,
    #         train_pipeline             = train_pipeline,
    #         train_coco                 = train_coco,
    #         val_coco                   = val_coco,
    #         test_coco                  = test_coco,
    #         input_img_size             = input_img_size,
    #         base_model_checkpoint_path = base_model_checkpoint_path,
    #         repeat_train_dataset       = repeat_train_dataset,
    #         seed                       = seed,
    #     )
    
    # elif base_model_name.split("_")[0] == "efficientdet":
    #     from .train_DETR import create_DETR_train_config
    #     cfg = create_DETR_train_config(
    #         base_cfg                   = cfg,
    #         data_root                  = data_root,
    #         batch_size                 = dataloader_batch_size,
    #         max_epochs                 = max_epochs,
    #         stage2_num_epochs          = stage2_num_epochs,
    #         base_lr                    = base_lr,
    #         classes                    = classes,
    #         palette                    = palette,
    #         train_pipeline             = train_pipeline,
    #         train_coco                 = train_coco,
    #         val_coco                   = val_coco,
    #         test_coco                  = test_coco,
    #         input_img_size             = input_img_size,
    #         base_model_checkpoint_path = base_model_checkpoint_path,
    #         repeat_train_dataset       = repeat_train_dataset,
    #         seed                       = seed,
    #     )
    
    # elif base_model_name.split("_")[0] == "detr":
    #     from .train_EfficientDet import create_EfficientDet_train_config
    #     cfg = create_EfficientDet_train_config(
    #         base_cfg                   = cfg,
    #         data_root                  = data_root,
    #         batch_size                 = dataloader_batch_size,
    #         max_epochs                 = max_epochs,
    #         stage2_num_epochs          = stage2_num_epochs,
    #         base_lr                    = base_lr,
    #         classes                    = classes,
    #         palette                    = palette,
    #         train_pipeline             = train_pipeline,
    #         train_coco                 = train_coco,
    #         val_coco                   = val_coco,
    #         test_coco                  = test_coco,
    #         input_img_size             = input_img_size,
    #         base_model_checkpoint_path = base_model_checkpoint_path,
    #         repeat_train_dataset       = repeat_train_dataset,
    #         seed                       = seed,
    #     )
    
    # elif base_model_name.split("_")[0] == "dino-4scale":
    #     from .train_DINO import create_DINO_train_config
    #     cfg = create_DINO_train_config(
    #         base_cfg                   = cfg,
    #         data_root                  = data_root,
    #         batch_size                 = dataloader_batch_size,
    #         max_epochs                 = max_epochs,
    #         stage2_num_epochs          = stage2_num_epochs,
    #         base_lr                    = base_lr,
    #         classes                    = classes,
    #         palette                    = palette,
    #         train_pipeline             = train_pipeline,
    #         train_coco                 = train_coco,
    #         val_coco                   = val_coco,
    #         test_coco                  = test_coco,
    #         input_img_size             = input_img_size,
    #         base_model_checkpoint_path = base_model_checkpoint_path,
    #         repeat_train_dataset       = repeat_train_dataset,
    #         seed                       = seed,
    #     )
    
    else:
        from .train_all import create_normal_train_config
        cfg = create_normal_train_config(
            base_cfg                   = cfg,
            data_root                  = data_root,
            batch_size                 = dataloader_batch_size,
            max_epochs                 = max_epochs,
            stage2_num_epochs          = stage2_num_epochs,
            base_lr                    = base_lr,
            classes                    = classes,
            palette                    = palette,
            train_pipeline             = train_pipeline,
            train_coco                 = train_coco,
            val_coco                   = val_coco,
            test_coco                  = test_coco,
            input_img_size             = input_img_size,
            base_model_checkpoint_path = base_model_checkpoint_path,
            repeat_train_dataset       = repeat_train_dataset,
            seed                       = seed,
        )
    
    # --------------------------------------------------------------------------- # 

    cfg.dump(output_config_path)
    
    return output_config_path
