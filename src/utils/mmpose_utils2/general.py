from ..general_utils import *
from ..cocodataset_utils import *

from ..mmdet_utils2 import *

from mmengine.config import Config

def get_MMPose_model(
                        pose_model_name,
                        det_model_name, 

                        pose_checkpoint_path = None, 
                        pose_config_path     = None, 
                        pose_model_root      = "../__model__/04_mmpose/",

                        det_checkpoint_path  = None,
                        det_config_path      = None, 
                        det_cat_ids          = None,
                        det_model_root       = "../__model__/03_mmdetection/",

                        device = 'cuda', 
                    ):
    
    if not torch.cuda.is_available():
        print("Chenge device to CPU")
        device = 'cpu' 

    # Pose Model
    if pose_checkpoint_path is None and pose_config_path is None:
        pose_checkpoint_path, pose_config_path = get_local_public_model(pose_model_name, model_root=pose_model_root, scope="mmpose")

    # Detection Model
    if det_checkpoint_path is None and det_config_path is None:
        det_checkpoint_path, det_config_path = get_local_public_model(det_model_name, model_root=det_model_root, scope="mmdet")

    # build detector
    detector = init_detector(
                                config     = str(det_config_path),
                                checkpoint = str(det_checkpoint_path),
                                device     = device
                            )


    # build pose estimator
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    pose_estimator = init_pose_estimator(
                                            config      = str(pose_config_path),
                                            checkpoint  = str(pose_checkpoint_path),
                                            device      = device,
                                            cfg_options = cfg_options
                                        )
    
    return detector, pose_estimator

def mmpose_inferencer(  detector, 
                        pose_estimator, 
                        img_path:str, 
                        
                        det_cat_ids:list[int] = [0], 
                        bbox_thr     = 0.3,
                        bbox_nms_thr = 0.7,
                        device       = 'cuda'
                    ):

    '''
    bbox_nms_thr : 
        2つのBBoxがどれくらい重複しているかを表す指標の1つ。1.0に近づくほど重複。
        IoUがbbox_nms_thrを超えて重なっているBBoxの集合から、スコアが最大のBBoxを残して、それ以外を除去
    '''
    
    
    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate( (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1 )
    
    tmp = None
    if det_cat_ids is not None:
        for cat_id in det_cat_ids:
            new_bboxes = bboxes[np.logical_and(pred_instance.labels == cat_id, pred_instance.scores > bbox_thr)]
            
            if tmp is None: tmp = new_bboxes
            else:           tmp = np.vstack([tmp, new_bboxes])

    bboxes = tmp[nms(tmp, bbox_nms_thr)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)
    
    return data_samples

def mmpose_result_visualize(    
                                img_path:str, 
                                visualizer,
                                pose_results,
                                kpt_thr = 0.3
                            ):
    
    '''
    document : https://mmpose.readthedocs.io/en/latest/api.html#mmpose.visualization.Pose3dLocalVisualizer.add_datasample
    '''
    
    # show the results
    img = mmcv.imread(img_path, channel_order='rgb')
    
    visualizer.add_datasample(
        name         = 'result',
        image        = img,
        data_sample  = pose_results,
        draw_gt      = True,
        draw_heatmap = True,
        draw_bbox    = True,
        show         = False,
        wait_time    = 0,
        out_file     = None,
        kpt_thr      = kpt_thr
    )
    
    vis_result = visualizer.get_image()

    return vis_result

def create_original_dataset_info(
                                    cfg, 
                                    train_coco, 
                                    kp_class_name=None,
                                    default_joint_weight = 1.0,
                                    default_kp_sigma = 0.1,
                                    dataset_name = 'MyCustomDataset',
                                ):
    
    # dataset_infoの作成
    # https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html
    # https://zhuanlan.zhihu.com/p/646402767
    
    #TODO:
    # custom_dataset_meta = None
    # dataset_meta = parse_pose_metainfo(
    #         dict(from_file='configs/_base_/datasets/coco.py'))
    
    classes = tuple( cat["name"] for cat in train_coco.kw_dataset.cats.values())
    palette = [ (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(len(classes)) ]
    
    if kp_class_name is None:
        kp_class_name = [cat for cat in train_coco.kw_dataset.cats.values() if "keypoints" in cat.keys()][0]["name"]
    
    if kp_class_name not in classes:
        raise ValueError(f"kp_class_name must be in {classes}")
    
    keypoint_cat_info = train_coco.kw_dataset.name_to_cat[kp_class_name]
    keypoints_info = {id:{"name":kp_name, "color":keypoint_cat_info["keypoint_colors"][id]} for id, kp_name in enumerate(keypoint_cat_info["keypoints"])}

    # create dataset_info param
    cfg.dataset_info = dict()
    cfg.dataset_info.dataset_name = dataset_name
    cfg.dataset_info.paper_info = dict()
    
    cfg.dataset_info.keypoint_info = dict()
    for kp_id in keypoints_info:
        cfg.dataset_info.keypoint_info[kp_id] = dict(
            name = keypoints_info[kp_id]["name"],
            id   = kp_id,
            color= html2rgb(keypoints_info[kp_id]["color"]),
            type = "upper",
            swap = "",
        )
        
    cfg.dataset_info.skeleton_info = dict()
    for i, (kp_id1, kp_id2) in enumerate(keypoint_cat_info["skeleton"]):
        kp_id1 = kp_id1 - 1
        kp_id2 = kp_id2 - 1
        
        kp1_rgb_color = html2rgb(keypoints_info[kp_id1]["color"])
        kp2_rgb_color = html2rgb(keypoints_info[kp_id2]["color"])
        
        cfg.dataset_info.skeleton_info[i] = dict(
            link  = (keypoints_info[kp_id1]["name"], keypoints_info[kp_id2]["name"]),
            id    = i,
            color = list(np.array([kp1_rgb_color, kp2_rgb_color]).mean(axis=0).astype(np.uint8)),
        )
        
    cfg.dataset_info.joint_weights = [default_joint_weight for _ in range(len(keypoints_info))]
        
    cfg.dataset_info.sigmas = [default_kp_sigma for _ in range(len(keypoints_info))]
    
    return cfg

def get_det_and_pose_model(
                                train_dir,
                                det_dir_name = "det_model",
                                pose_dir_name = "pose_model",
                                det_model_type = "best",
                                pose_model_type = "best",
                            ):
    
    res = {}
    
    res["det_model_path"], res["det_config_path"] = get_train_model_and_config(Path(train_dir)/det_dir_name, model_type=det_model_type)
    res["pose_model_path"], res["pose_config_path"] = get_train_model_and_config(Path(train_dir)/pose_dir_name, model_type=pose_model_type)
    
    return res

def create_original_mmpose_config(
        base_pose_model_name : str,

        train_coco : kwcoco_v2.COCO_dataset,
        val_coco   : kwcoco_v2.COCO_dataset = None,
        test_coco  : kwcoco_v2.COCO_dataset = None,
        
        kp_class_name : str      = None,

        pose_model_root : Union[str, Path]    = "./models/",
        output_config_name : Union[str, Path] = None,
        output_config_dir  : Union[str, Path] = Path("./custom_config"),

        max_epochs : int      = 20,
        input_img_size        = (512, 512), # 256, 256
        dataloader_batch_size = 32,
        seed:int              = None
    ):

    # 出力するconfigの名前を決定
    if output_config_name is None:
        output_config_name = f"{base_pose_model_name}_{Path(train_coco.json_path).stem}.py"

    # 出力するconfigのパスを決定
    if isinstance(output_config_name, str):
        output_config_name = Path(output_config_name)
    output_config_path = Path(output_config_dir) / output_config_name
    
    # val_coco, test_cocoが指定されていない場合は，train_cocoを利用
    if val_coco is None:
        val_coco = train_coco
    if test_coco is None:
        test_coco = val_coco
    
    # data_rootを決定
    train_dir_parents = set(map(str, list(Path(train_coco.json_path).parents)))
    val_dir_parents   = set(map(str, list(Path(val_coco.json_path).parents)))
    test_dir_parents  = set(map(str, list(Path(test_coco.json_path).parents)))
    data_root = Path(sorted(list(train_dir_parents & val_dir_parents & test_dir_parents), key=len)[-1])
    
    # MMposeのモデルについて
    base_pose_model_checkpoint_path, base_pose_model_config_path = get_local_public_model(base_pose_model_name, model_root=pose_model_root, scope="mmpose")



    # create config
    cfg = Config()
    cfg = Config.fromfile(str(base_pose_model_config_path))
    
    # --------------------------------------------------------------------------- # 
    if base_pose_model_name.split("_")[1] == "hrnet-w48":
        from .train_HRNET import create_hrnet_train_config
        cfg = create_hrnet_train_config(
            base_cfg                        = cfg,
            data_root                       = data_root,
            batch_size                      = dataloader_batch_size,
            max_epochs                      = max_epochs,
            train_coco                      = train_coco,
            val_coco                        = val_coco,
            test_coco                       = test_coco,
            input_img_size                  = input_img_size,
            base_pose_model_checkpoint_path = base_pose_model_checkpoint_path,
            output_config_path              = output_config_path,
            kp_class_name                   = kp_class_name,
            seed                            = seed,
        )
    
    
    elif base_pose_model_name.split("_")[1] == "ViTPose-large-simple":
        from .train_ViTPose import create_vitpose_train_config
        cfg = create_vitpose_train_config(
            base_cfg                        = cfg,
            data_root                       = data_root,
            batch_size                      = dataloader_batch_size,
            max_epochs                      = max_epochs,
            train_coco                      = train_coco,
            val_coco                        = val_coco,
            test_coco                       = test_coco,
            input_img_size                  = input_img_size,
            base_pose_model_checkpoint_path = base_pose_model_checkpoint_path,
            output_config_path              = output_config_path,
            kp_class_name                   = kp_class_name,
            seed                            = seed,
        )
    
    else:
        from .train_all import create_normal_train_config
        cfg = create_normal_train_config(
            base_cfg                        = cfg,
            data_root                       = data_root,
            batch_size                      = dataloader_batch_size,
            max_epochs                      = max_epochs,
            train_coco                      = train_coco,
            val_coco                        = val_coco,
            test_coco                       = test_coco,
            input_img_size                  = input_img_size,
            base_pose_model_checkpoint_path = base_pose_model_checkpoint_path,
            output_config_path              = output_config_path,
            kp_class_name                   = kp_class_name,
            seed                            = seed,
        )
    
    # --------------------------------------------------------------------------- #

    cfg.dump(output_config_path)
    
    return output_config_path