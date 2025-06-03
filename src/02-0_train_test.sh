source ~/anaconda3/etc/profile.d/conda.sh
conda activate ymc

work_dir="../__output__/08_MMPose/test/"

# pose_model="td-hm_hrnet-w48_8xb32-210e_coco-256x192" # VRAM : 3000 MB
# pose_model="td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192" # VRAM : 6000 MB
pose_model="td-hm_ViTPose-small_8xb64-210e_coco-256x192"
# pose_model="td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256" # VRAM : 4000 MB

epoch=5
batch_size=4
seed=42
pose_model_root="../__model__/04_mmpose/"

# find `pwd` -maxdepth 1 | sort
train_dataset_paths=(
    "../__dataset__/07_MMDetection/02_GTt+GENX_generate_num_compare/00_Ground_Truth_train_n50"
)

for dataset_dir in ${train_dataset_paths[@]}
do
    echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
    echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
    echo $dataset_dir

    python "./02_train_MMPose_model.py"\
        -train_coco  $dataset_dir/train/train.json\
        -val_coco    $dataset_dir/valid/valid.json\
        -eval_coco   $dataset_dir/test/test.json\
        -pose_model  $pose_model\
        -epoch       $epoch\
        -batch       $batch_size\
        --work-dir   $work_dir\
        --model_root $pose_model_root\
        --seed       $seed\
        --auto-scale-lr \
        \

    echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
    echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+"
done
