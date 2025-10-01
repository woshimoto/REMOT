PRETRAIN=/home/linzhongxu/code/kdgtrack/DKGTrack-main/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=saved_models_rk/motion_4
OUT='/dkgtrack/DKGTrack-main/exps/saved_models_rk/motion_4'
TRAIN_LOG_FILE=/dkgtrack/DKGTrack-main/exps/saved_models_rk/motion_4/train_log.txt
PID_FILE=/dkgtrack/DKGTrack-main/exps/saved_models_rk/motion_4/train_pid.txt

python3 -m torch.distributed.run --nproc_per_node=3 --master_port 23333 main.py \
   --meta_arch temp_rmot \
   --use_checkpoint \
   --dataset_file e2e_rmot \
   --epoch 20 \
   --with_box_refine \
   --lr_drop 40 \
   --lr 1e-4 \
   --lr_backbone 1e-5 \
   --pretrained ${PRETRAIN}\
   --output_dir exps/${EXP_DIR} \
   --save_dir exps/${EXP_DIR} \
   --batch_size 1 \
   --sample_mode random_interval \
   --sample_interval 1 \
   --sampler_steps 60 80 90 \
   --sampler_lengths 4 4 4 4 \
   --update_query_pos \
   --merger_dropout 0 \
   --dropout 0 \
   --random_drop 0.1 \
   --fp_ratio 0.3 \
   --query_interaction_layer QIM \
   --data_txt_path_train /home/linzhongxu/code/kdgtrack/DKGTrack-main/datasets/data_path/refer-kitti/refer-kitti.train \
   --rmot_path /share2/share/dataset/ref-kitti/ \
   --hist_len 4 \
   --refer_loss_coef 2 \
   #--resume /home/linzhongxu/code/kdgtrack/DKGTrack-main/exps/saved_models_rk/motion_4/checkpoint0099.pth \

   # >"$TRAIN_LOG_FILE" & echo $! >"$PID_FILE"


