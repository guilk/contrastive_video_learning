#!/usr/bin/env bash

#python3 youcook_data_loader.py --split_dir /home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/data_splits \
#--src_cap /home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/data_splits/all_data.lst \
#--vfeat_dir /home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/features \
#--is_train --num_roi 128

feat_type=resnet
model=${feat_type}_3head_t_rnn_v_linear_cnn2_dp03_vin_norm

mkdir model/$model

python3 main.py --split_dir /home/liangkeg/third_hand/data/YouCookII/meta_data/data_splits \
--src_cap /home/liangkeg/third_hand/data/YouCookII/meta_data/data_splits/all_data.lst \
--vfeat_dir /home/liangkeg/third_hand/data/YouCookII/meta_data/features/${feat_type}_features \
--is_train --num_roi 128 --model_name $model --glove --vis_encoder linear --text_encoder rnn \
--model_dir /home/liangkeg/third_hand/data/YouCookII/models \
--cnn2 --vin_norm --batch_size 64 --vfeat_type ${feat_type}

python3 main.py --split_dir /home/liangkeg/third_hand/data/YouCookII/meta_data/data_splits \
--src_cap /home/liangkeg/third_hand/data/YouCookII/meta_data/data_splits/all_data.lst \
--vfeat_dir /home/liangkeg/third_hand/data/YouCookII/meta_data/features/${feat_type}_features \
--load_best --num_roi 128 --glove --model_name $model --vis_encoder linear --text_encoder rnn \
--model_dir /home/liangkeg/third_hand/data/YouCookII/models \
--cnn2 --vin_norm --batch_size 64 --vfeat_type ${feat_type}

#python3 main.py --split_dir /home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/data_splits \
#--src_cap /home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/data_splits/all_data.lst \
#--vfeat_dir /home/liangkeg/gpu4ssd/third_hand/YouCookII/meta_data/features \
#--load_best --num_roi 128 --glove --model_name $model --vis_encoder linear --text_encoder rnn --cnn2 --vin_norm --model_ckpt_name debug_2.pt