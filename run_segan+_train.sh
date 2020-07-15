#!/bin/bash
#
#train on time field
#
# CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/.cudacache' \
# python -u train.py --save_path ppgan_time/model0307/epoch50 \
# 	--clean_trainset data_veu4/expanded_segan1_additive/clean_trainset \
# 	--noisy_trainset data_veu4/expanded_segan1_additive/noisy_trainset \
# 	--cache_dir data_tmp --batch_size 100 --no_bias --epoch 50 --slice_size 256 \
# --l1_weight 0 --g_lr 4e-5 --d_lr 5e-5 --z_dim 512  --dpool_slen 8 --ppgantype ppgan_time


#
# train on frequency field
#

#change dpool_slen ,srate,generator lablefc 

export CUDA_VISIBLE_DEVICES=0
CUDA_CACHE_PATH='~/.cudacache' \
python train.py --save_path ppg_time/model0715/train1 \
	--clean_trainset NewData/clean_trainset \
	--noisy_trainset NewData/noisy_trainset \
	--val_spilt 0.9 \
	--cache_dir data_tmp --batch_size 50 --no_bias \
        --epoch 500  \
        --l1_weight 100 --g_lr 5e-5 --d_lr 1e-5 --z_dim 256 --dpool_slen 31 \
	--opt adam\
	--ppgantype ppgan_freq
	#--clean_valset  NewData/clean_valset \
    #--noisy_valset  NewData/noisy_valset \
