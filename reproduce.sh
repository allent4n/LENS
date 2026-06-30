#!/bin/bash


### MFS-SciSum Data

gdown --id 1dV27LfJSyvr5SHi6tObXWfmc-ZTcrjEB
unzip ASR.zip
mv -r ASR ./data/


gdown --id 1S2UZMpbvmYdcewb1mA272hmFog6O2flh
unzip ASR_feats_all-MiniLM-L6-v2.zip
mv -r ASR_feats_all-MiniLM-L6-v2 ./data/


gdown --id 1njxK74RUvj_Uhu3uwPIHXF0-H1mVNpRA
unzip eva_clip_features_new.zip
mv -r eva_clip_features_new ./data/

#### Pretrained Models
mkdir pretrained_weights
mkdir checkpoints

gdown --id 1ZOBhh10W44lIGNGvfjCEwNQwZF7zvPpm
mv clip4caption_vit-b-32_model.bin ./pretrained_weights/clip4caption_vit-b-32_model.bin

gdown --id 166NPXiOF47YkUoxXwt9VGKGy05IRBy4o
mv eva_clip_psz14.pt ./pretrained_weights/eva_clip_psz14.pt

gdown --id 1M0FWM89-SadGEMAqEc4sodXs-htq8Owh
mv BEST.pth ./checkpoints/BEST.pth


### Reproduce the results
python run.py \
        --data_dir 'data/val_testing/' \
        --video_feature_dir 'data/eva_clip_features_new' \
        --asr_dir 'data/ASR' \
        --asr_feature_dir 'data/ASR_feats_all-MiniLM-L6-v2' \
        --optim adamw \
        --warmup_steps 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-5 \
        --epochs 50 \
        --num_workers 2 \
        --num_beams 5 \
        --train_batch_size 1 \
        --eval_batch_size 1 \
        --task_moment_retrieval \
        --task_memsum \
        --ckpt_dir 'checkpoints' \
        --end_to_end 

python score.py \
--label_path "data/splits/test.json" \
--moment_path "checkpoints/test_moment_retrieval_end_to_end.json" \
--awesome_path "checkpoints/test_memsum_end_to_end.json"
