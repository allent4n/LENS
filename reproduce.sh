#!/bin/bash

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



