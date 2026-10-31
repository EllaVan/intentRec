#!/usr/bin/env bash

for dataset in 'MIntRec'
do
    for seed in 2 1 0 2025 2026
    do
        python run.py \
        --dataset MIntRec2.0 \
        --data_path '/media/data2/wf/intentRec/datasets' \
        --logger_name 'mine' \
        --method 'mine' \
        --data_mode 'multi-class' \
        --train \
        --save_model \
        --save_results \
        --seed $seed \
        --gpu_id '1' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone '/media/data2/wf/intentRec/materials/bert-large-uncased' \
        --config_file_name 'mine_MIntRec2' \
        --results_file_name 'mine_MIntRec2.csv'
    done
done
