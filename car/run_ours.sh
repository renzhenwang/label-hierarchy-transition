#!/bin/bash

# deploy parameters for running our methods
data_dir='../data_target/Cars'           ## path to images
ratio=0.1                    ## relabeled ratio for semisupervised/semi-supervised setting
GPU_ID='0'                   ## device id
relabel='family'             ## chosen from ['order', 'family']
loss_type='hierarchy_kl'
beta=0.01                     ## trade-off between cross-entropy and confusion loss
# beta=2.0                    ## set as ratio > 0.1

# regular deploy parameters
epochs=200
batch_size=8
crop_size=448
scale_size=550
learning_rate=0.002
workers=4

# run under three random seeds: 10, 100, 1
for seed in 10 100 1; do
    echo 'Execute 3 times'
    echo $seed
    out_dir='result/ours@ratio_'${ratio}'_seed_'${seed}
    # snapshot=$out_dir'/model_best.pth.tar'
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./main_ours.py \
        ${data_dir} \
        --ratio ${ratio} \
        --loss_type ${loss_type} \
        -b ${batch_size} \
        -j ${workers} \
        --lr ${learning_rate} \
        --epochs ${epochs} \
        --crop_size ${crop_size} \
        --scale_size ${scale_size} \
        --relabel ${relabel} \
        --out ${out_dir} \
        --beta ${beta} \
        --seed ${seed} \
        # --snapshot ${snapshot}
done