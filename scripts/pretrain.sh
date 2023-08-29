#!/bin/bash

# THIS IS AN EXAMPLE SCRIPT. 
# PLEASE CONFIGURE FOR YOUR SETUP.
NUM_GPUS=1

# torchrun \
#     --nnodes=1 \
#     --nproc_per_node=$NUM_GPUS \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=localhost:0 \
#     scripts/pretrain.py --config your_model

# CUDA_VISIBLE_DEVICES=3 torchrun \

#     --nnodes=1 \

#     --nproc_per_node=$NUM_GPUS \

#     --rdzv-backend=c10d \

#     --rdzv-endpoint=localhost:0 \

#     scripts/pretrain.py --config TransformerWithTokenizer-weather-S --disable_slurm --note trucs_special_run

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    scripts/pretrain.py --config TransformerWithGaussian-weather-S --disable_slurm --note test_run
