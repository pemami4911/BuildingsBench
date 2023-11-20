#!/bin/bash

#SBATCH --job-name=clip
#SBATCH --output=/projects/foundation/zhaonan/logs/clip_test.log
#SBATCH --error=/projects/foundation/zhaonan/logs/clip_test.error
#SBATCH --account=foundation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=29
#SBATCH --partition=gpu
#SBATCH --mem=100gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zli4@nrel.gov


source /nopt/nrel/apps/210929a/myenv.2110041605

module purge
module load conda
module load cuda/11.7.0
module load openmpi


conda deactivate
conda activate /projects/foundation/pemami/conda/foundation

export BUILDINGS_BENCH=/projects/foundation/eulp/v1.1.0/BuildingsBench
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export WANDB_PROJECT=clip_pretrain
export WANDB_ENTITY=nrel-ml-foundations

export WORLD_SIZE=1
NUM_GPUS=1

torchrun --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        scripts/surrogate_clip_pretrain.py \
        --config CLIP \
        --train_idx_filename train_simcap_300k.idx \
        --val_idx_filename val_simcap_300k.idx \
        --use-weather