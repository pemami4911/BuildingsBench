#!/bin/bash

#SBATCH --job-name=simcap_train
#SBATCH --output=/projects/foundation/zhaonan/logs/transformer.log
#SBATCH --error=/projects/foundation/zhaonan/logs/transformer.error
#SBATCH --account=foundation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=29
#SBATCH --partition=gpu
#SBATCH --mem=100gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ssinha@nrel.gov


source /nopt/nrel/apps/210929a/myenv.2110041605

module purge
module load conda
module load cuda/11.7.0
module load openmpi


conda deactivate
conda activate /projects/foundation/ssinha/conda/simcaps

export BUILDINGS_BENCH=/projects/foundation/eulp/v1.1.0/BuildingsBench
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export WANDB_PROJECT=simcap
export WANDB_ENTITY=nrel-ml-foundations

srun python3 scripts/surrogate_train.py \
        --config sequentialsurrogateMLP \
        --train_idx_filename train_simcap_300k.idx \
        --val_idx_filename val_simcap_300k.idx \
        --use-weather \
        --use_building_description \
        --ignore_scoring_rules 
