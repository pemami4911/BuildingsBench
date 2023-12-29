#!/bin/bash
#SBATCH --job-name=comcap_iid
#SBATCH --output=/projects/foundation/zhaonan/logs/comcap_iid.log
#SBATCH --error=/projects/foundation/zhaonan/logs/comcap_iid.error
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

srun python3 scripts/create_comcap_iid.py