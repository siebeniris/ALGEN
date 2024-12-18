#!/bin/bash -e

#SBATCH --job-name=sinkhorn
#SBATCH --output=sinkhorn_%j.out
#SBATCH --error=sinkhorn_%j.err
#SBATCH --mem=50GB
#SBATCH --time=6-00:00:00

set -x

sourcemodel=$1

wd=$(pwd)
echo "working directory ${wd}"

export HF_HOME="${wd}/.cache"
export HF_DATASETS_CACHE="${wd}/.cache/datasets"
export DATASET_CACHE_PATH="${wd}/.cache"
export DISABLE_APEX=1



SIF=/home/cs.aau.dk/ng78zb/pytorch_23.10-py3.sif
echo "sif ${SIF}"


srun singularity exec --nv --cleanenv --bind ${wd}:${wd} ${SIF} \
    python src/normal_equation_optimal_transport_grid_search_hyperparameter.py ${sourcemodel}
