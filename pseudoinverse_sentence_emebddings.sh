#!/bin/bash -e

#SBATCH --job-name=inversion
#SBATCH --output=inversion_%j.out
#SBATCH --error=inversion_%j.err
#SBATCH --mem=50GB
#SBATCH --time=6-00:00:00

set -x


wd=$(pwd)
echo "working directory ${wd}"

export HF_HOME="${wd}/.cache"
export HF_DATASETS_CACHE="${wd}/.cache/datasets"
export DATASET_CACHE_PATH="${wd}/.cache"
export DISABLE_APEX=1



SIF=/home/cs.aau.dk/ng78zb/pytorch_23.10-py3.sif
echo "sif ${SIF}"


srun singularity exec --nv --cleanenv --bind ${wd}:${wd} ${SIF} \
    python src/pseudoinverse_sentence_embeddings.py
