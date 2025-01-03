#!/bin/bash -e

#SBATCH --job-name=attacker
#SBATCH --output=attacker_%j.out
#SBATCH --error=attacker_%j.err
#SBATCH --mem=50GB
#SBATCH --time=6-00:00:00

set -x


wd=$(pwd)
echo "working directory ${wd}"

export HF_HOME="${wd}/.cache"
export HF_DATASETS_CACHE="${wd}/.cache/datasets"
export DATASET_CACHE_PATH="${wd}/.cache"
export DISABLE_APEX=1

CHECKPOINT_PATH=$1
SOURCE_MODEL_NAME=$2
TRAIN_SAMPLES=$3
TEST_SAMPLES=$4


SIF=/home/cs.aau.dk/ng78zb/pytorch_23.10-py3.sif
echo "sif ${SIF}"


srun singularity exec --nv --cleanenv --bind ${wd}:${wd} ${SIF} \
    python src/attacker.py ${CHECKPOINT_PATH} ${SOURCE_MODEL_NAME} ${TRAIN_SAMPLES} ${TEST_SAMPLES}