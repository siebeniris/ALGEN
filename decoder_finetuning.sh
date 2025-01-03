#!/bin/bash -e

#SBATCH --job-name=decoder_finetune
#SBATCH --output=decoderft_%j.out
#SBATCH --error=decoderft_%j.err
#SBATCH --mem=50GB
#SBATCH --time=6-00:00:00

set -x


wd=$(pwd)
echo "working directory ${wd}"

export HF_HOME="${wd}/.cache"
export HF_DATASETS_CACHE="${wd}/.cache/datasets"
export DATASET_CACHE_PATH="${wd}/.cache"
export DISABLE_APEX=1

MODEL_NAME=$1
OUTPUT_DIR=$2
MAX_LENGTH=$3
DATA_FOLDER=$4
LANG=$5
TRAIN_SAMPLES=$6
VAL_SAMPLES=$7
BATCH_SIZE=$8
LR=$9
WEIGHT_DECAY=${10}
NUM_EPOCHS=${11}
WAND_RUN_NAME=${12}





SIF=/home/cs.aau.dk/ng78zb/pytorch_23.10-py3.sif
echo "sif ${SIF}"


srun singularity exec --nv --cleanenv --bind ${wd}:${wd} ${SIF} \
    python src/exp.py \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR}\
    --max_length ${MAX_LENGTH} \
    --data_folder ${DATA_FOLDER} \
    --lang ${LANG}\
    --train_samples ${TRAIN_SAMPLES} \
    --val_samples ${VAL_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_epochs ${NUM_EPOCHS} \
    --wandb_run_name ${WAND_RUN_NAME}