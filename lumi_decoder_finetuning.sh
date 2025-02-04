#!/bin/bash -e
#SBATCH --job-name=fewshot-inverter
#SBATCH --account=project_465001270
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=50G
#SBATCH --time=3-00:00:00
#SBATCH --output=decoder_%j.out
#SBATCH --error=decoder_%j.err

set -x


wd=$(pwd)
echo "working directory ${wd}"

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
TRAINING_MODE=${13}


export OPENAI_API_KEY="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"
export HF_HOME="/scratch/project_465001270/.cache"
export HF_DATASETS_CACHE="/scratch/project_465001270/.cache/datasets"
export DATASET_CACHE_PATH="/scratch/project_465001270/.cache"
export WANDB_CACHE_DIR="/scratch/project_465001270/.cache/wandb/artifcats/"
export HUGGINGFACE_HUB_TOKEN="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"

export NCCL_P2P_LEVEL=PHB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3


# pytorch multiprocessing. semaphore.
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

SIF=/scratch/project_465001270/multivec2text.sif


echo $SIF
chmod +x $HF_HOME
chmod +x $HF_DATASETS_CACHE


srun singularity exec \
    -B /scratch/project_465001270:/scratch/project_465001270 \
    -B ${wd}:${wd} \
    -B ${HF_HOME}:${HF_HOME} \
    -B ${HF_DATASETS_CACHE}:${HF_DATASETS_CACHE} \
    ${SIF} python src/exp.py \
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
    --wandb_run_name ${WAND_RUN_NAME} \
    --training_mode ${TRAINING_MODE}
