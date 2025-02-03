#!/bin/bash -e
#SBATCH --job-name=classifier
#SBATCH --account=project_465001270
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=50G
#SBATCH --time=12:00:00
#SBATCH --output=classifier_%j.out
#SBATCH --error=classifier_%j.err

set -x


wd=$(pwd)
echo "working directory ${wd}"

dataset_name=$1
task_name=$2
num_labels=$3
model_name=$4
batch_size=$5
defense_method=$6



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
    ${SIF} python -m src.classifiers.trainer ${dataset_name} ${task_name} ${num_labels} ${model_name} ${batch_size} ${defense_method}
