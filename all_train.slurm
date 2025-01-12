#!/bin/bash

#SBATCH --job-name=cocoa_training
#SBATCH --output=logs/cocoa_%A_%a.out
#SBATCH --error=logs/cocoa_%A_%a.err
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --array=[1-21]  # Run up to 4 jobs simultaneously (adjust based on your models count)

# Create logs directory
mkdir -p logs

# Array of models (must match your models.json)
declare -a models=(
    "vit_large"
    "vit_base"
    "convnext_xxlarge"
    "convnext_xlarge"
    "swin_large"
    "efficientnet_b7"
    "efficientnet_b5"
    "efficientnet_b3"
    "efficientnet_b0"
    "mobilenetv3_large"
    "mobilenetv3_small"
    "eva_giant"
    "maxvit_xlarge"
    "beit_large"
    "resnet_152"
    "resnet_101"
    "resnet_50"
    "resnet_34"
    "vgg19"
    "vgg16"
    "vgg13"
)

# Get the model name for this array task
MODEL=${models[$SLURM_ARRAY_TASK_ID]}

# Convert underscore to hyphen for the script
MODEL_ARG=$(echo $MODEL | tr '_' '_')

# Load necessary modules and activate conda environment
module purge

# Activate your conda environment
hostname
nvidia-smi
pwd

conda activate test

echo "running python from...."
which python

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL_ARG"
echo "Precision: $PRECISION"
echo "Running on node: $HOSTNAME"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Run the training script
./train.sh -m $MODEL_ARG $PRECISION \
    2>&1 | tee "logs/training_${MODEL_ARG}.log"

# Deactivate conda environment
conda deactivate