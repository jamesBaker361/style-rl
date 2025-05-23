#!/bin/bash

#SBATCH --partition=gpu        # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --gres=gpu:2

#SBATCH --mem=64000                # Real memory (RAM) required (MB)

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm/generic/%j.out  # STDOUT output file

#SBATCH --error=slurm/generic/%j.err   # STDERR output file (optional)

day=$(date +'%m/%d/%Y %R')
echo "gpu"  ${day} $SLURM_JOBID "node_list" $SLURM_NODELIST $@  "\n" >> jobs.txt
module purge
module load intel/17.0.4
#module load cudnn/7.0.3
module load cuda/11.3
eval "$(conda shell.bash hook)"
conda activate deephands
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TORCH_USE_CUDA_DSA="1"
export CUDA_LAUNCH_BLOCKING="1"
export TRANSFORMERS_CACHE="/scratch/jlb638/trans_cache"
export HF_HOME="/scratch/jlb638/trans_cache"
export HF_HUB_CACHE="/scratch/jlb638/trans_cache"
export TORCH_CACHE="/scratch/jlb638/torch_hub_cache"
export TORCH_HOME="/scratch/jlb638/torch_home"
export WANDB_DIR="/scratch/jlb638/wandb"
export WANDB_CACHE_DIR="/scratch/jlb638/wandb_cache"
export HPS_ROOT="/scratch/jlb638/hps-cache"
export IMAGE_REWARD_PATH="/scratch/jlb638/reward-blob"
export IMAGE_REWARD_CONFIG="/scratch/jlb638/ImageReward/med_config.json"
export BRAIN_DATA_DIR='/scratch/jlb638/brain-diffuser/data'
export CUDA_LAUNCH_BLOCKING="1"
export SCIKIT_LEARN_DATA="/scratch/jlb638/scikit-learn-data"
export BRAIN_DATA_DIR="/scratch/jlb638/brain/data"
srun accelerate launch  $@
conda deactivate