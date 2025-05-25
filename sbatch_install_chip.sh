#!/bin/bash

#SBATCH --partition=gpu        # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --gres=gpu:1

#SBATCH --mem=64000                # Real memory (RAM) required (MB)

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm_chip/generic/%j.out  # STDOUT output file

#SBATCH --error=slurm_chip/generic/%j.err   # STDERR output file (optional)

#SBATCH --gres-flags=enforce-binding

 
module load   Autoconf/2.72-GCCcore-13.3.0 
module load  CUDA/12.8.0  
moduel load Python/3.11.5-GCCcore-13.2.0

srun chip_install.sh