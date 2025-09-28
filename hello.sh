#!/bin/bash

#SBATCH --job-name=tide-training
#SBATCH -p LIGHTGPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/maxcopy/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out



echo "Job is running on host: $(hostname)"
echo "Sleeping for 60 seconds..."
sleep 60
echo "Job finished!"