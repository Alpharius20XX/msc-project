#!/bin/bash -l

#SBATCH --job-name=tide-training
#SBATCH -p LIGHTGPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/xucabeeh/maxnewcopy/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out
#SBATCH --error=/home/xucabeeh/maxnewcopy/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out
#SBATCH --time=03-00:00:00


#48 mem
# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd /home/xucabeeh/maxnewcopy/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/home/xucabeeh/maxnewcopy/tmp/

# Run the training
echo "Running training script..."
# Define the path to your singularity image for easier reuse
SIF_PATH="/home/xucabeeh/maxnewcopy/hepattn/pixi.sif"
BIND_PATHS="/home/xucabeeh/maxnewcopy,/home/xzcappon/shared/for-max/ambi"

# 1. Manually fix the typo in the sophia library inside the container
# Using 'singularity exec' is a reliable way to run a command on the container's filesystem.
echo "Patching sophia library..."
singularity exec --nv --bind "$BIND_PATHS" "$SIF_PATH" \
    sed -i 's/SophiaG, sophiag/SophiaG/' /home/xucabeeh/maxnewcopy/hepattn/.pixi/envs/default/lib/python3.12/site-packages/sophia/__init__.py

# 2. Install the package using pixi run
echo "Installing sophia-optim..."
singularity run --nv --bind "$BIND_PATHS" "$SIF_PATH" \
    pixi run pip install sophia-optim

# 3. Run the training script
echo "Running training script..."
singularity run --nv --bind "$BIND_PATHS" "$SIF_PATH" \
    pixi run python src/hepattn/experiments/tide/main.py fit -c src/hepattn/experiments/tide/configs/base.yaml

echo "Done!"