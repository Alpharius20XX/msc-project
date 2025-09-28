#!/bin/bash -l


#SBATCH --job-name=tide-training

#SBATCH -p LIGHTGPU

#SBATCH --nodes=1

#SBATCH --export=ALL

#SBATCH --gres=gpu:a100:1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --mem=96G

#SBATCH --output=/home/xucabeeh/maxcopy/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out

#SBATCH --error=/home/xucabeeh/maxcopy/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out

#SBATCH --time=02-00:00:00



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

cd /home/xucabeeh/maxcopy/hepattn/

echo "Moved dir, now in: ${PWD}"


# Set tmpdir

export TMPDIR=/home/xucabeeh/maxcopy/tmp/


# Run the training

echo "Running test script..."


# Python command that will be run

#PYTORCH_CMD="python src/hepattn/experiments/tide/main.py fit -c src/hepattn/experiments/tide/configs/base.yaml -c src/hepattn/experiments/tide/configs/tagging/base.yaml -c src/hepattn/experiments/tide/configs/tagging/sudo.yaml "

# PYTORCH_CMD="python src/hepattn/experiments/tide/main.py fit --config /share/rcifdata/maxhart/hepattn/logs/TIDE_1M_100_32trk_F32_20250517-T092110/config.yaml --ckpt_path /share/rcifdata/maxhart/hepattn/logs/TIDE_1M_100_32trk_F32_20250517-T092110/ckpts/epoch=001-train_loss=73.99285.ckpt"

# PYTORCH_CMD="python src/hepattn/experiments/tide/main.py fit -c src/hepattn/experiments/tide/configs/base.yaml"


#PYTORCH_CMD="python src/hepattn/experiments/tide/main.py test --config logs/TIDE_32trk_F32_20250709-T170659/config.yaml --ckpt_path logs/TIDE_32trk_F32_20250709-T170659/ckpts/epoch=006-train_loss=5.90920.ckpt" #--trainer.default_root_dir logs/test_output_dir

#/home/xucabeeh/maxcopy/hepattn/logs/TIDE_32trk_F32_20250830-T131554/ckpts/epoch=003-train_loss=7.16745.ckpt

folder="/seperate_field_rope"
file="/epoch=006-train_loss=4.80430.ckpt"

# The base directory pathmaxcopy/hepattn/logs/seperate_field_rope/ckpts/epoch=006-train_loss=4.80430.ckpt

base_dir="logs"
#maxcopy/hepattn/
# The updated command using the variables
PYTORCH_CMD="python src/hepattn/experiments/tide/main.py test --config ${base_dir}${folder}/config.yaml --ckpt_path ${base_dir}${folder}/ckpts${file} --data.num_workers 0"

#maxcopy/hepattn/logs/TIDE_32trk_F32_20250731-T161923/ckpts/epoch=006-train_loss=5.66957.ckpt

#/home/xucabeeh/maxcopy/hepattn/logs/rope sep just ang/ckpts/

# --data.test_dir /share/rcifdata/maxhart/data/

 

# Pixi commnand that runs the python command inside the pixi env

PIXI_CMD="pixi run $PYTORCH_CMD"

 

# Apptainer command that runs the pixi command inside the pixi apptainer image

#APPTAINER_CMD="singularity run --nv --bind /home/xucabeeh/maxcopy /home/xucabeeh/maxcopy/hepattn/pixi.sif $PIXI_CMD"


# Bind both the project directory and the data directory

APPTAINER_CMD="singularity run --nv --bind /home/xucabeeh/maxcopy,/home/xzcappon/shared/for-max/ambi /home/xucabeeh/maxcopy/hepattn/pixi.sif $PIXI_CMD"

#rename

#APPTAINER_CMD="singularity run --nv --bind /home/xucabeeh/maxcopy, /home/xzcappon/shared/for-max/ambi:/data /home/xucabeeh/maxcopy/hepattn/pixi.sif $PIXI_CMD"

# Run the final command

echo "Running command: $APPTAINER_CMD"

$APPTAINER_CMD

PYTORCH_CMD="python3 /home/xucabeeh/maxcopy/hepattn/src/hepattn/experiments/tide/eval_callable.py --output_dir ${base_dir}${folder}"

PIXI_CMD="pixi run $PYTORCH_CMD"

APPTAINER_CMD="singularity run --nv --bind /home/xucabeeh/maxcopy,/home/xzcappon/shared/for-max/ambi /home/xucabeeh/maxcopy/hepattn/pixi.sif $PIXI_CMD"

$APPTAINER_CMD

echo "Done!" change this to prevent issue 