#!/bin/bash -l

#SBATCH --job-name=tide-testing
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

# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(nproc)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd /home/xucabeeh/maxcopy/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/home/xucabeeh/maxcopy/tmp/

# Run the testing
echo "Running test script..."

# Python command that will be run.
# The `--output_dir` argument is incorrect for this script.
# The correct argument for setting the output directory is `--trainer.default_root_dir`.
# I've also added `--print_config` to show the full configuration before running. #/test_outputs
PYTORCH_CMD="python src/hepattn/experiments/tide/main.py test \
    --config logs/TIDE_32trk_F32_20250709-T170659/config.yaml \
    --ckpt_path logs/TIDE_32trk_F32_20250709-T170659/ckpts/epoch=006-train_loss=5.90920.ckpt \
    --trainer.default_root_dir src/hepattn/experiments/tide \
    --print_config \
    --data.num_workers 0 \
    --trainer.callbacks[2].init_args.write_outputs=true \
    --trainer.callbacks[2].init_args.write_targets=true"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image.
# Binds both the project directory and the data directory.
APPTAINER_CMD="singularity run --nv --bind /home/xucabeeh/maxcopy,/home/xzcappon/shared/for-max/ambi /home/xucabeeh/maxcopy/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"