#!/bin/bash

#SBATCH --job-name=sg-qat
#SBATCH --partition=volta
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=90G
#SBTACH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=./slurm/%j-srun-%n.out
#SBATCH --error=./slurm/%j-srun-%n.err

export PYTHONPATH=src && python -m super_gradients.train_from_recipe --config-name=aip_yolo_nas_pose_n_qat $@