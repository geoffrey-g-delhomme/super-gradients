#!/bin/bash

#SBATCH --job-name=yolo-nas-pose
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=200
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=./slurm/%j-srun-%n.out
#SBATCH --error=./slurm/%j-srun-%n.err

export PYTHONPATH=src && python -m super_gradients.train_from_recipe --config-name=aip_yolo_nas_pose_n_4096 $@