#!/bin/bash

#SBATCH --job-name=yolo-nas-pose
#SBATCH --partition=v100
#SBATCH --gpus-per-node=8
#SBTACH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=./slurm/%j-srun-%n.out
#SBATCH --error=./slurm/%j-srun-%n.err

export PYTHONPATH=src && python -m super_gradients.train_from_recipe --config-name=aip_yolo_nas_pose_n $@