#!/bin/bash

#SBATCH --job-name=yolo-nas-pose
#SBATCH --partition=quadro
#SBATCH --gpus-per-node=6
#SBATCH --cpus-per-gpu=30
#SBTACH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --output=./slurm/%j-srun-%n.out
#SBATCH --error=./slurm/%j-srun-%n.err

export PYTHONPATH=src && python -m super_gradients.train_from_recipe --config-name=aip_yolo_nas_pose_n num_gpus=6 $@ 