#!/bin/bash

# sh train.sh dataset_coco_root_dirpath=/Users/geoffreygdelhomme/Documents/Datasets/coco-1024x750
# sh train.sh training_hyperparams.max_train_batches=8 training_hyperparams.max_valid_batches=8
# sh train.sh training_hyperparams.max_train_batches=3600 training_hyperparams.max_valid_batches=1200

export PYTHONPATH=src && python -m super_gradients.train_from_recipe --config-name=aip_yolo_nas_intersect_n $@