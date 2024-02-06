#!/bin/bash

# sh train.sh dataset_coco_root_dirpath=/Users/geoffreygdelhomme/Documents/Datasets/coco-1024x750
# sh train.sh training_hyperparams.max_train_batches=8 training_hyperparams.max_valid_batches=8

# sh train.sh training_hyperparams.max_train_batches=2000
# sh train.sh training_hyperparams.max_train_batches=2000 num_gpus=8 dataset_params.train_dataloader_params.num_workers=8 dataset_params.val_dataloader_params.num_workers=8 dataset_params.train_dataloader_params.batch_size=48 dataset_params.val_dataloader_params.batch_size=48

# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=nominal training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_reg_loss_weight=0 training_hyperparams.criterion_params.line_cls_loss_weight=0
# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=proj_oks training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=proj_oks
# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=mse training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=mse
# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=mse training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=bce
# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=oks training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=oks
# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=proj_oks_clip training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=proj_oks training_hyperparams.criterion_params.line_regression_loss_clip_weight=0.1

# ng-201-41 (A100x4) (run)
# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=proj_oks training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=proj_oks

# ng-201-1 (V100x8) (run)
# sh train.sh training_hyperparams.max_train_batches=2000 num_gpus=8 dataset_params.train_dataloader_params.num_workers=8 dataset_params.val_dataloader_params.num_workers=8 dataset_params.train_dataloader_params.batch_size=48 dataset_params.val_dataloader_params.batch_size=48 experiment_suffix=proj_oks_clip training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=proj_oks training_hyperparams.criterion_params.line_regression_loss_clip_weight=0.1

# ng-202-32 (V100x4) (run)
# sh train.sh training_hyperparams.max_train_batches=2000 experiment_suffix=mse training_hyperparams.max_epochs=20 training_hyperparams.criterion_params.line_regression_loss_type=mse dataset_params.train_dataloader_params.batch_size=48 dataset_params.val_dataloader_params.batch_size=48


export PYTHONPATH=src && python -m super_gradients.train_from_recipe --config-name=aip_yolo_nas_intersect_n $@