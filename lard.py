# %%
NOTEBOOK = False
DATA_DIRPATH = '/aip/datasets/a3/preprocessed/07-02-23/yolo-formated/origin/detect'
DATA_DIRPATH = '/aip/datasets/deel/processed/origin/detect'
CHECKPOINT_DIRPATH = './checkpoints'
EXPERIMENT = "yolo_nas_s_lard"
MODEL = "yolo_nas_s"
INPUT = [2048, 2048]
BATCH_SIZE = 64

# %%
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('./src')

# %%
import copy
import super_gradients

from super_gradients.training import models
from super_gradients.training import Trainer

from torchinfo import summary

from pathlib import Path

from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.utils.distributed_training_utils import setup_device, MultiGPUMode

# %%
setup_device(multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=8)

# %%
# model = models.get("yolo_nas_s", pretrained_weights="coco") # s, m or l
model = models.get(MODEL, num_classes=1) # s, m or l

print(summary(model=model, 
        input_size=(1, 3, *INPUT),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
))


trainer = Trainer(experiment_name=EXPERIMENT, ckpt_root_dir=CHECKPOINT_DIRPATH)

# %%
dataset_params = {
    'data_dir': DATA_DIRPATH,
    'train_images_dir':'images/train',
    'train_labels_dir':'labels/train',
    'train_cache_dir': DATA_DIRPATH+'/cache/train',
    'val_images_dir':'images/valid',
    'val_labels_dir':'labels/valid',
    'val_cache_dir': DATA_DIRPATH+'/cache/valid',
    'test_images_dir':'images/valid',
    'test_labels_dir':'labels/valid',
    'test_cache_dir': DATA_DIRPATH+'/cache/valid',
    'classes': ['Runway'],
    'input_dim': INPUT,
    'cache_annotations': True
}

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes'],
        'input_dim': dataset_params['input_dim'],
        'cache_dir': dataset_params['train_cache_dir'],
        'cache_annotations': dataset_params['cache_annotations']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers':8,
        'sampler': {"DistributedSampler": {}}
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes'],
        'input_dim': dataset_params['input_dim'],
        'cache_dir': dataset_params['val_cache_dir'],
        'cache_annotations': dataset_params['cache_annotations']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':8,
        'sampler': {"DistributedSampler": {}}
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes'],
        'input_dim': dataset_params['input_dim'],
        'cache_dir': dataset_params['val_cache_dir'],
        'cache_annotations': dataset_params['cache_annotations']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':8,
        'sampler': {"DistributedSampler": {}}
    }
)

# %%
print(train_data.dataset.transforms)

# %%
# Plot a batch
if NOTEBOOK:
  train_data.dataset.plot()

# %%
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

train_params = {
    'silent_mode': False, # True: do not slow down in notebook
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 4,
    "initial_lr": 1e-3,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 1e-2,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 100,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=1,
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=1,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.3,
                nms_top_k=200,
                max_predictions=20,
                nms_threshold=0.5
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

trainer.train(model=model, 
              training_params=train_params, 
              train_loader=train_data, 
              valid_loader=val_data)

# %%
best_model = models.get(MODEL, num_classes=1, checkpoint_path=f"{CHECKPOINT_DIRPATH}/{EXPERIMENT}/average_model.pth").cuda()

# %%
res = best_model.predict("/efs/players/to122838/resources/landing/videos/paphos.mp4")

# %%
res.save(f"./tmp/{EXPERIMENT}-paphos.mp4")


# %%
best_model = models.get(MODEL, num_classes=1, checkpoint_path=Path(f"{CHECKPOINT_DIRPATH}/{EXPERIMENT}/ckpt_best.pth").resolve().as_posix()).cuda()

from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

q_util = SelectiveQuantizer(
    default_quant_modules_calibrator_weights="max",
    default_quant_modules_calibrator_inputs="histogram",
    default_per_channel_quant_weights=True,
    default_learn_amax=False,
    verbose=True,
)
q_util.quantize_module(best_model)

from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator

calibrator = QuantizationCalibrator(verbose=True)
calibrator.calibrate_model(
    model,
    method="percentile",
    calib_data_loader=val_dataloader,
    num_calib_batches=16,
    percentile=99.99,
)