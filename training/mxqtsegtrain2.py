#!/usr/bin/env python3

# TODO: Only use ignoring on TmEnc


"""
Demo of a 2D semantic segmentation on TUM ML data v2, distinguishing QtEnc from MxEnc particles

# TODO: Update description when it's clear what this script does.
"""

import argparse
import datetime
import logging
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Dict
import os
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import pandas as pd
import yaml

# Don't move this stuff, it needs to be run this early to work
import elektronn3
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.utils import data

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.config_store import ConfigStore


elektronn3.select_mpl_backend('Agg')
logger = logging.getLogger('elektronn3log')

from elektronn3.training import Trainer, Backup, SWA
from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.models.unet import UNet
from elektronn3.modules.loss import CombinedLoss, DiceLoss

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import albumentations

from tqdm import tqdm

from training.tifdirdata import V6TifDirData2d
from utils import V5NAMES_TO_OLDNAMES


# @dataclass
# class TrainingConf:
#     data_root: str = '/wholebrain/scratch/mdraw/tum/Single-table_database'
#     dsel: str = '1x'
#     horg: Literal['HEK cell culture', 'Drosophila'] = 'HEK cell culture'

#     disable_cuda: bool = False
#     n: Optional[str] = None
#     max_steps: int = 80_000
#     seed: int = 0
#     deterministic: bool = False
#     save_root: str = '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_hek4_bin'



# cs = ConfigStore.instance()
# cs.store(name='training_conf', node=TrainingConf)


# @hydra.main(config_path='conf', config_name='training_conf')
# def main(cfg: DictConfig) -> None:
#     global conf
#     conf = cfg
#     print(OmegaConf.to_yaml(conf))


# if __name__ == "__main__":
#     main()


# print(conf.db)

# conf = OmegaConf.create()

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
parser.add_argument(
    '-m', '--max-steps', type=int, default=160_001,
    help='Maximum number of training steps to perform.'
)
parser.add_argument(
    '-r', '--resume', metavar='PATH',
    help='Path to pretrained model state dict from which to resume training.'
)
parser.add_argument('--seed', type=int, default=0, help='Base seed for all RNGs.')
parser.add_argument(
    '--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).'
)
parser.add_argument('-c', '--constraintype', default=None, help='Constrain training and validation to only one encapsulin type (via v5name, e.g. `-c 1M-Qt`).')
args = parser.parse_args()

conf = args # TODO

# Set up all RNG seeds, set level of determinism
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

# 0: background
# 1: membranes
# 2: encapsulins
# 3: nuclear_membrane
# 4: nuclear_region
# 5: cytoplasmic_region


SHEET_NAME = 'all_metadata'


if args.constraintype is None:
    DATA_SELECTION_V5NAMES = [
        '1M-Mx',
        '1M-Qt',
        '2M-Mx',
        '2M-Qt',
        '3M-Qt',
        '1M-Tm',
        # 'DRO-1M-Mx',
        # 'DRO-1M-Qt',
    ]
else:
    DATA_SELECTION_V5NAMES = [args.constraintype]

DATA_SELECTION = [V5NAMES_TO_OLDNAMES[n] for n in DATA_SELECTION_V5NAMES]



IGNORE_INDEX = -1
# IGNORE_FAR_BACKGROUND_DISTANCE = 16
IGNORE_FAR_BACKGROUND_DISTANCE = 0

BG_WEIGHT = 0.2


# TODO: WARNING: This inverts some of the labels depending on image origin. Don't forget to turn this off when it's not necessary (on other images)
ENABLE_PARTIAL_INVERSION_HACK = False


INPUTMASK = False

INVERT_LABELS = False
# INVERT_LABELS = True

# USE_GRAY_AUG = False
USE_GRAY_AUG = True

DILATE_TARGETS_BY = 0


data_root = Path('/wholebrain/scratch/mdraw/tum/Single-table_database').expanduser()
# data_root = Path(conf.data_root).expanduser()

label_names = ['=ZEROS=', 'encapsulins']
target_dtype = np.int64

out_channels = 2
model = UNet(
    out_channels=out_channels,
    n_blocks=5,
    start_filts=64,
    activation='relu',
    normalization='batch',
    dim=2
).to(device)

# USER PATHS
sr_suffix = ''
save_root = Path(f'/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v7{sr_suffix}').expanduser()


max_steps = conf.max_steps
lr = 1e-3
lr_stepsize = 1000
lr_dec = 0.95
batch_size = 8


if conf.resume is not None:  # Load pretrained network params
    model.load_state_dict(torch.load(os.path.expanduser(conf.resume)))

dataset_mean = (128.0,)
dataset_std = (128.0,)


# TODO: https://github.com/Project-MONAI/MONAI/blob/384c7181e3730988fe5318e9592f4d65c12af843/monai/transforms/croppad/array.py#L831

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.RandomCrop((512, 512)),
    transforms.Normalize(mean=dataset_mean, std=dataset_std, inplace=False),
    transforms.RandomFlip(ndim_spatial=2),
]

train_transform = common_transforms + [
    # transforms.RandomCrop((512, 512)),
    transforms.AlbuSeg2d(albumentations.ShiftScaleRotate(
        p=0.9, rotate_limit=180, shift_limit=0.0625, scale_limit=0.1, interpolation=2
    )),  # interpolation=2 means cubic interpolation (-> cv2.CUBIC constant).
    # transforms.ElasticTransform(prob=0.5, sigma=2, alpha=5),
    transforms.RandomCrop((384, 384)),
]
if USE_GRAY_AUG:
    train_transform.extend([
        transforms.AdditiveGaussianNoise(prob=0.4, sigma=0.1),
        transforms.RandomGammaCorrection(prob=0.4, gamma_std=0.1),
        transforms.RandomBrightnessContrast(prob=0.4, brightness_std=0.1, contrast_std=0.1),
    ])

valid_transform = common_transforms + []


train_transform = transforms.Compose(train_transform)
valid_transform = transforms.Compose(valid_transform)


def meta_filter(meta):
    meta = meta.copy()
    meta = meta.loc[meta['scond'].isin(DATA_SELECTION)]
    return meta


train_dataset = V6TifDirData2d(
    descr_sheet=(data_root / 'Image_annotation_for_ML_single_table.xlsx', SHEET_NAME),
    meta_filter=meta_filter,
    # valid_nums=valid_image_numbers,  # read from table
    train=True,
    data_subdirname='isplitdata_v7',
    label_names=label_names,
    transform=train_transform,
    target_dtype=target_dtype,
    invert_labels=INVERT_LABELS,
    enable_inputmask=INPUTMASK,
    enable_binary_seg=BINARY_SEG,
    enable_partial_inversion_hack=ENABLE_PARTIAL_INVERSION_HACK,
    ignore_far_background_distance=IGNORE_FAR_BACKGROUND_DISTANCE,
    dilate_targets_by=DILATE_TARGETS_BY,
    epoch_multiplier=100,
)

valid_dataset = V6TifDirData2d(
    descr_sheet=(data_root / 'Image_annotation_for_ML_single_table.xlsx', SHEET_NAME),
    meta_filter=meta_filter,
    # valid_nums=valid_image_numbers,  # read from table
    train=False,
    data_subdirname='isplitdata_v7',
    label_names=label_names,
    transform=valid_transform,
    target_dtype=target_dtype,
    invert_labels=INVERT_LABELS,
    enable_inputmask=INPUTMASK,
    enable_binary_seg=BINARY_SEG,
    enable_partial_inversion_hack=ENABLE_PARTIAL_INVERSION_HACK,
    ignore_far_background_distance=IGNORE_FAR_BACKGROUND_DISTANCE,
    dilate_targets_by=DILATE_TARGETS_BY,
    epoch_multiplier=20,
)


# Set up optimization
optimizer = optim.Adam(
    model.parameters(),
    weight_decay=5e-5,
    lr=lr,
    amsgrad=True
)
lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)


# Validation metrics

valid_metrics = {}
if not DT and not MULTILABEL:
    for evaluator in [metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.DSC, metrics.IoU]:
        valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()  # Mean metrics
        for c in range(out_channels):
            valid_metrics[f'val_{evaluator.name}_c{c}'] = evaluator(c)


class_weights = torch.tensor([BG_WEIGHT, 1.0]).to(device)
ce = CrossEntropyLoss(weight=torch.tensor([BG_WEIGHT, 1.0])).to(device)
gdl = DiceLoss(apply_softmax=True, weight=torch.tensor([BG_WEIGHT, 1.0]).to(device))  # TODO: Support ignore_index
criterion = CombinedLoss([ce, gdl], device=device)

inference_kwargs = {
    'apply_softmax': True,
    'transform': valid_transform,
}

_GRAY_AUG = 'GA_' if USE_GRAY_AUG else ''

if len(DATA_SELECTION_V5NAMES) == 1:
    _CONSTR_TYPE = f'{DATA_SELECTION_V5NAMES[0]}_'
else:
    _CONSTR_TYPE = ''

exp_name = conf.exp_name
if exp_name is None:
    exp_name = ''
timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
exp_name = f'{exp_name}__{model.__class__.__name__ + "__" + timestamp}'
exp_name = f'{_CONSTR_TYPE}{_GRAY_AUG}{exp_name}'



# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batch_size=batch_size,
    num_workers=8,  # TODO
    save_root=save_root,
    exp_name=exp_name,
    inference_kwargs=inference_kwargs,
    save_jit='script',
    schedulers={"lr": lr_sched},
    valid_metrics=valid_metrics,
    out_channels=out_channels,
    mixed_precision=True,
    extra_save_steps=list(range(40_000, max_steps + 1, 40_000)),
)



# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
