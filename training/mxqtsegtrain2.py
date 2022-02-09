#!/usr/bin/env python3


"""
Demo of a 2D semantic segmentation on TUM ML data v2, distinguishing QtEnc from MxEnc particles

# TODO: Update description when it's clear what this script does.
"""

import argparse
import datetime
from math import inf
from pathlib import Path
from typing import Tuple
import os
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import pandas as pd

# Don't move this stuff, it needs to be run this early to work
import elektronn3
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.utils import data

from omegaconf import OmegaConf, DictConfig
import hydra

# from monai.losses import GeneralizedDiceLoss


elektronn3.select_mpl_backend('Agg')

from elektronn3.training import Trainer, Backup
from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.models.unet import UNet
from elektronn3.modules.loss import CombinedLoss, DiceLoss

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import albumentations

from training.tifdirdata import ZTifDirData2d


# @hydra.main(config_path='conf', config_name='config')
# def my_app(cfg: DictConfig) -> None:
#     global conf
#     conf = cfg
#     print(OmegaConf.to_yaml(conf))


# if __name__ == "__main__":
#     my_app()


# print(conf.db)

# conf = OmegaConf.create()

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
parser.add_argument(
    '-m', '--max-steps', type=int, default=80_000,
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
args = parser.parse_args()

# Set up all RNG seeds, set level of determinism
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
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


# SHEET_NAME = 'Copy of Image_origin_information_GGW'
SHEET_NAME = 'outdated_Image_origin_informati'

DATA_SELECTION = 'all'
# DATA_SELECTION = '1x'
# DATA_SELECTION = '2x'



# TODO: WARNING: This inverts some of the labels depending on image origin. Don't forget to turn this off when it's not necessary (on other images)
ENABLE_TERRIBLE_INVERSION_HACK = True


VEC_DT = False
DT = False
MULTILABEL = False

INPUTMASK = False

INVERT_LABELS = False
# INVERT_LABELS = True

# BINARY_SEG = False
BINARY_SEG = True


USE_MTCE = False
# USE_MTCE = True

# USE_GDL_CE = False
USE_GDL_CE = True

# USE_GRAY_AUG = False
USE_GRAY_AUG = True

data_root = Path('~/tum/Single-table_database').expanduser()

if MULTILABEL:
    label_names = [
        'background',
        'membranes',
        'encapsulins',
        'nuclear_membrane',
        'nuclear_region',
        'cytoplasmic_region',
    ]
else:
    # label_names = ['=ZEROS=', 'membranes']
    label_names = ['=ZEROS=', 'encapsulins']

if DT or MULTILABEL:
    target_dtype = np.float32
else:
    target_dtype = np.int64

# if DT:
#     out_channels = 1 if not VEC_DT else 3
# else:
#     out_channels = len(label_names)

out_channels = 3
if BINARY_SEG:
    out_channels = 2
if USE_MTCE:
    out_channels = 4

model = UNet(
    out_channels=out_channels,
    n_blocks=2,
    start_filts=64,
    activation='relu',
    normalization='batch',
    dim=2
).to(device)

# USER PATHS
save_root = Path('~/tum/mxqtsegtrain2_trainings_hek4_bin').expanduser()

max_steps = args.max_steps
lr = 1e-3
lr_stepsize = 1000
lr_dec = 0.95
batch_size = 8


if args.resume is not None:  # Load pretrained network params
    model.load_state_dict(torch.load(os.path.expanduser(args.resume)))

dataset_mean = (128.0,)
dataset_std = (128.0,)

dt_scale = 30

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.RandomCrop((768, 768)),
    transforms.DropIfTooMuchBG(threshold=1 - (1 / 500**2)),
    transforms.Normalize(mean=dataset_mean, std=dataset_std, inplace=False),
    transforms.RandomFlip(ndim_spatial=2),
]

train_transform = common_transforms + [
    transforms.RandomCrop((512, 512)),
    transforms.AlbuSeg2d(albumentations.ShiftScaleRotate(
        p=0.99, rotate_limit=180, shift_limit=0.0625, scale_limit=0.1, interpolation=2
    )),  # interpolation=2 means cubic interpolation (-> cv2.CUBIC constant).
    # transforms.ElasticTransform(prob=0.5, sigma=2, alpha=5),
]
if USE_GRAY_AUG:
    train_transform.extend([
        transforms.AdditiveGaussianNoise(prob=0.5, sigma=0.1),
        transforms.RandomGammaCorrection(prob=0.5, gamma_std=0.2),
        transforms.RandomBrightnessContrast(prob=0.5, brightness_std=0.125, contrast_std=0.125),
    ])

valid_transform = common_transforms + []

if DT:
    train_transform.append(transforms.DistanceTransformTarget(scale=dt_scale, vector=VEC_DT))
    valid_transform.append(transforms.DistanceTransformTarget(scale=dt_scale, vector=VEC_DT))

train_transform = transforms.Compose(train_transform)
valid_transform = transforms.Compose(valid_transform)

# Specify data set


# valid_image_numbers = [
#     40, 60, 114,  # MxEnc
#     43, 106, 109,  # QtEnc
# ]

# valid_image_numbers = [
#     66, 83,  # MxEnc: 1x, 2x
#     22, 70,  # QtEnc: 1x, 2x
# ]

valid_image_numbers_1x = [
    66,  # MxEnc
    22,  # QtEnc
]

valid_image_numbers_2x = [
    83,  # MxEnc
    70,  # QtEnc
]

if DATA_SELECTION == 'all':
    valid_image_numbers = valid_image_numbers_1x + valid_image_numbers_2x
elif DATA_SELECTION == '1x':
    valid_image_numbers = valid_image_numbers_1x
elif DATA_SELECTION == '2x':
    valid_image_numbers = valid_image_numbers_2x

# print('Training on images ', train_image_numbers)
# print('Validating on images ', valid_image_numbers)



def meta_filter(meta):
    meta_orig = meta
    meta = meta.copy()
    if DATA_SELECTION == 'all':
        meta = meta.loc[(meta['1xMmMT3'] | meta['2xMmMT3'])]
    elif DATA_SELECTION != 'all':
        meta = meta.loc[meta[f'{DATA_SELECTION}MmMT3']]
    meta = meta.loc[meta['Host organism'] == 'HEK cell culture']
    meta = meta.loc[meta['Modality'] == 'TEM']

    meta = meta[['num', 'MxEnc', 'QtEnc', '1xMmMT3', '2xMmMT3']]


    return meta

train_dataset = ZTifDirData2d(
    descr_sheet=(data_root / 'Image_annotation_for_ML_single_table.xlsx', SHEET_NAME),
    meta_filter=meta_filter,
    valid_nums=valid_image_numbers,
    train=True,
    label_names=label_names,
    transform=train_transform,
    target_dtype=target_dtype,
    invert_labels=INVERT_LABELS,
    enable_inputmask=INPUTMASK,
    enable_binary_seg=BINARY_SEG,
    enable_partial_inversion_hack=ENABLE_TERRIBLE_INVERSION_HACK,
    epoch_multiplier=10,
)

valid_dataset = ZTifDirData2d(
    descr_sheet=(data_root / 'Image_annotation_for_ML_single_table.xlsx', SHEET_NAME),
    meta_filter=meta_filter,
    valid_nums=valid_image_numbers,
    train=False,
    label_names=label_names,
    transform=valid_transform,
    target_dtype=target_dtype,
    invert_labels=INVERT_LABELS,
    enable_inputmask=INPUTMASK,
    enable_binary_seg=BINARY_SEG,
    enable_partial_inversion_hack=ENABLE_TERRIBLE_INVERSION_HACK,
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


class MTCELoss(nn.Module):
    def __init__(self, binseg_weight=1., fgtype_weight=1., fgtype_per_pix=False, *args, **kwargs) -> None:
        super().__init__()
        self.binseg_ce = nn.CrossEntropyLoss(*args, **kwargs)  # fg vs bg
        self.fgtype_ce = nn.CrossEntropyLoss()  # fg type vs other fg type
        self.binseg_weight = binseg_weight
        self.fgtype_weight = fgtype_weight
        self.fgtype_per_pix = fgtype_per_pix


    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Prepare targets and weight mask
        with torch.no_grad():
            binseg_target = (target > 0).to(torch.int64)
            fg_mask = binseg_target.to(output.dtype)
            fgtype_target = target - 1
        binseg_loss = self.binseg_ce(output[:, :2], binseg_target)
        if self.fgtype_per_pix:  # Per pixel loss
            # fgtype_loss = F.cross_entropy(output[:, 2:], fgtype_target, weight=fg_mask)
            fgtype_loss_pix = F.cross_entropy(output[:, 2:], fgtype_target, reduction='none')
            fgtype_loss = torch.mean(fgtype_loss_pix * fg_mask)
        else:  # Average -> one scalar per image
            out_avg = F.adaptive_avg_pool2d(output[:, 2:], 1)
            with torch.no_grad():
                fgtype_target = target
                uniq = torch.unique(fgtype_target)
                if len(uniq) == 0:  # No fg -> no fg type targets, only bg
                    fgtype_loss = 0.
            if len(uniq) == 1:  # containing fg type targets of one type
                with torch.no_grad():
                    fgtype_target_single = torch.max(uniq).to(torch.int64)  # foreground class label is always maximum
                fgtype_loss = F.cross_entropy(out_avg, fgtype_target_single)
            elif len(uniq) > 1:
                print('Oh no')
                import IPython ; IPython.embed(); raise SystemExit
                raise ValueError(uniq)
        return self.binseg_weight * binseg_loss + self.fgtype_weight * fgtype_loss


if DT:
    criterion = MSELoss()
else:
    if MULTILABEL:
        _cw = [1.0 for _ in label_names]
        _cw[0] = 0.2  # reduced loss weight for background labels
        class_weights = torch.tensor(_cw).to(device)
        criterion = nn.BCEWithLogitsLoss()#(pos_weight=class_weights)
    else:
        if BINARY_SEG:
            class_weights = torch.tensor([0.2, 1.0]).to(device)
        else:
            class_weights = torch.tensor([0.2, 1.0, 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)


if USE_MTCE:
    criterion = MTCELoss(weight=torch.tensor([0.2, 1.0]).to(device))

if USE_GDL_CE:
    ce = CrossEntropyLoss(weight=torch.tensor([0.2, 1.0])).to(device)
    # gdl = GeneralizedDiceLoss(softmax=True, to_onehot_y=True, w_type='simple').to(device)
    gdl = DiceLoss(apply_softmax=True, weight=torch.tensor([0.2, 1.0]).to(device))
    criterion = CombinedLoss([ce, gdl], device=device)

if USE_MTCE:
    inference_kwargs = {
        'apply_softmax': False,
        'transform': valid_transform,
    }
elif DT:
    inference_kwargs = {
        'apply_softmax': False,
        'transform': valid_transform,
    }
elif MULTILABEL:
    inference_kwargs = {
        'apply_softmax': False,
        'apply_sigmoid': True,
        'transform': valid_transform,
    }
else:
    inference_kwargs = {
        'apply_softmax': True,
        'transform': valid_transform,
    }

_MULTILABEL = 'M_' if MULTILABEL else ''
_DT = 'D_' if DT else ''
_VEC_DT = 'V_' if VEC_DT else ''
_MQ = 'MQ_' if not BINARY_SEG else 'B_'
_MTCE = 'MTCE_' if USE_MTCE else ''
_GRAY_AUG = 'GA_' if USE_GRAY_AUG else ''
_GDL_CE = 'GDL_CE_' if USE_GDL_CE else ''

exp_name = args.exp_name
if exp_name is None:
    exp_name = ''
timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
exp_name = f'{exp_name}__{model.__class__.__name__ + "__" + timestamp}'
exp_name = f'{_GDL_CE}{_MTCE}{_MQ}{_VEC_DT}{_MULTILABEL}{_DT}{_GRAY_AUG}{exp_name}'



# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batch_size=batch_size,
    num_workers=1,  # TODO
    save_root=save_root,
    exp_name=exp_name,
    inference_kwargs=inference_kwargs,
    save_jit=None,
    schedulers={"lr": lr_sched},
    valid_metrics=valid_metrics,
    out_channels=out_channels,
    mixed_precision=True,
    extra_save_steps=list(range(0, max_steps, 10_000)),
)




# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
