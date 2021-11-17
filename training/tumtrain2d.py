#!/usr/bin/env python3


"""
Demo of a 2D semantic segmentation on TUM ML data.

"""

import argparse
import datetime
from math import inf
import os
import random
from elektronn3.data.transforms.transforms import RandomCrop

import torch
from torch import nn
from torch import optim
import numpy as np

# Don't move this stuff, it needs to be run this early to work
import elektronn3
from torch.nn.modules.loss import MSELoss
from torch.utils import data
elektronn3.select_mpl_backend('Agg')

from elektronn3.training import Trainer, Backup
from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.models.unet import UNet

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import albumentations

from training.tifdirdata import TifDirData2d

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
parser.add_argument(
    '-m', '--max-steps', type=int, default=40_000,
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

VEC_DT = False
DT = True
MULTILABEL = False

data_root = '~/tumdata/'

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

if DT:
    out_channels = 1 if not VEC_DT else 2
else:
    out_channels = len(label_names)

model = UNet(
    out_channels=out_channels,
    n_blocks=4,
    start_filts=32,
    activation='relu',
    normalization='batch',
    dim=2
).to(device)

# USER PATHS
save_root = os.path.expanduser('~/tumtrainings')

max_steps = args.max_steps
lr = 1e-3
lr_stepsize = 1000
lr_dec = 0.9
batch_size = 8


if args.resume is not None:  # Load pretrained network params
    model.load_state_dict(torch.load(os.path.expanduser(args.resume)))

dataset_mean = (128.0,)
dataset_std = (128.0,)

dt_scale = 30

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.RandomCrop((1024, 1024)),
    transforms.Normalize(mean=dataset_mean, std=dataset_std, inplace=True),
    transforms.RandomFlip(ndim_spatial=2),
]

train_transform = common_transforms + [
    transforms.RandomCrop((768, 768)),
    transforms.AlbuSeg2d(albumentations.ShiftScaleRotate(
        p=0.99, rotate_limit=180, shift_limit=0.0625, scale_limit=0.1, interpolation=2
    )),  # interpolation=2 means cubic interpolation (-> cv2.CUBIC constant).
    # transforms.ElasticTransform(prob=0.5, sigma=2, alpha=5),
    # transforms.AdditiveGaussianNoise(prob=0.9, sigma=0.1),
    # transforms.RandomGammaCorrection(prob=0.9, gamma_std=0.2),
    # transforms.RandomBrightnessContrast(prob=0.9, brightness_std=0.125, contrast_std=0.125),
]

valid_transform = common_transforms + []

if DT:
    train_transform.append(transforms.DistanceTransformTarget(scale=dt_scale, vector=VEC_DT))
    valid_transform.append(transforms.DistanceTransformTarget(scale=dt_scale, vector=VEC_DT))

train_transform = transforms.Compose(train_transform)
valid_transform = transforms.Compose(valid_transform)

# Specify data set

if MULTILABEL:
    valid_image_numbers = [5, 10, 32, 42]
    train_image_numbers = [i for i in range(1, 54 + 1) if i not in valid_image_numbers]
else:
    # valid_image_numbers = [5, 10]
    # train_image_numbers = [i for i in range(1, 15 + 1) if i not in valid_image_numbers]
    valid_image_numbers = [22, 32, 42]
    train_image_numbers = [i for i in range(16, 54 + 1) if i not in valid_image_numbers]
    
print('Training on images ', train_image_numbers)
print('Validating on images ', valid_image_numbers)


train_dataset = TifDirData2d(
    data_root=data_root,
    label_names=label_names,
    image_numbers=train_image_numbers,
    multilabel_targets=MULTILABEL,
    transform=train_transform,
    target_dtype=target_dtype,
    invert_labels=True,
    epoch_multiplier=600 // len(train_image_numbers),
)

valid_dataset = TifDirData2d(
    data_root=data_root,
    label_names=label_names,
    image_numbers=valid_image_numbers,
    multilabel_targets=MULTILABEL,
    transform=valid_transform,
    target_dtype=target_dtype,
    invert_labels=True,
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

if DT:
    criterion = MSELoss()
else:
    if MULTILABEL:
        _cw = [1.0 for _ in label_names]
        _cw[0] = 0.2  # reduced loss weight for background labels
        class_weights = torch.tensor(_cw).to(device)
        criterion = nn.BCEWithLogitsLoss()#(pos_weight=class_weights)
    else:
        class_weights = torch.tensor([0.2, 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

if DT:
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

exp_name = args.exp_name
if exp_name is None:
    exp_name = ''
timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
exp_name = f'{exp_name}__{model.__class__.__name__ + "__" + timestamp}'
exp_name = f'{_VEC_DT}{_MULTILABEL}{_DT}{exp_name}'

# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batch_size=batch_size,
    num_workers=4,
    save_root=save_root,
    exp_name=exp_name,
    inference_kwargs=inference_kwargs,
    save_jit=None,
    schedulers={"lr": lr_sched},
    valid_metrics=valid_metrics,
    out_channels=out_channels,
    mixed_precision=True,
    extra_save_steps=list(range(0, 20_000, 2000)),
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
