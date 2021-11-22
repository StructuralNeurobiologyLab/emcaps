#!/usr/bin/env python3


"""
Evaluates a patch classifier model trained by training/patchtrain.py

"""

import argparse
import datetime
from math import inf
import os
import random
from typing import Literal
from elektronn3.data.transforms.transforms import RandomCrop

import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd


# Don't move this stuff, it needs to be run this early to work
import elektronn3
from torch.nn.modules.loss import MSELoss
from torch.utils import data
elektronn3.select_mpl_backend('Agg')

from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.inference import Predictor

from training.tifdirdata import Patches

from models.effnetv2 import effnetv2_s, effnetv2_m
from models import effnetv2


parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument(
    '-m', '--model-path', metavar='PATH',
    help='Path to pretrained model which to use.',
    default='~/tum/trainings3/__EffNetV2__21-11-15_19-27-08/model_step18000.pt'
)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()


# Set up all RNG seeds, set level of determinism
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

EFULLNAMES = {
    'mx': 'MxEnc',
    'qt': 'QtEnc',
    0: 'MxEnc',
    1: 'QtEnc'
}

data_root = '~/tumdata2/'

out_channels = 2


# USER PATHS
save_root = os.path.expanduser('~/tum/trainings3')

patches_v2_root = os.path.expanduser('~/tum/patches_v2')

dataset_mean = (128.0,)
dataset_std = (128.0,)

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.Normalize(mean=dataset_mean, std=dataset_std, inplace=False),
    # transforms.RandomFlip(ndim_spatial=2),
]
valid_transform = common_transforms + []
valid_transform = transforms.Compose(valid_transform)


valid_dataset = Patches(
    train=False,
    # transform=valid_transform,  # Don't transform twice (already transformed by predictor below)
    epoch_multiplier=1,
)


predictor = Predictor(
        model=os.path.expanduser(args.model_path),
        device=device,
        # float16=True,
        transform=valid_transform,  
        apply_softmax=True,  # Not necessary because of subsequent argmax
        # apply_argmax=True,
)

n_correct = 0
n_total = len(valid_dataset)

# Load gt sheet for restoring original patch_id indexing
gt_sheet = pd.read_excel(f'{patches_v2_root}/samples_gt.xlsx')

predictions = {}
meta = valid_dataset.meta
for i in range(len(valid_dataset)):
    sample = valid_dataset[i]
    inp = sample['inp'][None]
    out = predictor.predict(inp)
    pred = out[0].argmax(0).item()

    confidence = out[0].numpy().ptp()  # peak-to-peak as confidence proxy

    target = sample['target'].item()
    pred_label = EFULLNAMES[pred]
    target_label = EFULLNAMES[target]

    if pred == target:
        n_correct += 1

    # We have different patch_id indices in the gt_sheet for human eval because
    # of an index reset at the bottom of patchifyseg.py,
    # so we have to remap to the gt_sheet entry by finding the corresponding
    # patch_fname (which hasn't changed).
    patch_fname = valid_dataset.meta.patch_fname.iloc[i]
    gts_id = gt_sheet[gt_sheet.patch_fname == patch_fname].patch_id.item()

    predictions[gts_id] = (pred_label, confidence)

print(f'{n_correct} correct out of {n_total}')
print(f' -> accuracy: {100 * n_correct / n_total:.2f}%')

predictions = pd.DataFrame.from_dict(predictions, orient='index', columns=['enctype', 'confidence'])

predictions = predictions.sort_index().convert_dtypes()
predictions.to_excel(f'{patches_v2_root}/samples_nnpredictions.xlsx', index_label='patch_id', float_format='%.2f')

# TODO: Save predictions

import IPython ; IPython.embed(); raise SystemExit
