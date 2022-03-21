#!/usr/bin/env python3


"""
Evaluates a patch classifier model trained by training/patchtrain.py

"""

# TODO: Update for v4 data format

import argparse
import datetime
from math import inf
import os
import random
from typing import Literal
from elektronn3.data.transforms.transforms import RandomCrop

import matplotlib.pyplot as plt
import yaml
import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Don't move this stuff, it needs to be run this early to work
import elektronn3
from torch.nn.modules.loss import MSELoss
from torch.utils import data
elektronn3.select_mpl_backend('Agg')

from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.inference import Predictor

from training.tifdirdata import UPatches

from models.effnetv2 import effnetv2_s, effnetv2_m


parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument(
    '-m', '--model-path', metavar='PATH',
    help='Path to pretrained model which to use.',
    default='/wholebrain/scratch/mdraw/tum/patch_trainings_v4a_uni/erasemaskbg___EffNetV2__22-03-19_02-42-10/model_final.pt',
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


out_channels = 8

DILATE_MASKS_BY = 5

valid_split_path = './class_info.yaml'
with open(valid_split_path) as f:
    CLASS_INFO = yaml.load(f, Loader=yaml.FullLoader)

CLASS_IDS = CLASS_INFO['class_ids']
CLASS_NAMES = list(CLASS_IDS.keys())

SHORT_CLASS_NAMES = [name[4:15] for name in CLASS_NAMES]

# USER PATHS

patches_root = os.path.expanduser('/wholebrain/scratch/mdraw/tum/patches_v4a_uni/')

dataset_mean = (128.0,)
dataset_std = (128.0,)

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.Normalize(mean=dataset_mean, std=dataset_std, inplace=False),
]
valid_transform = common_transforms + []
valid_transform = transforms.Compose(valid_transform)


valid_dataset = UPatches(
    descr_sheet=(f'{patches_root}/patchmeta_traintest.xlsx', 'Sheet1'),
    train=False,
    # transform=valid_transform,  # Don't transform twice (already transformed by predictor below)
    dilate_masks_by=DILATE_MASKS_BY,
    erase_mask_bg=True,
)


predictor = Predictor(
    model=os.path.expanduser(args.model_path),
    device=device,
    # float16=True,
    transform=valid_transform,  
    apply_softmax=True,
    # apply_argmax=True,
)

n_correct = 0
n_total = 0

# Load gt sheet for restoring original patch_id indexing
gt_sheet = pd.read_excel(f'{patches_root}/samples_gt.xlsx')


pred_labels = []
target_labels = []

predictions = {}
meta = valid_dataset.meta
for i in range(len(valid_dataset)):

    # We have different patch_id indices in the gt_sheet for human eval because
    # of an index reset at the bottom of patchifyseg.py,
    # so we have to remap to the gt_sheet entry by finding the corresponding
    # patch_fname (which hasn't changed).
    patch_fname = valid_dataset.meta.patch_fname.iloc[i]

    # TODO: FIX
    ####
    # gt_match = gts_id = gt_sheet[gt_sheet.patch_fname == patch_fname]
    # if gt_match.empty:
    #     import IPython ; IPython.embed(); raise SystemExit
    #     continue  # Not found in gt_sheet, so skip this patch
    # gts_id = gt_match.patch_id.item()

    sample = valid_dataset[i]
    inp = sample['inp'][None]
    out = predictor.predict(inp)
    pred = out[0].argmax(0).item()

    confidence = out[0].numpy().ptp()  # peak-to-peak as confidence proxy

    target = sample['target'].item()

    pred_label = SHORT_CLASS_NAMES[pred]
    target_label = SHORT_CLASS_NAMES[target]

    pred_labels.append(pred_label)
    target_labels.append(target_label)

    n_total += 1
    if pred == target:
        n_correct += 1


    # TODO: Fix ##
    ###
    # predictions[gts_id] = (pred_label, confidence)

print(f'{n_correct} correct out of {n_total}')
print(f' -> accuracy: {100 * n_correct / n_total:.2f}%')

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 10))


cma = ConfusionMatrixDisplay.from_predictions(target_labels, pred_labels, labels=SHORT_CLASS_NAMES[2:], normalize='pred', xticks_rotation='vertical', ax=ax)
cma.figure_.savefig(f'{patches_root}/patch_confusion_matrix.pdf')

predictions = pd.DataFrame.from_dict(predictions, orient='index', columns=['class', 'confidence'])

predictions = predictions.sort_index().convert_dtypes()
predictions.to_excel(f'{patches_root}/samples_nnpredictions.xlsx', index_label='patch_id', float_format='%.2f')

# TODO: Save predictions

import IPython ; IPython.embed(); raise SystemExit

