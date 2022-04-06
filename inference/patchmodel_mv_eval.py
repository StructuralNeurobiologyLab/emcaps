#!/usr/bin/env python3


"""
Evaluates a patch classifier model trained by training/patchtrain.py
Supports majority votes.

"""

import argparse
import datetime
from locale import normalize
from math import inf
import os
import random
from tkinter.tix import MAX
from typing import Literal
from pathlib import Path
from unicodedata import category
from elektronn3.data.transforms.transforms import RandomCrop

import imageio
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
# elektronn3.select_mpl_backend('Agg')

from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.inference import Predictor

from training.tifdirdata import UPatches

from models.effnetv2 import effnetv2_s, effnetv2_m
from analysis.cf_matrix import plot_confusion_matrix

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument(
    '-m', '--model-path', metavar='PATH',
    help='Path to pretrained model which to use.',
    default='/wholebrain/scratch/mdraw/tum/patch_trainings_v4a_uni/erasemaskbg___EffNetV2__22-03-19_02-42-10/model_final.pt',
)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
    '-n', '--nmaxsamples', type=int, default=0,
    help='Maximum of patch samples per image for majority vote. 0 means no limit (all patches are used). (default: 0).'
)
args = parser.parse_args()


# Set up all RNG seeds, set level of determinism
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')


out_channels = 8

valid_split_path = './class_info.yaml'
with open(valid_split_path) as f:
    CLASS_INFO = yaml.load(f, Loader=yaml.FullLoader)

# CLASS_IDS = CLASS_INFO['class_ids']
CLASS_IDS = CLASS_INFO['class_ids_v5']
CLASS_NAMES = list(CLASS_IDS.keys())
CLASS_NAMES_IN_USE = CLASS_NAMES[2:]


# USER PATHS

if os.getenv('CLUSTER') == 'WHOLEBRAIN':
    path_prefix = Path('/wholebrain/scratch/mdraw/tum/').expanduser()
else:
    path_prefix = Path('~/tum/').expanduser()

# patches_root = path_prefix / 'patches_v4a_uni/'
patches_root = path_prefix / 'patches_v5/'

dataset_mean = (128.0,)
dataset_std = (128.0,)

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.Normalize(mean=dataset_mean, std=dataset_std, inplace=False),
]
valid_transform = common_transforms + []
valid_transform = transforms.Compose(valid_transform)


# MAX_SAMPLES_PER_IMG = 'all'
# MAX_SAMPLES_PER_IMG = 1
MAX_SAMPLES_PER_IMG = args.nmaxsamples


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

img_preds = {}
img_pred_labels = {}
img_targets = {}
img_target_labels = {}

# meta = pd.read_excel(f'{patches_root}/patchmeta_traintest_v5names.xlsx', sheet_name='Sheet1', index_col=0)
meta = pd.read_excel(f'{patches_root}/patchmeta_traintest.xlsx', sheet_name='Sheet1', index_col=0)

vmeta = meta.loc[meta.validation == True]

for i in vmeta.img_num.unique():
    # For each source image:
    imgmeta = vmeta.loc[vmeta.img_num == i]
    # Each image only contains one enctype
    assert len(imgmeta.enctype.unique() == 1)
    target_label = imgmeta.iloc[0].enctype
    target = CLASS_IDS[target_label]

    print(f'Image {i:03d} (class {target_label}) yields {imgmeta.shape[0]} patches.')

    if MAX_SAMPLES_PER_IMG > 0:  # Randomly sample only MAX_SAMPLES_PER_IMG patches
        imgmeta = imgmeta.sample(min(imgmeta.shape[0], MAX_SAMPLES_PER_IMG))
        print(f'-> After reducing to a maximum of {MAX_SAMPLES_PER_IMG}, we now have:')
        print(f'Image {i:03d} (class {target_label}) yields {imgmeta.shape[0]} patches.')


    img_preds[i] = []
    img_pred_labels[i] = []
    img_targets[i] = target
    img_target_labels[i] = target_label

    preds = []
    targets = []
    pred_labels = []
    target_labels = []
    for patch_entry in imgmeta.itertuples():
        raw_fname = patch_entry.patch_fname
        nobg_fname = patches_root / 'nobg' / raw_fname.replace('raw', 'nobg')
        patch = imageio.imread(nobg_fname).astype(np.float32)[None][None]

        out = predictor.predict(patch)
        pred = out[0].argmax(0).item()
        confidence = out[0].numpy().ptp()  # peak-to-peak as confidence proxy

        pred_label = CLASS_NAMES[pred]

        preds.append(pred)
        targets.append(target)
        pred_labels.append(pred_label)
        target_labels.append(target_label)

        img_preds[i].append(pred)
        img_pred_labels[i].append(pred_label)

    preds = np.array(preds)
    targets = np.array(targets)

img_majority_preds = {}
img_majority_pred_names = {}
img_correct_ratios = {}
for k, v in img_preds.items():
    img_majority_preds[k] = np.argmax(np.bincount(v))
    img_correct_ratios[k] = np.bincount(v)[img_targets[k]] / len(v)
    img_majority_pred_names[k] = CLASS_NAMES[img_majority_preds[k]]

for i in img_preds.keys():
    print(f'Image {i}\nTrue: {img_target_labels[i]}\nPred: {img_pred_labels[i]}\nMaVo: {img_majority_pred_names[i]}')
    print(f'MaVo correct ratio: {img_correct_ratios[i] * 100:.1f}%\n')



if False:  # Sanity check: Calculate confusion matrix entries myself
    for a in range(2, 8):
        for b in range(2, 8):
            v = np.sum((targets == a) & (preds == b))
            print(f'T: {CLASS_NAMES[a]}, P: {CLASS_NAMES[b]} -> {v}')

    # T: 1xMT3-MxEnc, P: 1xMT3-MxEnc -> 3
    # T: 1xMT3-MxEnc, P: 1xMT3-QtEnc -> 0
    # T: 1xMT3-MxEnc, P: 2xMT3-MxEnc -> 23
    # T: 1xMT3-MxEnc, P: 2xMT3-QtEnc -> 0
    # T: 1xMT3-MxEnc, P: 3xMT3-QtEnc -> 0
    # T: 1xMT3-MxEnc, P: 1xTmEnc-BC2 -> 4
    # T: 1xMT3-QtEnc, P: 1xMT3-MxEnc -> 0
    # T: 1xMT3-QtEnc, P: 1xMT3-QtEnc -> 27
    # T: 1xMT3-QtEnc, P: 2xMT3-MxEnc -> 0
    # T: 1xMT3-QtEnc, P: 2xMT3-QtEnc -> 2
    # T: 1xMT3-QtEnc, P: 3xMT3-QtEnc -> 1
    # T: 1xMT3-QtEnc, P: 1xTmEnc-BC2 -> 0
    # T: 2xMT3-MxEnc, P: 1xMT3-MxEnc -> 0
    # T: 2xMT3-MxEnc, P: 1xMT3-QtEnc -> 0
    # T: 2xMT3-MxEnc, P: 2xMT3-MxEnc -> 26
    # T: 2xMT3-MxEnc, P: 2xMT3-QtEnc -> 0
    # T: 2xMT3-MxEnc, P: 3xMT3-QtEnc -> 4
    # T: 2xMT3-MxEnc, P: 1xTmEnc-BC2 -> 0
    # T: 2xMT3-QtEnc, P: 1xMT3-MxEnc -> 2
    # T: 2xMT3-QtEnc, P: 1xMT3-QtEnc -> 0
    # T: 2xMT3-QtEnc, P: 2xMT3-MxEnc -> 17
    # T: 2xMT3-QtEnc, P: 2xMT3-QtEnc -> 6
    # T: 2xMT3-QtEnc, P: 3xMT3-QtEnc -> 5
    # T: 2xMT3-QtEnc, P: 1xTmEnc-BC2 -> 0
    # T: 3xMT3-QtEnc, P: 1xMT3-MxEnc -> 0
    # T: 3xMT3-QtEnc, P: 1xMT3-QtEnc -> 0
    # T: 3xMT3-QtEnc, P: 2xMT3-MxEnc -> 9
    # T: 3xMT3-QtEnc, P: 2xMT3-QtEnc -> 0
    # T: 3xMT3-QtEnc, P: 3xMT3-QtEnc -> 21
    # T: 3xMT3-QtEnc, P: 1xTmEnc-BC2 -> 0
    # T: 1xTmEnc-BC2, P: 1xMT3-MxEnc -> 0
    # T: 1xTmEnc-BC2, P: 1xMT3-QtEnc -> 1
    # T: 1xTmEnc-BC2, P: 2xMT3-MxEnc -> 16
    # T: 1xTmEnc-BC2, P: 2xMT3-QtEnc -> 0
    # T: 1xTmEnc-BC2, P: 3xMT3-QtEnc -> 2
    # T: 1xTmEnc-BC2, P: 1xTmEnc-BC2 -> 11


img_targets_list = []
img_majority_preds_list = []
for img_num in img_targets.keys():
    img_targets_list.append(img_targets[img_num])
    img_majority_preds_list.append(img_majority_preds[img_num])

cm = confusion_matrix(img_targets_list, img_majority_preds_list)

fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5.5))

repr_max_samples = MAX_SAMPLES_PER_IMG if MAX_SAMPLES_PER_IMG > 0 else 'all'

cma = plot_confusion_matrix(cm, categories=CLASS_NAMES_IN_USE, normalize='pred', cmap='viridis', sum_stats=False, ax=ax)
ax.set_title(f'Majority vote for N = {repr_max_samples} patches per image (top: count, bottom: percentages normalized over true labels)\n')
plt.tight_layout()
plt.savefig(f'{patches_root}/patch_confusion_matrix_n{repr_max_samples}.pdf', bbox_inches='tight')

# cma = ConfusionMatrixDisplay.from_predictions(target_labels, pred_labels, labels=SHORT_CLASS_NAMES[2:], normalize='pred', xticks_rotation='vertical', ax=ax)
# cma.figure_.savefig(f'{patches_root}/patch_confusion_matrix.pdf')

# predictions = pd.DataFrame.from_dict(img_majority_preds, orient='index', columns=['class', 'confidence'])

# predictions = predictions.sort_index().convert_dtypes()
# predictions.to_excel(f'{patches_root}/samples_nnpredictions.xlsx', index_label='patch_id', float_format='%.2f')

# TODO: Save predictions

# import IPython ; IPython.embed(); raise SystemExit

# label_names = [
#     '1xMT3-MxEnc',
#     '1xMT3-QtEnc',
#     '2xMT3-MxEnc',
#     '2xMT3-QtEnc',
#     '3xMT3-QtEnc',
#     '1xTmEnc-BC2',
# ]