#!/usr/bin/env python3


"""
Evaluates a patch classifier model trained by training/patchtrain.py
Supports majority votes.

"""

import argparse
import os
import random
from typing import Literal
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('auto')

from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.inference import Predictor

from emcaps.analysis.cf_matrix import plot_confusion_matrix

parser = argparse.ArgumentParser(description='Evaluate a network.')
parser.add_argument(
    '-m', '--model-path', metavar='PATH',
    help='Path to pretrained model which to use.',
    # default='/wholebrain/scratch/mdraw/tum/patch_trainings_v6_notm_nodro/erasemaskbg_notm_dr5_M_ra__EffNetV2__22-05-16_01-56-49/model_step80000.pt'  # best without Tm, for patches_v6_dr5
    # default='/wholebrain/scratch/mdraw/tum/patch_trainings_v6e/erasemaskbg_S__EffNetV2__22-05-20_17-02-05/model_step80000.pts'  # best with Tm, for patches_v6e_dr5

    # default='/wholebrain/scratch/mdraw/tum/patch_trainings_v7_trdro_evdro_dr5/M_erasemaskbg___EffNetV2__22-06-03_16-15-30/model_final.pts'  # Best for DRO v7
    # default='/wholebrain/scratch/mdraw/tum/patch_trainings_v7_trhek_evhek_dr5/M_erasemaskbg___EffNetV2__22-06-03_16-25-11/model_final.pts',  # Best for HEK v7
    default='/wholebrain/scratch/mdraw/tum/patch_trainings_v10c_tr-gt_ev-all_dr5__gt/erasemaskbg___EffNetV2__22-10-07_03-33-09/model_step80000.pts', # Best for ALL v10c (human gt patches)

    # default='/wholebrain/scratch/mdraw/tum/patch_trainings_v7_tr-hgt_ev-dro_gdr5__gt/erasemaskbg___EffNetV2__22-06-07_12-13-56/model_final.pts'  # Human GT -> DRO
)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
    '-n', '--nmaxsamples', type=int, default=0,
    help='Maximum of patch samples per image for majority vote. 0 means no limit (all patches are used). (default: 0).'
)
parser.add_argument(
    '-r', '--rdraws', type=int, default=1000,
    help='Number of independent draws for majority vote.'
)
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()


# Set up all RNG seeds, set level of determinism
random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')


CM_SHOW_PERCENTAGES = True

out_channels = 8

from emcaps.utils import CLASS_NAMES, CLASS_IDS


ENABLE_BINARY_1M = False  # restrict to only binary classification into 1M-Qt vs 1M-Mx
ENABLE_TM = True  # Enable 1M-Tm type in evaluation

CLASS_NAMES_IN_USE = list(CLASS_NAMES.values())[2:]

if not ENABLE_TM:
    CLASS_NAMES_IN_USE = CLASS_NAMES_IN_USE[:-1]

if ENABLE_BINARY_1M:
    CLASS_NAMES_IN_USE = CLASS_NAMES_IN_USE[:2]

# USER PATHS

if os.getenv('CLUSTER') == 'WHOLEBRAIN':
    path_prefix = Path('/wholebrain/scratch/mdraw/tum/').expanduser()
else:
    path_prefix = Path('~/tum/').expanduser()

# patches_root = path_prefix / 'patches_v6d_generalization_dro_dr5/'
# patches_root = path_prefix / 'patches_v6_notm_nodro_dr5/'
# patches_root = path_prefix / 'patches_v6_dr5/'
patches_root = path_prefix / 'patches_v6e_dr5/'

# patches_root = path_prefix / 'patches_v7_trhek_evdro_dr5'
# patches_root = path_prefix / 'patches_v7_trdro_evdro_dr5'


dataset_mean = (128.0,)
dataset_std = (128.0,)

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.Normalize(mean=dataset_mean, std=dataset_std, inplace=False),
]
valid_transform = common_transforms + []
valid_transform = transforms.Compose(valid_transform)


# SAMPLES_PER_IMG = 'all'
# SAMPLES_PER_IMG = 1
MAX_SAMPLES_PER_GROUP = args.nmaxsamples



# GROUPKEY = 'img_num'
GROUPKEY = 'enctype'


predictor = Predictor(
    model=os.path.expanduser(args.model_path),
    device=device,
    # float16=True,
    transform=valid_transform,
    apply_softmax=True,
    # apply_argmax=True,
)

# meta = pd.read_excel(f'{patches_root}/patchmeta_traintest_v5names.xlsx', sheet_name='Sheet1', index_col=0)
meta = pd.read_excel(f'{patches_root}/patchmeta_traintest.xlsx', sheet_name='Sheet1', index_col=0)

vmeta = meta.loc[meta.validation == True]

# vmeta = vmeta.loc[vmeta.img_num == 136]  # TEST

# vmeta = vmeta.loc[vmeta.enctype != '1M-Tm']

# assert vmeta.loc[vmeta.img_num.isin(range(90, 94+1))].enctype.unique() == '1M-Qt'  # Ensure v7 labeling
# vmeta.loc[vmeta.img_num.isin(range(90, 94+1)), 'enctype'] = '1M-Qt'  # Relabel wrong images
# assert vmeta.loc[vmeta.img_num.isin(range(90, 94+1))].enctype.unique() == '1M-Qt'

# qtp = vmeta.loc[vmeta.enctype == '1M-Qt'].sample(387).index
# vmeta = vmeta.loc[(vmeta.index.isin(qtp)) | (vmeta.enctype == '1M-Mx')]

print('\n== Patch selection ==')

all_targets = []
all_preds = []


# min_group_samples = min((vmeta[GROUPKEY].isin([g])).shape[0] for g in vmeta[GROUPKEY].unique())
min_group_samples = np.inf
for g in vmeta[GROUPKEY].unique():
    min_group_samples = min(min_group_samples, (vmeta.loc[vmeta[GROUPKEY] == g]).shape[0])

if MAX_SAMPLES_PER_GROUP == 0:
    subseq_splits = [None]  # TODO: [None] has ambiguous meaning below
else:
    subseq_splits = []
    for k in range(0, min_group_samples // MAX_SAMPLES_PER_GROUP, MAX_SAMPLES_PER_GROUP):
        subseq_splits.append(range(k, k + MAX_SAMPLES_PER_GROUP))

if MAX_SAMPLES_PER_GROUP > 1:
    splits = [None] * args.rdraws  # do random sampling
else:
    splits = subseq_splits

# splits = range(min_group_samples)  # iterate over all individuals


if args.verbose:
    _print = print
else:
    def _print(*args, **kwargs): pass


def evaluate(vmeta, groupkey, split=None):
    group_preds = {}
    group_pred_labels = {}
    group_targets = {}
    group_target_labels = {}

    # img_preds = {k: [] for k in CLASS_IDS.keys()}
    # img_pred_labels = {k: [] for k in CLASS_IDS.keys()}
    # img_targets = {k: [] for k in CLASS_IDS.keys()}
    # img_target_labels = {k: [] for k in CLASS_IDS.keys()}

    _print(f'Grouping by {groupkey}')
    for group in vmeta[groupkey].unique():
        # For each group instance:
        gvmeta = vmeta.loc[vmeta[groupkey] == group]
        assert len(gvmeta.enctype.unique() == 1)
        target_label = gvmeta.iloc[0].enctype
        target = CLASS_IDS[target_label]

        _print(f'Group {group} yields {gvmeta.shape[0]} patches.')

        group_preds[group] = []
        group_pred_labels[group] = []
        group_targets[group] = []
        group_target_labels[group] = []

        if MAX_SAMPLES_PER_GROUP > 0:
            if split is None:  # Randomly sample only MAX_SAMPLES_PER_GROUP patches
                gvmeta = gvmeta.sample(min(gvmeta.shape[0], MAX_SAMPLES_PER_GROUP))
            elif isinstance(split, int):
                gvmeta = gvmeta.iloc[split:split+1]
            else:
                gvmeta = gvmeta.iloc[split]
            _print(f'-> After reducing to a maximum of {MAX_SAMPLES_PER_GROUP}, we now have:')
            _print(f'Group {group} yields {gvmeta.shape[0]} patches.')


        preds = []
        targets = []
        pred_labels = []
        target_labels = []
        for patch_entry in gvmeta.itertuples():
            raw_fname = patch_entry.patch_fname
            nobg_fpath = patches_root / 'nobg' / raw_fname.replace('raw', 'nobg')
            patch = iio.imread(nobg_fpath).astype(np.float32)[None][None]

            out = predictor.predict(patch)
            if ENABLE_BINARY_1M:
                # Slice the 2 classes of interest, do argmax between them and restore class ID
                pred = out[0, 2:4].argmax(0).item() + 2
            else:
                pred = out[0].argmax(0).item()

            pred_label = CLASS_NAMES[pred]

            preds.append(pred)
            targets.append(target)
            pred_labels.append(pred_label)
            target_labels.append(target_label)

            group_preds[group].append(pred)
            group_pred_labels[group].append(pred_label)

            # group_targets[group].append(target)
            # group_target_labels[group].append(target_label)

            group_targets[group] = target
            group_target_labels[group] = target_label


            all_targets.append(target)
            all_preds.append(pred)

        preds = np.array(preds)
        targets = np.array(targets)

        # print(group_pred_labels)
        # print(group_target_labels)
        # print(group_preds)
        # print(group_targets)


    group_majority_preds = {}
    group_majority_pred_names = {}
    group_correct_ratios = {}
    for k, v in group_preds.items():
        group_majority_preds[k] = np.argmax(np.bincount(v))
        # TODO: This is broken!
        # if pred in v:
        #     group_correct_ratios[k] = np.bincount(v)[pred] / len(v)
        # else:  # target does not appear in predicted values -> 0 correct
        #     group_correct_ratios[k] = 0.
        group_majority_pred_names[k] = CLASS_NAMES[group_majority_preds[k]]

    _print('\n\n==  Patch classification ==\n')
    for group in group_preds.keys():
        _print(f'Group {group}\nTrue class: {group_target_labels[group]}\nPredicted classes: {group_pred_labels[group]}\n-> Majority vote result: {group_majority_pred_names[group]}')
        # print(f'-> Fraction of correct predictions: {img_correct_ratios[i] * 100:.1f}%\n')  # Broken



    if False:  # Sanity check: Calculate confusion matrix entries myself
        for a in range(2, 8):
            for b in range(2, 8):
                v = np.sum((targets == a) & (preds == b))
                print(f'T: {CLASS_NAMES[a]}, P: {CLASS_NAMES[b]} -> {v}')

    group_targets_list = []
    group_majority_preds_list = []
    for g in group_targets.keys():
        group_targets_list.append(group_targets[g])
        group_majority_preds_list.append(group_majority_preds[g])

    return group_targets_list, group_majority_preds_list


full_group_targets = []
full_group_majority_preds = []


for split in tqdm(splits):
    split_group_targets, split_group_majority_preds = evaluate(vmeta, groupkey=GROUPKEY, split=split)
    full_group_targets.extend(split_group_targets)
    full_group_majority_preds.extend(split_group_majority_preds)


all_preds = np.stack(all_preds)
all_targets = np.stack(all_targets)
instance_n_correct = np.sum(all_targets == all_preds)
instance_n_total = all_targets.shape[0]
instance_avg_accuracy = instance_n_correct / instance_n_total
print(f'Instance-level average accuracy: {instance_avg_accuracy * 100:.2f}%')

full_group_targets = np.stack(full_group_targets)
full_group_majority_preds = np.stack(full_group_majority_preds)
group_n_correct = np.sum(full_group_targets == full_group_majority_preds)
group_n_total = full_group_targets.shape[0]
group_avg_accuracy = group_n_correct / group_n_total
print(f'Group-level average accuracy: {group_avg_accuracy * 100:.2f}%')


cm = confusion_matrix(full_group_targets, full_group_majority_preds)

fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5.5))

repr_max_samples = MAX_SAMPLES_PER_GROUP if MAX_SAMPLES_PER_GROUP > 0 else 'all'

if CM_SHOW_PERCENTAGES:
    cma = plot_confusion_matrix(cm, categories=CLASS_NAMES_IN_USE, normalize='true', cmap='viridis', sum_stats=False, ax=ax, cbar=False, percent=True)
    # ax.set_title(f'Majority vote for N = {repr_max_samples} patches per image (top: count, bottom: percentages normalized over true labels)\n')
    ax.set_title(f'Confusion matrix for N = {repr_max_samples} (top: count, bottom: percentages normalized over true labels)\nAvg. accuracy: {group_avg_accuracy * 100:.2f}%\n')
else:
    cma = plot_confusion_matrix(cm, categories=CLASS_NAMES_IN_USE, normalize='true', cmap='viridis', sum_stats=False, ax=ax, cbar=False, percent=False)
    ax.set_title(f'Confusion matrix for N = {repr_max_samples} (absolute counts)\n')


plt.tight_layout()
plt.savefig(f'{patches_root}/{"bin_" if ENABLE_BINARY_1M else ""}patch_confusion_matrix_n{repr_max_samples}.pdf', bbox_inches='tight')

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