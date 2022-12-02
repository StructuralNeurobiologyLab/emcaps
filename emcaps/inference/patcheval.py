#!/usr/bin/env python3
"""
Evaluates a patch classifier model trained by training/patchtrain.py
Supports majority votes.

Example sage (fish loop):
```
for g in all all2 all3 hek hek2
    emcaps-patcheval tr_group=$g patcheval.rdraws=1000 patcheval.max_samples=1
end
```
"""

import random
from typing import Literal
from pathlib import Path

import hydra
import logging
import imageio.v3 as iio
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from omegaconf import DictConfig

# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('auto')

from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.inference import Predictor

from emcaps.analysis.cf_matrix import plot_confusion_matrix
from emcaps import utils
from emcaps.utils import inference_utils as iu


@hydra.main(version_base='1.2', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Set up all RNG seeds, set level of determinism
    random_seed = cfg.patchtrain.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    all_enctypes = utils.CLASS_GROUPS['simple_hek']
    constrain_classifier = cfg.patcheval.constrain_classifier
    eval_path = Path(cfg.patcheval.eval_out_path)
    if cfg.patcheval.use_constraint_suffix and not constrain_classifier == all_enctypes:
        suffix = '__constrained_to'
        for c in constrain_classifier:
            suffix = f'{suffix}_{c}'
        eval_path = eval_path.with_name(f'{eval_path.name}_{suffix}')
    eval_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('emcaps-patcheval')
    fh = logging.FileHandler(f'{eval_path}/patcheval.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on device: {device}')

    classifier_path = cfg.patcheval.classifier
    logger.info(f'Using classifier {classifier_path}')
    logger.info(f'Constraining classifier to classes:\n  {constrain_classifier}')

    CLASS_NAMES_IN_USE = all_enctypes  # TODO: Dynamic population from constrain_classifier
    cm_labels = range(2, 8)

    # USER PATHS
    ds_sheet_path = Path(cfg.patcheval.patch_ds_sheet)
    patches_path = ds_sheet_path.parent

    MAX_SAMPLES_PER_GROUP = cfg.patcheval.max_samples

    GROUPKEY = 'enctype'

    meta = pd.read_excel(ds_sheet_path, 0, index_col=0)

    # Metadata filtered to only include validation patches
    vmeta = meta.loc[meta.validation == True]
    # vmeta = vmeta.loc[vmeta.img_num == 136]  # TEST

    if not 'dataset_name' in vmeta.columns:
        # Workaroud until 'dataset_name' is always present in patch meta: Populate from image-level source meta sheet
        vmeta = utils.attach_dataset_name_column(vmeta, src_sheet_path=cfg.sheet_path)

    dataset_names = vmeta.dataset_name.unique().tolist()
    # Prepend special dataset name that instructs to use all datasets
    dataset_names = ['all_datasets'] + dataset_names
    for _i, dataset_name in enumerate(dataset_names):
        if dataset_name == 'all_datasets':  #  Use all datasets
            # dvmeta = vmeta.loc[vmeta.dataset_name.isin(dataset_names)]
            dvmeta = vmeta.copy()  # Keep all rows
        else:  # Metadata filtered to only include validation patches that belong to one "Dataset Name"
            dvmeta = vmeta.loc[vmeta.dataset_name == dataset_name]

        logger.info(f'\n== Patch selection: {dataset_name}  ({_i} / {len(dataset_names)}) ==')

        all_targets = []
        all_preds = []

        min_group_samples = np.inf
        for g in dvmeta[GROUPKEY].unique():
            min_group_samples = min(
                min_group_samples,
                (dvmeta.loc[dvmeta[GROUPKEY] == g]).shape[0]
            )

        if MAX_SAMPLES_PER_GROUP == 0:
            subseq_splits = [None]  # TODO: [None] has ambiguous meaning below
        else:
            subseq_splits = []
            for k in range(0, min_group_samples // MAX_SAMPLES_PER_GROUP, MAX_SAMPLES_PER_GROUP):
                subseq_splits.append(range(k, k + MAX_SAMPLES_PER_GROUP))

        if MAX_SAMPLES_PER_GROUP > 1:
            splits = [None] * cfg.patcheval.rdraws  # do random sampling
        else:
            splits = subseq_splits

        # splits = range(min_group_samples)  # iterate over all individuals

        def evaluate(dvmeta, groupkey, split=None):
            """Perform eval on one split. Usually groupkey is enctype."""
            group_preds = {}
            group_pred_labels = {}
            group_targets = {}
            group_target_labels = {}

            logger.debug(f'Grouping by {groupkey}')
            for group in dvmeta[groupkey].unique():
                # For each group instance:
                gdvmeta = dvmeta.loc[dvmeta[groupkey] == group]
                assert len(gdvmeta.enctype.unique() == 1)
                target_label = gdvmeta.iloc[0].enctype
                target = utils.CLASS_IDS[target_label]

                logger.debug(f'Group {group} yields {gdvmeta.shape[0]} patches.')

                group_preds[group] = []
                group_pred_labels[group] = []
                group_targets[group] = []
                group_target_labels[group] = []

                if MAX_SAMPLES_PER_GROUP > 0:
                    if split is None:  # Randomly sample only MAX_SAMPLES_PER_GROUP patches
                        gdvmeta = gdvmeta.sample(min(gdvmeta.shape[0], MAX_SAMPLES_PER_GROUP))
                    elif isinstance(split, int):
                        gdvmeta = gdvmeta.iloc[split:split + 1]
                    else:
                        gdvmeta = gdvmeta.iloc[split]
                    logger.debug(f'-> After reducing to a maximum of {MAX_SAMPLES_PER_GROUP}, we now have:')
                    logger.debug(f'Group {group} yields {gdvmeta.shape[0]} patches.')

                preds = []
                targets = []
                pred_labels = []
                target_labels = []
                for patch_entry in gdvmeta.itertuples():
                    raw_fname = patch_entry.patch_fname
                    nobg_fpath = patches_path / 'nobg' / raw_fname.replace('raw', 'nobg')
                    patch = iio.imread(nobg_fpath).astype(np.float32)

                    pred = iu.classify_patch(patch, classifier_variant=classifier_path, allowed_classes=constrain_classifier)

                    pred_label = utils.CLASS_NAMES[pred]

                    preds.append(pred)
                    targets.append(target)
                    pred_labels.append(pred_label)
                    target_labels.append(target_label)

                    group_preds[group].append(pred)
                    group_pred_labels[group].append(pred_label)

                    group_targets[group] = target
                    group_target_labels[group] = target_label

                    all_targets.append(target)
                    all_preds.append(pred)

                preds = np.array(preds)
                targets = np.array(targets)

            group_majority_preds = {}
            group_majority_pred_names = {}
            for k, v in group_preds.items():
                group_majority_preds[k] = np.argmax(np.bincount(v))
                group_majority_pred_names[k] = utils.CLASS_NAMES[group_majority_preds[k]]

            logger.debug('\n\n==  Patch classification ==\n')
            for group in group_preds.keys():
                logger.debug(f'Group {group}\nTrue class: {group_target_labels[group]}\nPredicted classes: {group_pred_labels[group]}\n-> Majority vote result: {group_majority_pred_names[group]}')

            if False:  # Sanity check: Calculate confusion matrix entries myself
                for a in range(2, 8):
                    for b in range(2, 8):
                        v = np.sum((targets == a) & (preds == b))
                        logger.debug(f'T: {utils.CLASS_NAMES[a]}, P: {utils.CLASS_NAMES[b]} -> {v}')

            group_targets_list = []
            group_majority_preds_list = []
            for g in group_targets.keys():
                group_targets_list.append(group_targets[g])
                group_majority_preds_list.append(group_majority_preds[g])

            return group_targets_list, group_majority_preds_list

        full_group_targets = []
        full_group_majority_preds = []

        for split in tqdm(splits, dynamic_ncols=True):
            split_group_targets, split_group_majority_preds = evaluate(dvmeta, groupkey=GROUPKEY, split=split)
            full_group_targets.extend(split_group_targets)
            full_group_majority_preds.extend(split_group_majority_preds)

        all_preds = np.stack(all_preds)
        all_targets = np.stack(all_targets)
        instance_n_correct = np.sum(all_targets == all_preds)
        instance_n_total = all_targets.shape[0]
        instance_avg_accuracy = instance_n_correct / instance_n_total
        logger.info(f'Instance-level average accuracy: {instance_avg_accuracy * 100:.2f}%')

        full_group_targets = np.stack(full_group_targets)
        full_group_majority_preds = np.stack(full_group_majority_preds)
        group_n_correct = np.sum(full_group_targets == full_group_majority_preds)
        group_n_total = full_group_targets.shape[0]
        group_avg_accuracy = group_n_correct / group_n_total
        logger.info(f'Group-level average accuracy: {group_avg_accuracy * 100:.2f}%')

        cm = confusion_matrix(full_group_targets, full_group_majority_preds, labels=cm_labels)

        fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5.5))

        repr_max_samples = MAX_SAMPLES_PER_GROUP if MAX_SAMPLES_PER_GROUP > 0 else 'all'

        cma = plot_confusion_matrix(cm, categories=CLASS_NAMES_IN_USE, normalize='true', cmap='viridis', sum_stats=False, ax=ax, cbar=False, percent=True)
        ax.set_title(f'Confusion matrix for {dataset_name}, N = {repr_max_samples} (top: count, bottom: percentages normalized over true labels)\nAvg. accuracy: {group_avg_accuracy * 100:.2f}%\n')

        plt.tight_layout()
        plt.savefig(f'{eval_path}/patch_cm_{dataset_name}_n{repr_max_samples}.pdf', bbox_inches='tight')

        # TODO: Save predictions

        # predictions = pd.DataFrame.from_dict(img_majority_preds, orient='index', columns=['class', 'confidence'])
        # predictions = predictions.sort_index().convert_dtypes()
        # predictions.to_excel(f'{eval_path}/samples_nnpredictions.xlsx', index_label='patch_id', float_format='%.2f')

        # import IPython ; IPython.embed(); raise SystemExit


if __name__ == '__main__':
    main()