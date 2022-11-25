#!/usr/bin/env python3


"""
Evaluates a patch classifier model trained by training/patchtrain.py
Supports majority votes.

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


def attach_dataset_name_column(patch_meta: pd.DataFrame, src_sheet_path: Path | str, inplace: bool = False) -> pd.DataFrame:
    """Get missing dataset name column from the orginal meta sheet, matching patch img_num to source image num"""
    if not inplace:
        patch_meta = patch_meta.copy()
    src_meta = utils.get_meta(sheet_path=src_sheet_path)  # Get image level meta information
    # There is probably some way to vectorize this but raw iteration is fast enough...
    for row in patch_meta.itertuples():
        dn = src_meta.loc[src_meta.num == row.img_num, 'Dataset Name'].item()
        patch_meta.at[row.Index, 'dataset_name'] = dn
    return patch_meta


@hydra.main(version_base='1.2', config_path='../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Set up all RNG seeds, set level of determinism
    random_seed = cfg.patchtrain.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    logger = logging.getLogger('emcaps-patcheval')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    CM_SHOW_PERCENTAGES = True

    CLASS_NAMES_IN_USE = utils.CLASS_GROUPS['simple_hek']

    # USER PATHS

    ds_sheet_path = Path(cfg.patcheval.patch_ds_sheet)
    patches_root = ds_sheet_path.parent

    # Transformations to be applied to samples before feeding them to the network
    common_transforms = [
        transforms.Normalize(mean=cfg.dataset_mean, std=cfg.dataset_std, inplace=False),
    ]
    valid_transform = common_transforms + []
    valid_transform = transforms.Compose(valid_transform)


    MAX_SAMPLES_PER_GROUP = cfg.patcheval.max_samples

    GROUPKEY = 'enctype'

    classifier_path = cfg.patcheval.classifier
    if classifier_path == 'auto':
        classifier_path = f'effnet_{cfg.tr_group}_{cfg.v}'
        logger.info(f'Using default classifier {classifier_path} based on other config values')
    elif classifier_path != '':
        logger.info(f'Using classifier {classifier_path}')
    classifier = iu.get_model(classifier_path)

    predictor = Predictor(
        model=classifier,
        device=device,
        # float16=True,
        transform=valid_transform,
        apply_softmax=True,
        # apply_argmax=True,
        augmentations=cfg.patcheval.tta_num,
    )

    meta = pd.read_excel(ds_sheet_path, 0, index_col=0)

    vmeta = meta.loc[meta.validation == True]

    # vmeta = vmeta.loc[vmeta.img_num == 136]  # TEST

    if not 'dataset_name' in vmeta.columns:
        # Workaroud until 'dataset_name' is always present in patch meta: Populate from image-level source meta sheet
        vmeta = attach_dataset_name_column(vmeta, src_sheet_path=cfg.sheet_path)


    # dataset_name = ...
    # ds_vmeta = vmeta.loc[vmeta.dataset_name == dataset_name]

    logger.info('\n== Patch selection ==')

    all_targets = []
    all_preds = []

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
        splits = [None] * cfg.patcheval.rdraws  # do random sampling
    else:
        splits = subseq_splits

    # splits = range(min_group_samples)  # iterate over all individuals


    def evaluate(vmeta, groupkey, split=None):
        group_preds = {}
        group_pred_labels = {}
        group_targets = {}
        group_target_labels = {}

        logger.info(f'Grouping by {groupkey}')
        for group in vmeta[groupkey].unique():
            # For each group instance:
            gvmeta = vmeta.loc[vmeta[groupkey] == group]
            assert len(gvmeta.enctype.unique() == 1)
            target_label = gvmeta.iloc[0].enctype
            target = utils.CLASS_IDS[target_label]

            logger.info(f'Group {group} yields {gvmeta.shape[0]} patches.')

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
                logger.info(f'-> After reducing to a maximum of {MAX_SAMPLES_PER_GROUP}, we now have:')
                logger.info(f'Group {group} yields {gvmeta.shape[0]} patches.')


            preds = []
            targets = []
            pred_labels = []
            target_labels = []
            for patch_entry in gvmeta.itertuples():
                raw_fname = patch_entry.patch_fname
                nobg_fpath = patches_root / 'nobg' / raw_fname.replace('raw', 'nobg')
                patch = iio.imread(nobg_fpath).astype(np.float32)[None][None]

                out = predictor.predict(patch)
                pred = out[0].argmax(0).item()

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

        logger.info('\n\n==  Patch classification ==\n')
        for group in group_preds.keys():
            logger.info(f'Group {group}\nTrue class: {group_target_labels[group]}\nPredicted classes: {group_pred_labels[group]}\n-> Majority vote result: {group_majority_pred_names[group]}')


        if False:  # Sanity check: Calculate confusion matrix entries myself
            for a in range(2, 8):
                for b in range(2, 8):
                    v = np.sum((targets == a) & (preds == b))
                    print(f'T: {utils.CLASS_NAMES[a]}, P: {utils.CLASS_NAMES[b]} -> {v}')

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
    plt.savefig(f'{patches_root}/patch_confusion_matrix_n{repr_max_samples}.pdf', bbox_inches='tight')

    # TODO: Save predictions

    # predictions = pd.DataFrame.from_dict(img_majority_preds, orient='index', columns=['class', 'confidence'])
    # predictions = predictions.sort_index().convert_dtypes()
    # predictions.to_excel(f'{patches_root}/samples_nnpredictions.xlsx', index_label='patch_id', float_format='%.2f')


    # import IPython ; IPython.embed(); raise SystemExit

if __name__ == '__main__':
    main()