#!/usr/bin/env python3


"""
Produce an average intensity image for each class
"""

import random
import logging
from pathlib import Path

import imageio.v3 as iio
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from emcaps import utils


def get_enctype_patches(meta, enctype, patches_root, max_samples=None):
    enctypemeta = meta.loc[meta.enctype == enctype]
    patches = []
    for patch_entry in enctypemeta.itertuples():
        raw_fname = patch_entry.patch_fname
        raw_fname = patches_root / 'raw' / raw_fname
        patch = iio.imread(raw_fname)
        patches.append(patch)
    if max_samples is not None and len(patches) > max_samples:
        patches = random.sample(patches, max_samples)
    return patches


def get_enctype_patches_by_img(meta, enctype, patches_root, max_samples=None):
    enctypemeta = meta.loc[meta.enctype == enctype]
    patches = {num: [] for num in enctypemeta.img_num.unique()}
    for patch_entry in enctypemeta.itertuples():
        raw_fname = patch_entry.patch_fname
        raw_fname = patches_root / 'raw' / raw_fname
        patch = iio.imread(raw_fname)
        patches[patch_entry.img_num].append(patch)
    for num in patches.keys():
        npatches = patches[num]
        if max_samples is not None and len(npatches) > max_samples:
            npatches = random.sample(npatches, max_samples)
        patches[num] = np.stack(npatches)
    return patches


def create_avg_img(imgs):
    return np.mean(imgs, axis=0).astype(imgs.dtype)


@hydra.main(version_base='1.2', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    random_seed = cfg.averagepatches.seed
    np.random.seed(random_seed)
    random.seed(random_seed)

    max_num_patches_per_avg = cfg.averagepatches.max_num_patches_per_avg
    ds_sheet_path = Path(cfg.averagepatches.patch_ds_sheet)
    patches_root = ds_sheet_path.parent

    logger = logging.getLogger('emcaps-averagepatches')
    fh = logging.FileHandler(patches_root / 'averagepatches.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    patch_meta = pd.read_excel(f'{patches_root}/patchmeta.xlsx', sheet_name=0, index_col=0)
    if not 'dataset_name' in patch_meta.columns:
        # Workaroud until 'dataset_name' is always present in patch meta: Populate from image-level source meta sheet
        patch_meta = utils.attach_dataset_name_column(patch_meta, src_sheet_path=cfg.sheet_path)
    
    dataset_name = cfg.averagepatches.dataset_name
    if dataset_name is not None:
        patch_meta = patch_meta.loc[patch_meta.dataset_name == dataset_name]
    path_suffix = '' if dataset_name is None else f'_{dataset_name}'

    avg_output_dir = patches_root / f'avg_patches{path_suffix}'
    avg_output_dir.mkdir(exist_ok=True)
    raw_by_enctype_dir = patches_root / f'by_enctype_raw{path_suffix}'
    raw_by_enctype_dir.mkdir(exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg.averagepatches, resolve=True))
    logger.info(f'Writing to {avg_output_dir} and {raw_by_enctype_dir}')

    if cfg.averagepatches.exclude_train_data:
        patch_meta = patch_meta.loc[~patch_meta.train]
    if cfg.averagepatches.exclude_validation_data:
        patch_meta = patch_meta.loc[~patch_meta.validation]

    all_enctypes = patch_meta.enctype.unique().tolist()

    for enctype in all_enctypes:
        (raw_by_enctype_dir / enctype).mkdir(exist_ok=True)

    for enctype in all_enctypes:
        if cfg.averagepatches.by_img:
            patches = get_enctype_patches_by_img(patch_meta, enctype=enctype, patches_root=patches_root, max_samples=max_num_patches_per_avg)
            # logger.info(f'{enctype}: got {patches.shape[0]} patches.')
            for num, npa in patches.items():
                avg_patch = create_avg_img(npa).astype(np.uint8)
                iio.imwrite(avg_output_dir / f'avg-{enctype}_{num}_n{len(npa)}.png', avg_patch)
        else:
            patches = get_enctype_patches(patch_meta, enctype=enctype, patches_root=patches_root, max_samples=max_num_patches_per_avg)
            logger.info(f'{enctype}: {len(patches)} patches selected.')
            avg_patch = create_avg_img(np.stack(patches)).astype(np.uint8)
            iio.imwrite(avg_output_dir / f'avg_{enctype}.png', avg_patch)
            # Write out patches by enctypes (not averages)
            for i in range(len(patches)):
                rpath = raw_by_enctype_dir / enctype / f'r_{i:04d}.png'
                iio.imwrite(rpath, patches[i])

if __name__ == '__main__':
    main()