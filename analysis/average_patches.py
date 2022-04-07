#!/usr/bin/env python3


"""
Produce an average intensity image for each class
"""

import os
import random
from typing import Literal
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
from skimage.filters import sobel

# Set up all RNG seeds, set level of determinism
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)


valid_split_path = './class_info.yaml'
with open(valid_split_path) as f:
    CLASS_INFO = yaml.load(f, Loader=yaml.FullLoader)

CLASS_IDS = CLASS_INFO['class_ids_v5']
CLASS_NAMES = list(CLASS_IDS.keys())
CLASS_NAMES_IN_USE = CLASS_NAMES[2:]


# USER PATHS

if os.getenv('CLUSTER') == 'WHOLEBRAIN':
    path_prefix = Path('/wholebrain/scratch/mdraw/tum/').expanduser()
else:
    path_prefix = Path('~/tum/').expanduser()

patches_root = path_prefix / 'patches_v5/'
avg_output_dir = patches_root / 'avg_patches'
avg_output_dir.mkdir(exist_ok=True)
raw_by_class_dir = patches_root / 'by_class_validation_raw'
raw_by_class_dir.mkdir(exist_ok=True)
sobel_by_class_dir = patches_root / 'by_class_validation_sobel'
sobel_by_class_dir.mkdir(exist_ok=True)

full_meta = pd.read_excel(f'{patches_root}/patchmeta_traintest.xlsx', sheet_name='Sheet1', index_col=0)
# full_meta = pd.read_excel(f'{patches_root}/patchmeta.xlsx', sheet_name='Sheet1', index_col=0)

vmeta = full_meta.loc[full_meta.validation == True]
# vmeta = full_meta#.loc[full_meta.validation == True]

def get_enctype_patches(meta, enctype, use_sobel=False):
    enctypemeta = meta.loc[meta.enctype == enctype]
    patches = []
    for patch_entry in enctypemeta.itertuples():
        raw_fname = patch_entry.patch_fname
        raw_fname = patches_root / 'raw' / raw_fname
        patch = imageio.imread(raw_fname)
        if use_sobel:
            patch = sobel(patch)
        patches.append(patch)
    patches = np.stack(patches)
    return patches


def create_avg_img(imgs):
    return np.mean(imgs, axis=0).astype(imgs.dtype)


for enctype in CLASS_NAMES_IN_USE:
    (raw_by_class_dir / enctype).mkdir(exist_ok=True)
    (sobel_by_class_dir / enctype).mkdir(exist_ok=True)

# for enctype in vmeta.enctype.unique():
for enctype in CLASS_NAMES_IN_USE:
    patches = get_enctype_patches(vmeta, enctype=enctype)
    print(f'{enctype}: got {patches.shape[0]} patches.')
    avg_patch = create_avg_img(patches).astype(np.uint8)
    sobel_patches = get_enctype_patches(vmeta, enctype=enctype, use_sobel=True)
    sobel_avg_patch = create_avg_img(sobel_patches)
    sobel_avg_patch = (sobel_avg_patch * 255).astype(np.uint8)
    imageio.imwrite(avg_output_dir / f'avg_{enctype}.png', avg_patch)
    imageio.imwrite(avg_output_dir / f'sobel_avg_{enctype}.png', sobel_avg_patch)
    for i in range(patches.shape[0]):
        rpath = raw_by_class_dir / enctype / f'rv_{i:03d}.png'
        spath = sobel_by_class_dir / enctype / f'sv_{i:03d}.png'
        imageio.imwrite(rpath, patches[i])
        imageio.imwrite(spath, (sobel_patches[i] * 255).astype(np.uint8))

