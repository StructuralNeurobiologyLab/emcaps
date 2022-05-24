#!/usr/bin/env python3


"""
Produce an average intensity image for each class
"""

import os
import random
from pathlib import Path

import imageio
import yaml
import numpy as np
import pandas as pd

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

patches_root = path_prefix / 'patches_v6e_dr5/'
# patches_root = path_prefix / 'patches_v6d_generalization_dro_dr5/'
avg_output_dir = patches_root / 'avg_patches'
avg_output_dir.mkdir(exist_ok=True)
raw_by_class_dir = patches_root / 'by_class_validation_raw'
raw_by_class_dir.mkdir(exist_ok=True)

full_meta = pd.read_excel(f'{patches_root}/patchmeta_traintest.xlsx', sheet_name='Sheet1', index_col=0)
# full_meta = pd.read_excel(f'{patches_root}/patchmeta.xlsx', sheet_name='Sheet1', index_col=0)

vmeta = full_meta.loc[full_meta.validation == True]
# vmeta = full_meta#.loc[full_meta.validation == True]

def get_enctype_patches(meta, enctype):
    enctypemeta = meta.loc[meta.enctype == enctype]
    patches = []
    for patch_entry in enctypemeta.itertuples():
        raw_fname = patch_entry.patch_fname
        raw_fname = patches_root / 'raw' / raw_fname
        patch = imageio.imread(raw_fname)
        patches.append(patch)
    patches = np.stack(patches)
    return patches


def create_avg_img(imgs):
    return np.mean(imgs, axis=0).astype(imgs.dtype)


for enctype in CLASS_NAMES_IN_USE:
    (raw_by_class_dir / enctype).mkdir(exist_ok=True)

# for enctype in vmeta.enctype.unique():
for enctype in CLASS_NAMES_IN_USE:
    patches = get_enctype_patches(vmeta, enctype=enctype)
    print(f'{enctype}: got {patches.shape[0]} patches.')
    avg_patch = create_avg_img(patches).astype(np.uint8)
    imageio.imwrite(avg_output_dir / f'avg_{enctype}.png', avg_patch)
    for i in range(patches.shape[0]):
        rpath = raw_by_class_dir / enctype / f'rv_{i:04d}.png'
        imageio.imwrite(rpath, patches[i])

