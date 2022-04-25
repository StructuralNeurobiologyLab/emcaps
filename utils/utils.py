"""
Utility functions and resources.
"""

# TODO: Eliminate duplicated code elsewhere by importing from here.

import os
from typing import Optional, Union
from os.path import expanduser as eu
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import imageio
import numpy as np
import pandas as pd
import skimage
import tqdm
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure
from skimage import morphology as sm



def eul(paths):
    """Shortcut for expanding all user paths in a list"""
    return [os.path.expanduser(p) for p in paths]


def image_grid(imgs, rows, cols, enable_grid_lines=True, text_color=255) -> Image.Image:
    """Draw images sequentially on a (rows * cols) grid."""
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('L', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        drw = ImageDraw.Draw(img)
        drw.text((0, 0), f'{i:02d}', fill=text_color)
        grid.paste(img, box=(i % cols * w, i // cols * h))

    if enable_grid_lines:
        gdrw = ImageDraw.Draw(grid)
        for i in range(1, rows):
            line = ((0, w * i), (h * cols - 1, w * i))
            gdrw.line(line, fill=128)
        for i in range(1, cols):
            line = ((h * i, 0), (h * i, w * rows - 1))
            gdrw.line(line, fill=128)
    return grid


# Load mapping from class names to class IDs
class_info_path = './class_info.yaml'
with open(class_info_path) as f:
    class_info = yaml.load(f, Loader=yaml.FullLoader)
CLASS_IDS = class_info['class_ids_v5']  # Use v5 class names
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}
LABEL_NAME = class_info['label_name']

OLD_CLASS_IDS = class_info['class_ids']  # Use v5 class names
OLD_CLASS_NAMES = {v: k for k, v in OLD_CLASS_IDS.items()}
OLDNAMES_TO_V5NAMES = class_info['_oldnames_to_v5names']


def get_path_prefix() -> Path:
    if os.getenv('CLUSTER') == 'WHOLEBRAIN':
        path_prefix = Path('/wholebrain/scratch/mdraw/tum/').expanduser()
    else:
        path_prefix = Path('~/tum/').expanduser()
    assert path_prefix.is_dir()
    return path_prefix


@lru_cache(maxsize=1024)
def get_meta(sheet_path=None, sheet_name=0, v5names=True) -> pd.DataFrame:
    if sheet_path is None:
        path_prefix = get_path_prefix()
        sheet_path = path_prefix / 'Single-table_database/Image_annotation_for_ML_single_table.xlsx'
        sheet_name = 'all_metadata'
    sheet = pd.read_excel(sheet_path, sheet_name=sheet_name)
    meta = sheet.copy()
    meta = meta.rename(columns={' Image': 'num'})
    meta = meta.loc[meta['num'] >= 16]
    meta = meta.rename(columns={'Short experimental condition': 'scond'})
    meta = meta.convert_dtypes()
    if v5names:
        meta.scond.replace(OLDNAMES_TO_V5NAMES, inplace=True)
    return meta


# Use new short names
OLDNAMES_TO_V5NAMES = class_info['_oldnames_to_v5names']
# meta.scond.replace(OLDNAMES_TO_V5NAMES, inplace=True)

# # Load train/validation split
# valid_split_path = './valid_split.yaml'
# with open(valid_split_path) as f:
#     valid_image_dict = yaml.load(f, Loader=yaml.FullLoader)


# Also works for numbers
@lru_cache(maxsize=1024)
def get_meta_row(path_or_num, *args, **kwargs) -> pd.Series:
    meta = get_meta(*args, **kwargs)
    if not isinstance(path_or_num, Path):
        path = Path(str(path_or_num))
    row = meta.loc[meta['num'] == int(path.stem)]
    assert row.shape[0] == 1  # num is unique
    row = row.squeeze(0) #  -> to pd.Series
    return row


@lru_cache(maxsize=1024)
def get_old_enctype(path) -> str:
    row = get_meta_row(path)
    old_enctype = row.scond.item()
    assert old_enctype in OLD_CLASS_NAMES.values(), f'{old_enctype} not in {OLD_CLASS_NAMES.values()}'
    return old_enctype


@lru_cache(maxsize=1024)
def get_v5_enctype(path) -> str:
    old_enctype = get_old_enctype(path)
    v5_enctype = OLDNAMES_TO_V5NAMES[old_enctype]
    assert v5_enctype in CLASS_NAMES.values(), f'{v5_enctype} not in {CLASS_NAMES.values()}'
    return v5_enctype


@lru_cache(maxsize=1024)
def is_for_validation(path) -> bool:
    row = get_meta_row(path)
    return row.Validation.item()


@lru_cache(maxsize=1024)
def get_raw_path(img_num: int, sheet_path=None) -> Path:
    if sheet_path is None:
        path_prefix = get_path_prefix()
        sheet_path = path_prefix / 'Single-table_database/Image_annotation_for_ML_single_table.xlsx'
    # meta = get_meta(sheet_path=sheet_path)
    subdir_path = sheet_path.parent / f'{img_num}'
    img_path = subdir_path / f'{img_num}.tif'
    return img_path

@lru_cache(maxsize=1024)
def get_raw(img_num: int, sheet_path=None) -> np.ndarray:
    img_path = get_raw_path(img_num=img_num, sheet_path=sheet_path)
    img = imageio.imread(img_path)
    return img


@lru_cache(maxsize=1024)
def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        uppercase_suf_path = path.with_suffix('.TIF')
        if uppercase_suf_path.is_file():
            path = uppercase_suf_path
    img = imageio.imread(path)
    return img


def ensure_not_inverted(lab: np.ndarray, threshold: float = 0.5, verbose=True) -> np.ndarray:
    """Heuristic to ensure that the label is not inverted.
    
    It is not plausible that there is more foreground than background in a label image."""
    if np.any(lab < 0) or np.any(lab > 1):
        raise ValueError('Labels must be in the range [0, 1] (binary).')
    if lab.mean().item() > 0.5:
        if verbose:
            print('ensure_not_inverted: re-inverting labels')
        lab = ~lab
    return lab


@dataclass
class ImageResources:
    """Image resources, holds all necessary data and metadata of one source image including labels"""
    metarow: pd.Series
    raw: Optional[np.ndarray] = None
    label: Optional[np.ndarray] = None
    roimask: Optional[np.ndarray] = None
    rawpath: Optional[Path] = None
    curated: bool = False


def get_image_resources(img_num, sheet_path=None, use_curated_if_available=True):
    metarow = get_meta_row(path_or_num=img_num, sheet_path=sheet_path)
    raw_path = get_raw_path(img_num=img_num, sheet_path=sheet_path)
    if raw_path.is_file():
        raw = imageio.imread(raw_path)
    else:
        raw = None
    # with_stem() requires py39 -> use with_name()
    label_path = raw_path.with_name(f'{raw_path.stem}_{LABEL_NAME}{raw_path.suffix}')
    # Look for improved labels in "curated" subdir. If nothing is found, use regular label file.
    is_curated = False
    if use_curated_if_available:
        for curated_dir_candidate in ['Curated', 'curated']:  # Sometimes lowercase
            candidate = label_path.parent / curated_dir_candidate / label_path.name
            if candidate.is_file():
                label_path = candidate  # Update label_path
                is_curated = True
                break  # Found it, stop searching
    if label_path.is_file():
        label = imageio.imread(label_path)
        label = label > 0  # Binarize
        label = ensure_not_inverted(label)
    else:
        label = None

    roimask = None  # TODO. Not implemented yet 

    imgres = ImageResources(
        raw=raw,
        label=label,
        roimask=roimask,
        metarow=metarow,
        rawpath=raw_path,
        curated=is_curated,
    )

    return imgres

