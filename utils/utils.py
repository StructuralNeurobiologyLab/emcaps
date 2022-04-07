"""
Utility functions and resources.
"""

# TODO: Eliminate duplicated code elsewhere by importing from here.

import os
from os.path import expanduser as eu
from pathlib import Path

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


def image_grid(imgs, rows, cols, enable_grid_lines=True, text_color=255):
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

OLD_CLASS_IDS = class_info['class_ids']  # Use v5 class names
OLD_CLASS_NAMES = {v: k for k, v in OLD_CLASS_IDS.items()}
OLDNAMES_TO_V5NAMES = class_info['_oldnames_to_v5names']



if os.getenv('CLUSTER') == 'WHOLEBRAIN':
    path_prefix = Path('/wholebrain/scratch/mdraw/tum/').expanduser()
else:
    path_prefix = Path('~/tum/').expanduser()

sheet = pd.read_excel(
    path_prefix / 'Single-table_database/Image_annotation_for_ML_single_table.xlsx',
    sheet_name='all_metadata'
)

meta = sheet.copy()
meta = meta.rename(columns={' Image': 'num'})
meta = meta.loc[meta['num'] >= 16]
meta = meta.rename(columns={'Short experimental condition': 'scond'})
meta = meta.convert_dtypes()

# Use new short names
OLDNAMES_TO_V5NAMES = class_info['_oldnames_to_v5names']
# meta.scond.replace(OLDNAMES_TO_V5NAMES, inplace=True)

# Load train/validation split
valid_split_path = './valid_split.yaml'
with open(valid_split_path) as f:
    valid_image_dict = yaml.load(f, Loader=yaml.FullLoader)


# Also works for numbers
def get_meta_row(path):
    if not isinstance(path, Path):
        path = Path(str(path))
    row = meta[meta['num'] == int(path.stem)]
    return row

def get_old_enctype(path):
    row = get_meta_row(path)
    old_enctype = row.scond.item()
    assert old_enctype in OLD_CLASS_NAMES.values(), f'{old_enctype} not in {OLD_CLASS_NAMES.values()}'
    return old_enctype

def get_v5_enctype(path):
    old_enctype = get_old_enctype(path)
    v5_enctype = OLDNAMES_TO_V5NAMES[old_enctype]
    assert v5_enctype in CLASS_NAMES.values(), f'{v5_enctype} not in {CLASS_NAMES.values()}'
    return v5_enctype

def is_for_validation(path):
    row = get_meta_row(path)
    return row.Validation.item()
