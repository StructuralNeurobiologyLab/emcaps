"""Image-based dataset splitting into train + validation parts."""

import logging
import os
from os.path import expanduser as eu
from pathlib import Path
from typing import Tuple, List

import tqdm
import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage import measure

from emcaps.utils import get_meta, get_path_prefix, get_image_resources


# Load original image sheet

path_prefix = get_path_prefix()
data_root = path_prefix / 'Single-table_database'
# Image based split
isplit_data_root = data_root / 'isplitdata_v8'
sheet_path = data_root / 'Image_annotation_for_ML_single_table.xlsx'
isplit_data_root.mkdir(exist_ok=True)

meta = get_meta(sheet_path=sheet_path, v5names=True)


# Set up logging
logger = logging.getLogger('splitdataset')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{isplit_data_root}/splitdataset.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def count_ccs(lab):
    """Count connected components in label image"""
    _, ncc = measure.label(lab, return_num=True)
    return ncc

def get_best_split_slices(lab: np.ndarray, valid_ratio: float = 0.3) -> Tuple[Tuple[slice, slice], Tuple[slice, slice]]:
    """Determine best slices for splitting the image by considering particle ratios"""
    sh = np.array(lab.shape)
    vert_border, horiz_border = np.round(sh * valid_ratio).astype(np.int64)
    vert_slices = (
        (slice(0, vert_border), slice(0, None)),
        (slice(vert_border, None), slice(0, None)),
    )
    hori_slices = (
        (slice(0, None), slice(0, horiz_border)),
        (slice(0, None), slice(horiz_border, None)),
    )

    # Count number of particles in each subimage
    val_vert_nc = count_ccs(lab[vert_slices[0]])
    trn_vert_nc = count_ccs(lab[vert_slices[1]])
    val_hori_nc = count_ccs(lab[hori_slices[0]])
    trn_hori_nc = count_ccs(lab[hori_slices[1]])
    vert_split_nc_ratio = val_vert_nc / (val_vert_nc + trn_vert_nc)
    hori_split_nc_ratio = val_hori_nc / (val_hori_nc + trn_hori_nc)
    logger.info(f'vsr: {vert_split_nc_ratio:.2f}, hsr: {hori_split_nc_ratio:.2f}')

    # Choose the split that minimizes difference between particle ratio and split ratio
    vert_split_penalty = abs(vert_split_nc_ratio - valid_ratio)
    hori_split_penalty = abs(hori_split_nc_ratio - valid_ratio)
    if vert_split_penalty <= hori_split_penalty:
        best_slices = vert_slices
    else:
        best_slices = hori_slices

    return best_slices


def split_by_slices(img: np.ndarray, slices: Tuple[Tuple[slice, slice], Tuple[slice, slice]]) -> Tuple[np.ndarray, np.ndarray]:
    val = img[slices[0]]
    train = img[slices[1]]
    return val, train


def is_excluded(resmeta: pd.Series) -> bool:
    return (not resmeta.Validation) and (not resmeta.Training)


def main():
    for entry in tqdm.tqdm(meta.itertuples(), total=len(meta)):
        img_num = int(entry.num)
        # Load original images and resources
        res = get_image_resources(img_num=img_num, sheet_path=sheet_path, use_curated_if_available=True)
        logger.info(f'Using label source {res.labelpath}{" (curated)" if res.curated else ""}')
        if res.was_inverted:
            logger.info(f'Image {img_num} was re-inverted.')
        if is_excluded(res.metarow):
            logger.info(f'Skipping image {img_num} because it is excluded from ML usage via meta spreadsheet.')
            continue
        if res.label is None:
            logger.info(f'Skipping image {img_num} because no label was found.')
            continue
        if res.raw is None:
            logger.info(f'Skipping image {img_num} because no raw image was found.')
            continue

        # Split images
        split_slices = get_best_split_slices(res.label)
        val_raw, trn_raw = split_by_slices(res.raw, split_slices)
        val_lab, trn_lab = split_by_slices(res.label, split_slices)

        # Scale for image viewer compat
        val_lab = val_lab.astype(np.uint8) * 255
        trn_lab = trn_lab.astype(np.uint8) * 255

        # Save newly split images
        img_subdir = isplit_data_root / str(img_num)
        img_subdir.mkdir(exist_ok=True)
        val_raw_path = img_subdir / f'{img_num}_val.png'
        trn_raw_path = img_subdir / f'{img_num}_trn.png'
        val_lab_path = img_subdir / f'{img_num}_val_encapsulins.png'
        trn_lab_path = img_subdir / f'{img_num}_trn_encapsulins.png'
        iio.imwrite(val_raw_path, val_raw)
        iio.imwrite(trn_raw_path, trn_raw)
        iio.imwrite(val_lab_path, val_lab)
        iio.imwrite(trn_lab_path, trn_lab)


if __name__ == '__main__':
    main()
