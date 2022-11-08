"""Image-based dataset splitting into train + validation parts."""

import logging
import os
from os.path import expanduser as eu
from pathlib import Path
from typing import Tuple, List, Optional
from enum import Enum, auto

import tqdm
import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage import measure
from omegaconf import DictConfig, OmegaConf
import hydra

from emcaps.utils import get_meta, get_path_prefix, get_image_resources, strip_host_prefix, get_meta_row


# Set up logging
# logger = logging.getLogger(__name__)
logger = logging.getLogger('emcaps-splitdataset')
# logger.setLevel(logging.DEBUG)
# fh = logging.FileHandler(f'{isplit_data_root}/splitdataset.log')
# fh.setLevel(logging.DEBUG)
# logger.addHandler(fh)


# Split definition constants
class Split(Enum):
    vertical_tr_val = auto()
    vertical_val_tr = auto()
    horizontal_tr_val = auto()
    horizontal_val_tr = auto()


def get_split_override(img_num: int, sheet_path) -> Optional[Split]:
    row = get_meta_row(img_num, sheet_path=sheet_path)
    dscr = row.get('Image Split')
    if pd.isna(dscr):
        return None
    return Split[dscr]


def count_ccs(lab):
    """Count connected components in label image"""
    _, ncc = measure.label(lab, return_num=True)
    return ncc


def get_slices(sh: np.ndarray, valid_ratio: float = 0.3, from_left=True):
    if from_left:
        relative_split_border = valid_ratio
    else:
        relative_split_border = 1 - valid_ratio
    vert_border, hori_border = np.round(sh * relative_split_border).astype(np.int64)
    hori_slices = (
        (slice(0, hori_border), slice(0, None)),
        (slice(hori_border, None), slice(0, None)),
    )
    vert_slices = (
        (slice(0, None), slice(0, vert_border)),
        (slice(0, None), slice(vert_border, None)),
    )
    if not from_left:  # Swap slices -> Preserve split ratio, but change direction from which to split
        vert_slices = vert_slices[::-1]
        hori_slices = hori_slices[::-1]
    return vert_slices, hori_slices


def get_best_split_slices(lab: np.ndarray, img_num: int, sheet_path, valid_ratio: float = 0.3) -> Tuple[Tuple[slice, slice], Tuple[slice, slice]]:
    """Determine best slices for splitting the image by considering particle ratios"""
    sh = np.array(lab.shape)
    vert_slices, hori_slices = get_slices(sh, valid_ratio=valid_ratio, from_left=True)
    right_vert_slices, right_hori_slices = get_slices(sh, valid_ratio=valid_ratio, from_left=False)

    split_override = get_split_override(img_num=img_num, sheet_path=sheet_path)

    if split_override is not None:
        # Override found, directly return determined split
        if split_override == Split.horizontal_val_tr:
            best_slices = hori_slices
        elif split_override == Split.vertical_val_tr:
            best_slices = vert_slices
        elif split_override == Split.horizontal_tr_val:
            best_slices = right_hori_slices
        elif split_override == Split.vertical_tr_val:
            best_slices = right_vert_slices
        else:
            raise ValueError(f'Invalid choice {split_override}')

        logger.info(f'Override split: {img_num}.')
        return best_slices

    # Else, if no override found, determine optimal splitting automatically:

    # Count number of particles in each subimage
    val_vert_nc = count_ccs(lab[vert_slices[0]])
    trn_vert_nc = count_ccs(lab[vert_slices[1]])
    val_hori_nc = count_ccs(lab[hori_slices[0]])
    trn_hori_nc = count_ccs(lab[hori_slices[1]])
    _eps = 1e-6  # Avoid zero division if no ccs were found
    vert_split_nc_ratio = val_vert_nc / (val_vert_nc + trn_vert_nc + _eps)
    hori_split_nc_ratio = val_hori_nc / (val_hori_nc + trn_hori_nc + _eps)
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
    # 'all' contains every usable image. All other groups are subsets, so they can be selected in later steps.
    # Could be made more explicit as `not (resmeta['all'] or resmeta['all2'] or ...)`
    return not resmeta['all']


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg: DictConfig) -> None:

    ONLY_TM = False
    NO_TM = False

    STRIP_HOST_PREFIX = True  # If True, strip 'DRO-' and 'MICE_' prefixes from output file names

    if ONLY_TM:
        logger.info('ONLY_TM mode active. Only 1M-Tm-labeled encapsulins are considered, everything else is ignored or regarded as background.\n\n')


    # path_prefix = get_path_prefix()
    # data_root = path_prefix / 'emcapsulin'
    # data_root = cfg.data_root
    # Image based split
    # isplit_data_root = data_root / 'isplitdata_v15'
    output_path = Path(cfg.isplit_data_path).expanduser()
    if ONLY_TM:
        output_path = output_path.with_name(f'{output_path.name}_onlytm')
    if NO_TM:
        output_path = output_path.with_name(f'{output_path.name}_notm')

    # sheet_path = data_root / 'emcapsulin_data.xlsx'
    sheet_path = Path(cfg.sheet_path).expanduser()
    output_path.mkdir(exist_ok=True)

    meta = get_meta(sheet_path=sheet_path)


    for entry in tqdm.tqdm(meta.itertuples(), total=len(meta)):
        img_num = int(entry.num)
        # Load original images and resources
        res = get_image_resources(img_num=img_num, sheet_path=sheet_path, only_tm=ONLY_TM)
        if res is None:
            logger.info(f'Skipping image {img_num} because of an image resource condition.')
            continue
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
        split_slices = get_best_split_slices(res.label, img_num=int(res.metarow.num), sheet_path=cfg.sheet_path)
        val_raw, trn_raw = split_by_slices(res.raw, split_slices)
        val_lab, trn_lab = split_by_slices(res.label, split_slices)

        # Scale for image viewer compat
        val_lab = val_lab.astype(np.uint8) * 255
        trn_lab = trn_lab.astype(np.uint8) * 255

        # Save newly split images
        img_subdir = output_path / str(img_num)
        img_subdir.mkdir(exist_ok=True)
        val_raw_path = img_subdir / f'{img_num}_val.png'
        trn_raw_path = img_subdir / f'{img_num}_trn.png'
        val_lab_path = img_subdir / f'{img_num}_val_encapsulins.png'
        trn_lab_path = img_subdir / f'{img_num}_trn_encapsulins.png'
        iio.imwrite(val_raw_path, val_raw)
        iio.imwrite(trn_raw_path, trn_raw)
        iio.imwrite(val_lab_path, val_lab)
        iio.imwrite(trn_lab_path, trn_lab)

        # Handle multiple individual per-enctype labels there are any
        if res.per_enctype_labels:
            for scond, elab in res.per_enctype_labels.items():
                if STRIP_HOST_PREFIX:
                    scond = strip_host_prefix(scond)
                logger.info(f'Splitting per-enctype labels for image {img_num}: {scond}')
                val_elab, trn_elab = split_by_slices(elab, split_slices)
                val_elab = val_elab.astype(np.uint8) * 255
                trn_elab = trn_elab.astype(np.uint8) * 255
                val_elab_path = img_subdir / f'{img_num}_val_elab_{scond}.png'
                trn_elab_path = img_subdir / f'{img_num}_trn_elab_{scond}.png'
                iio.imwrite(val_elab_path, val_elab)
                iio.imwrite(trn_elab_path, trn_elab)

        # Handle region masks if there are any
        if res.regmasks:
            for scond, mask in res.regmasks.items():
                logger.info(f'Splitting region masks for image {img_num}: {scond}')
                val_mask, trn_mask = split_by_slices(mask, split_slices)
                val_mask = val_mask.astype(np.uint8) * 255
                trn_mask = trn_mask.astype(np.uint8) * 255
                val_mask_path = img_subdir / f'{img_num}_val_mask_{scond}.png'
                trn_mask_path = img_subdir / f'{img_num}_trn_mask_{scond}.png'
                iio.imwrite(val_mask_path, val_mask)
                iio.imwrite(trn_mask_path, trn_mask)

if __name__ == '__main__':
    main()
