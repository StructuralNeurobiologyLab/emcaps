"""
Utility functions and resources.
"""

# TODO: Eliminate duplicated code elsewhere by importing from here.

import os
import platform
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import imageio.v3 as iio
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from skimage.color import label2rgb

import logging

logger = logging.getLogger('emcaps-utils')


TMPPATH = '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()


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
class_info_path = './emcaps/class_info.yaml'  # TODO: proper relative path
with open(class_info_path) as f:
    class_info = yaml.load(f, Loader=yaml.FullLoader)
CLASS_IDS = class_info['class_ids_v5']  # Use v5 class names
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}
LABEL_NAME = class_info['label_name']
CLASS_GROUPS = class_info['class_groups']

OLD_CLASS_IDS = class_info['class_ids']  # Use v5 class names
OLD_CLASS_NAMES = {v: k for k, v in OLD_CLASS_IDS.items()}
OLDNAMES_TO_V5NAMES = class_info['_oldnames_to_v5names']
V5NAMES_TO_OLDNAMES = {v: k for k, v in OLDNAMES_TO_V5NAMES.items()}


def render_skimage_overlay(img: np.ndarray, lab: np.ndarray, bg_label=0, alpha=0.5, **label2rgb_kwargs) -> np.ndarray:
    ov = label2rgb(label=lab, image=img, bg_label=bg_label, alpha=alpha, **label2rgb_kwargs)
    # Redraw raw image onto overlays where they were blended with 0, to restore original brightness
    img01 = img.astype(np.float64) / 255.
    ov[lab == 0, :] = img01[lab == 0, None]
    # Convert from [0, 1] float to [0, 255] uint8 for imageio
    ov = (ov * 255.).astype(np.uint8)
    return ov


def get_path_prefix() -> Path:
    if os.getenv('CLUSTER') in ['WHOLEBRAIN', 'CAJAL']:
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
    meta['scondv5'] = meta.scond.replace(OLDNAMES_TO_V5NAMES)
    return meta


# Use new short names
OLDNAMES_TO_V5NAMES = class_info['_oldnames_to_v5names']
# meta.scond.replace(OLDNAMES_TO_V5NAMES, inplace=True)


def clean_int(text: str) -> int:
    cleaned = ''.join([c for c in text if c.isdigit()])
    return int(cleaned)


# Also works for numbers
@lru_cache(maxsize=1024)
def get_meta_row(path_or_num, *args, **kwargs) -> pd.Series:
    meta = get_meta(*args, **kwargs)
    if isinstance(path_or_num, Path):
        path_or_num = path_or_num.stem
    if isinstance(path_or_num, str):
        dirty_img_num = str(path_or_num)  # can contain other non-digit characters (e.g. "_val")
        img_num = clean_int(dirty_img_num)  # only retain digits and convert to int
    elif isinstance(path_or_num, int):
        img_num = path_or_num
    else:
        raise TypeError(f'{path_or_num} has unhandled type {type(path_or_num)}.')
    row = meta.loc[meta.num == img_num]
    assert row.shape[0] == 1  # num is unique
    row = row.squeeze(0) #  -> to pd.Series
    return row


@lru_cache(maxsize=1024)
def get_old_enctype(path) -> str:
    row = get_meta_row(path)
    # old_enctype = row.scond.item()
    old_enctype = row.scond
    assert old_enctype in OLD_CLASS_NAMES.values(), f'{old_enctype} not in {OLD_CLASS_NAMES.values()}'
    return old_enctype


@lru_cache(maxsize=1024)
def get_v5_enctype(path) -> str:
    # old_enctype = get_old_enctype(path)
    # v5_enctype = OLDNAMES_TO_V5NAMES[old_enctype]
    row = get_meta_row(path)
    # old_enctype = row.scond.item()
    v5_enctype = row.scondv5
    assert v5_enctype in CLASS_NAMES.values(), f'{v5_enctype} not in {CLASS_NAMES.values()}'
    return v5_enctype


# @lru_cache(maxsize=1024)
# def get_isplit_enctype(path, pos: Optional[Tuple[int]] = None, isplitdata_root=None, role=None) -> str:
#     row = get_meta_row(path)
#     if pos is None:
#         v5_enctype = row.scondv5
#     else:
#         assert isplitdata_root is not None
#         assert role is not None
#         region_masks = get_isplit_multiclass_regions(img_num=row.num, isplitdata_root=isplitdata_root)
#         rmasks = region_masks[role]
#         if not rmasks:  # No rmasks found -> expect enctype to be known simply from table name
#             v5_enctype = row.scondv5
#         else:
#             scond_at_pos = '?'
#             for scond, rmask in rmasks.items():
#                 if rmask[pos] > 0:
#                     scond_at_pos = scond
#                     break
#             v5_enctype = scond_at_pos

#     if 'G_1M-Tm' in v5_enctype:
#         print('FAIL: Deduced dual v5_enctype for one patch pos')
#         # import IPython ; IPython.embed(); raise SystemExit

#     assert v5_enctype in CLASS_NAMES.values(), f'{v5_enctype} not in {CLASS_NAMES.values()}'
#     return v5_enctype


@lru_cache(maxsize=1024)
def get_isplit_enctype(path, pos: Optional[Tuple[int]] = None, isplitdata_root=None, role=None) -> str:
    row = get_meta_row(path)
    v5_enctype = '?'
    if pos is None:  # If no pos is supplied, just one type is expected (position-independent)
        v5_enctype = row.scondv5
    else:  # Find specific enctype of encapsulin at position pos
        assert isplitdata_root is not None
        assert role is not None
        # Prefer region masks because they cover more area
        elabs = get_isplit_per_enctype_regmasks(img_num=row.num, isplitdata_root=isplitdata_root)[role]
        if not elabs:  # No regmask found -> Look for enctype-specific label map instead
            elabs = get_isplit_per_enctype_labels(img_num=row.num, isplitdata_root=isplitdata_root)[role]
            if not elabs:  # Still nothing found -> expect enctype to be known simply from table name
                # return row.scondv5
                pass
        else:  # Masks or labels found -> Get enctype at pos
            # enctype_at_pos = row.scondv5
            enctype_at_pos = '?'
            for enctype, elab in elabs.items():
                if elab[pos] > 0:
                    enctype_at_pos = enctype
                    break
            v5_enctype = enctype_at_pos

    # if 'G_1M-Tm' in v5_enctype or 'and' in v5_enctype:
    #     logger.error(f'FAIL: Deduced dual v5_enctype for one patch pos ({path=}, {pos=}, {isplitdata_root=}, {role=}')
    # assert v5_enctype in CLASS_NAMES.values(), f'{v5_enctype} not in {CLASS_NAMES.values()}'
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
    img = iio.imread(img_path)
    return img


@lru_cache(maxsize=1024)
def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        uppercase_suf_path = path.with_suffix('.TIF')
        if uppercase_suf_path.is_file():
            path = uppercase_suf_path
    img = iio.imread(path)
    return img


def ensure_not_inverted(lab: np.ndarray, threshold: float = 0.5, verbose=True, error=False) -> Tuple[np.ndarray, bool]:
    """Heuristic to ensure that the label is not inverted.
    
    It is not plausible that there is more foreground than background in a label image."""
    if np.any(lab < 0) or np.any(lab > 1):
        raise ValueError('Labels must be in the range [0, 1] (binary).')
    mean = lab.mean().item()
    was_inverted = False
    if mean > threshold:
        if error:
            raise ValueError(f'Binary label has unplausibly high mean {mean:.2f}. Please check if it is inverted.')
        if verbose:
            logger.info('ensure_not_inverted: re-inverting labels')
        lab = ~lab
    return lab, was_inverted


@dataclass
class ImageResources:
    """Image resources, holds all necessary data and metadata of one source image including labels"""
    metarow: pd.Series
    raw: Optional[np.ndarray] = None
    label: Optional[np.ndarray] = None
    per_enctype_labels: Optional[Dict[str, np.ndarray]] = None
    roimask: Optional[np.ndarray] = None
    regmasks: Optional[Dict[str, np.ndarray]] = None
    rawpath: Optional[Path] = None
    labelpath: Optional[Path] = None
    curated: bool = False
    was_inverted: bool = False
    enctypes_present: Optional[Sequence[str]] = None
    host: Optional[str] = None


# @lru_cache(maxsize=1024)
# def get_isplit_multiclass_regions(img_num: int, isplitdata_root: Path):
#     region_masks = {'trn': {}, 'val': {}}
#     ipath = isplitdata_root / f'{img_num}'
#     for role in ['trn', 'val']:
#         for scond in CLASS_NAMES.values():
#             rpath = ipath / f'{img_num}_{role}_mask_{scond}.png'
#             if rpath.is_file():
#                 rmask = iio.imread(rpath) > 0
#                 region_masks[role][scond] = rmask
#     return region_masks

@lru_cache(maxsize=1024)
def get_isplit_per_enctype_labels(img_num: int, isplitdata_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    elabs = {'trn': {}, 'val': {}}
    ipath = isplitdata_root / f'{img_num}'
    for role in ['trn', 'val']:
        for scond in CLASS_NAMES.values():
            epath = ipath / f'{img_num}_{role}_elab_{scond}.png'
            if epath.is_file():
                rmask = iio.imread(epath) > 0
                elabs[role][scond] = rmask
    return elabs


@lru_cache(maxsize=1024)
def get_isplit_per_enctype_regmasks(img_num: int, isplitdata_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    rmasks = {'trn': {}, 'val': {}}
    ipath = isplitdata_root / f'{img_num}'
    for role in ['trn', 'val']:
        for scond in CLASS_NAMES.values():
            rpath = ipath / f'{img_num}_{role}_mask_{scond}.png'
            if rpath.is_file():
                rmask = iio.imread(rpath) > 0
                rmasks[role][scond] = rmask
    return rmasks

# @lru_cache(maxsize=1024)
# def get_isplit_per_enctype_regmasks_or_labels(img_num: int, isplitdata_root: Path):
#     regmasks = get_isplit_per_enctype_regmasks(img_num=img_num, isplitdata_root=isplitdata_root)



def strip_host_prefix(enctype: str, host_prefixes=('DRO-', 'MICE_')) -> str:
    """drop DRO, MICE because we want to treat DRO amd MICE the same class as HEK in most cases"""
    for pref in host_prefixes:
        enctype = enctype.replace(pref, '')
    return enctype


def get_image_resources(img_num, sheet_path=None, use_curated_if_available=True, merge_multilabel=True, only_tm=False, no_tm=False):
    metarow = get_meta_row(path_or_num=img_num, sheet_path=sheet_path)

    relevant_class_names = CLASS_NAMES.values()

    enctype = metarow.scondv5
    # enctype = strip_host_prefix(enctype)
    full_scond = metarow.scond

    if only_tm:
        if '1M-Tm' not in enctype:
            return None
        # For later **-**_1M-Tm image filter
        relevant_class_names = [name for name in relevant_class_names if '1M-Tm' in name]

    # Opposite
    if no_tm:
        if '1M-Tm' in enctype:
            return None
        # For later **-**_1M-Tm image filter
        relevant_class_names = [name for name in relevant_class_names if '1M-Tm' not in name]


    raw_path = get_raw_path(img_num=img_num, sheet_path=sheet_path)
    if raw_path.is_file():
        raw = iio.imread(raw_path)
    else:
        raw = None
    label_path = raw_path.with_stem(f'{raw_path.stem}_{LABEL_NAME}')
    # Look for improved labels in "curated" subdir. If nothing is found, use regular label file.
    is_curated = False
    if use_curated_if_available:
        for curated_dir_candidate in ['Curated', 'curated']:  # Sometimes lowercase
            candidate = label_path.parent / curated_dir_candidate / label_path.name
            if candidate.is_file():
                label_path = candidate  # Update label_path
                is_curated = True
                break  # Found it, stop searching
    was_inverted = False
    enctypes_present = []
    label = None
    per_enctype_labels = {}  # Collect individual labels where only one enctype is labeled
    if label_path.is_file():
        label = iio.imread(label_path)
        label = label > 0  # Binarize
        label, was_inverted = ensure_not_inverted(label)
        per_enctype_labels[enctype] = label
        enctypes_present.append(enctype)

    # Handle multiple label files (multiple classes in one source image)
    if merge_multilabel:# and label is None:
        # Look for multiple label files, which have a different naming pattern.
        # Filter the list of potential label file name candidates by their existence as a file.
        for scond in relevant_class_names:
            for stem_pattern in [f'{raw_path.stem}_{scond}', f'{raw_path.stem}_label_enc_{scond}']:
                if (m_path := raw_path.with_stem(stem_pattern)).is_file():
                    m_label = iio.imread(m_path) > 0
                    m_label, m_was_inverted = ensure_not_inverted(m_label)
                    was_inverted = was_inverted or m_was_inverted
                    enctypes_present.append(scond)
                    per_enctype_labels[scond] = m_label
                    if label is None:
                        label = m_label
                    else:
                        label = np.bitwise_or(label, m_label)  # If at least on label is foreground at some location, define that as foreground

    reg_masks = {}
    # Retrieve multiclass masks that define class-regions ("cell"), if they exist.
    for scond in CLASS_NAMES.values():
        for stem_pattern in [f'{raw_path.stem}_label_cell_{scond}']:
            if (reg_path := raw_path.with_stem(stem_pattern)).is_file():
                reg_label = iio.imread(reg_path) > 0
                reg_masks[scond] = reg_label


    roimask = None  # TODO. Not implemented yet 

    full_scond = metarow.scond
    host = 'hek'  # default to HEK
    if 'MICE' in full_scond:
        host = 'mice'
    elif 'DRO' in full_scond:
        host = 'dro'

    imgres = ImageResources(
        raw=raw,
        label=label,
        per_enctype_labels=per_enctype_labels,
        roimask=roimask,
        regmasks=reg_masks,
        metarow=metarow,
        rawpath=raw_path,
        labelpath=label_path,
        curated=is_curated,
        was_inverted=was_inverted,
        enctypes_present=enctypes_present,
        host=host
    )

    return imgres

