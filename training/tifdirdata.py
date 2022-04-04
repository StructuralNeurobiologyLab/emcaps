"""
PyTorch Dataset classes for loading encapsulin segmentation datasets.
"""


import glob
import logging
import os
from os.path import expanduser
from typing import Tuple, Dict, Optional, Union, Sequence, Any, List, Callable
from pathlib import Path
from functools import lru_cache

import pandas as pd
import imageio
import numpy as np
from skimage import morphology as sm
import torch
from torch.utils import data
import yaml

from elektronn3.data import transforms

logger = logging.getLogger('elektronn3log')



# Data v2: All 1xMmMT3 data (exclude only 1..15)
MXENC_NUMS = list(range(31, 40 + 1)) + list(range(51, 53 + 1)) + list(range(60, 69 + 1))
QTENC_NUMS = list(range(16, 30 + 1)) + list(range(41, 50 + 1)) + list(range(54, 59 + 1))

# Codes: MxEnc: 0, QtEnc: 1
ECODE = {'mx': 0, 'qt': 1}
EFULLNAMES = {
    'mx': 'MxEnc',
    'qt': 'QtEnc',
    0: 'MxEnc',
    1: 'QtEnc'
}

valid_split_path = './class_info.yaml'
with open(valid_split_path) as f:
    CLASS_INFO = yaml.load(f, Loader=yaml.FullLoader)

CLASS_IDS = CLASS_INFO['class_ids']
CLASS_NAMES = list(CLASS_IDS.keys())

MMMT3_TYPE = '1xMmMT3'
# MMMT3_TYPE = '2xMmMT3'

if MMMT3_TYPE == '1xMmMT3':
    VALID_NUMS = [22, 32, 42, 52, 54, 62]
    #            [qt, mx, qt, mx, qt, mx]
elif MMMT3_TYPE == '2xMmMT3':
    VALID_NUMS = [70, 75, 80, 85]
    #            [qt, qt, mx, mx]


@lru_cache(maxsize=1024)
def mimread(*args, **kwargs):
    """Memoize imread to avoid disk I/O"""
    return imageio.imread(*args, **kwargs)


# Credit: https://newbedev.com/how-can-i-create-a-circular-mask-for-a-numpy-array
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    yy, xx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    mask = dist_from_center <= radius
    return mask


class UPatches(data.Dataset):
    """Image-level classification dataset loader for small patches, similar to MNIST"""
    def __init__(
            self,
            # data_root: str,
            descr_sheet = (os.path.expanduser('/wholebrain/scratch/mdraw/tum/patches_v4_uni/patchmeta_traintest.xlsx'), 'Sheet1'),
            train: bool = True,
            transform=transforms.Identity(),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            dilate_masks_by: int = 0,
            erase_mask_bg: bool = False,
            erase_disk_mask_radius: int = 0,
            epoch_multiplier: int = 1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        # self.data_root = data_root
        self.train = train
        self.transform = transform
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.dilate_masks_by = dilate_masks_by
        self.erase_mask_bg = erase_mask_bg
        self.erase_disk_mask_radius = erase_disk_mask_radius
        self.epoch_multiplier = epoch_multiplier

        sheet = pd.read_excel(descr_sheet[0], sheet_name=descr_sheet[1])
        self._sheet = sheet
        meta = sheet.copy()

        if self.train:
            logger.info('\nTraining data:')
            meta = meta.loc[meta.train == True]  # Explicit comparison due to possible integer types
        else:
            logger.info('\nValidation data:')
            meta = meta.loc[meta.validation == True]  # Explicit comparison due to possible integer types

        self.meta = meta

        # self.root_path = Path(data_root).expanduser()
        self.root_path = Path(descr_sheet[0]).parent

        self.inps = []
        self.targets = []

        for patch_meta in self.meta.itertuples():
            inp = mimread(self.root_path / 'raw' / patch_meta.patch_fname)
            if self.erase_mask_bg:
                # Erase mask background from inputs
                mask = mimread(self.root_path / 'mask' / patch_meta.patch_fname.replace('raw', 'mask'))
                if self.dilate_masks_by > 0:
                    disk = sm.disk(self.dilate_masks_by)
                    # mask_patch = ndimage.binary_dilation(mask_patch, iterations=DILATE_MASKS_BY)
                    mask = sm.binary_dilation(mask, selem=disk)
                inp[mask == 0] = 0
            if self.erase_disk_mask_radius > 0:
                mask = create_circular_mask(*inp.shape, radius=self.erase_disk_mask_radius)
                inp[mask > 0] = 0

            # mask = mimread(self.root_path / 'mask' / patch_meta.patch_fname.replace('raw', 'mask'))
            # inp = mask * 255

            target = CLASS_IDS[patch_meta.enctype]
            self.inps.append(inp)
            self.targets.append(target)
        
        self.inps = np.stack(self.inps).astype(self.inp_dtype)
        self.targets = np.stack(self.targets).astype(self.target_dtype)

        for enctype in meta.enctype.unique():
            logger.info(f'{enctype}: {meta[meta.enctype == enctype].shape[0]}')



    def __getitem__(self, index):
        index %= len(self.meta)  # Wrap around to support epoch_multiplier
        inp = self.inps[index]
        target = self.targets[index]
        fname = self.meta.patch_fname.iloc[index]
        label_name = target
        fname = f'{fname} ({label_name})'
        inp = inp[None]  # (C=1, H, W)
        # Pass None instead of target because scalar targets are not to be augmented.
        inp, _ = self.transform(inp, None)
        sample = {
            'inp': torch.as_tensor(inp),
            'target': torch.as_tensor(target),
            'fname': fname,
        }
        return sample

    def __len__(self):
        return len(self.meta) * self.epoch_multiplier



class Patches(data.Dataset):
    """Image-level classification dataset loader for small patches, similar to MNIST"""
    def __init__(
            self,
            # data_root: str,
            descr_sheet = (os.path.expanduser('/wholebrain/scratch/mdraw/tum/patches_v4_uni/patchmeta_traintest.xlsx'), 'Sheet1'),
            train: bool = True,
            transform=transforms.Identity(),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            erase_mask_bg: bool = False,
            erase_disk_mask_radius: int = 0,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        # self.data_root = data_root
        self.train = train
        self.transform = transform
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.erase_mask_bg = erase_mask_bg
        self.erase_disk_mask_radius = erase_disk_mask_radius
        self.epoch_multiplier = epoch_multiplier

        sheet = pd.read_excel(descr_sheet[0], sheet_name=descr_sheet[1])
        self._sheet = sheet
        meta = sheet.copy()

        if self.train:
            logger.info('\nTraining data:')
            meta = meta[meta.train]
        else:
            logger.info('\nValidation data:')
            meta = meta[meta.test]

        self.meta = meta

        # self.root_path = Path(data_root).expanduser()
        self.root_path = Path(descr_sheet[0]).parent

        self.inps = []
        self.targets = []

        for patch_meta in self.meta.itertuples():
            inp = mimread(self.root_path / 'raw' / patch_meta.patch_fname)
            if self.erase_mask_bg:
                # Erase mask background from inputs
                mask = mimread(self.root_path / 'mask' / patch_meta.patch_fname.replace('raw', 'mask'))
                inp[mask == 0] = 0
            if self.erase_disk_mask_radius > 0:
                mask = create_circular_mask(*inp.shape, radius=self.erase_disk_mask_radius)
                inp[mask > 0] = 0

            # mask = mimread(self.root_path / 'mask' / patch_meta.patch_fname.replace('raw', 'mask'))
            # inp = mask * 255

            target = ECODE[patch_meta.enctype]
            self.inps.append(inp)
            self.targets.append(target)
        
        self.inps = np.stack(self.inps).astype(self.inp_dtype)
        self.targets = np.stack(self.targets).astype(self.target_dtype)

        logger.info(f'MxEnc: {meta[meta.enctype == "mx"].shape[0]}')
        logger.info(f'QtEnc: {meta[meta.enctype == "qt"].shape[0]}')



    def __getitem__(self, index):
        index %= len(self.meta)  # Wrap around to support epoch_multiplier
        inp = self.inps[index]
        target = self.targets[index]
        fname = self.meta.patch_fname.iloc[index]
        label_name = EFULLNAMES[target]
        fname = f'{fname} ({label_name})'
        inp = inp[None]  # (C=1, H, W)
        # Pass None instead of target because scalar targets are not to be augmented.
        inp, _ = self.transform(inp, None)
        sample = {
            'inp': torch.as_tensor(inp),
            'target': torch.as_tensor(target),
            'fname': fname,
        }
        return sample

    def __len__(self):
        return len(self.meta) * self.epoch_multiplier


class UTifDirData2d(data.Dataset):
    """Using a special TIF file directory structure for segmentation data loading.

    Version for mxqtsegtrain2.py.
    For training on all conditions or a subset thereof."""
    def __init__(
            self,
            # data_root: str,
            label_names: Sequence[str],
            valid_nums: Optional[Sequence[int]] = None,
            descr_sheet = (os.path.expanduser('/wholebrain/scratch/mdraw/tum/Single-table_database/Image_annotation_for_ML_single_table.xlsx'), 'all_metadata'),
            meta_filter = lambda x: x,
            train: bool = True,
            transform=transforms.Identity(),
            offset: Optional[Sequence[int]] = (0, 0),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            invert_labels=False,  # Fixes inverted TIF loading
            enable_inputmask: bool = False,
            enable_binary_seg: bool = False,
            ignore_far_background_distance: int = 0,
            enable_partial_inversion_hack: bool = False,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        # self.data_root = data_root
        self.label_names = label_names
        self.meta_filter = meta_filter
        self.train = train
        self.transform = transform
        self.offset = offset
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.invert_labels = invert_labels
        self.enable_inputmask = enable_inputmask
        self.ignore_far_background_distance = ignore_far_background_distance
        self.epoch_multiplier = epoch_multiplier
        self.valid_nums = valid_nums
        self.enable_binary_seg = enable_binary_seg
        self.enable_partial_inversion_hack = enable_partial_inversion_hack

        if self.ignore_far_background_distance:
            self.ifbd_disk = sm.disk(self.ignore_far_background_distance)

        sheet = pd.read_excel(descr_sheet[0], sheet_name=descr_sheet[1])
        self.sheet = sheet
        meta = sheet.copy()
        meta = meta.rename(columns={' Image': 'num'})
        meta = meta.rename(columns={'Short experimental condition': 'scond'})
        meta = meta.convert_dtypes()

        meta = self.meta_filter(meta)

        if self.train:
            logger.info('\nTraining data:')
            if valid_nums is None:  # Read from table
                meta = meta.loc[meta['Training'] == 1]
            else:  # Use list
                meta = meta[~meta['num'].isin(self.valid_nums)]
        else:
            logger.info('\nValidation data:')
            if valid_nums is None:  # Read from table
                meta = meta.loc[meta['Validation'] == 1]
            else:  # Use list
                meta = meta[meta['num'].isin(self.valid_nums)]

        self.meta = meta
        self.root_path = Path(descr_sheet[0]).parent
        self.image_numbers = self.meta['num'].to_list()

        conditions = self.meta['scond'].unique()
        for condition in conditions:
            _nums = meta.loc[meta['scond'] == condition]['num'].to_list()
            logger.info(f'{condition}:\t({len(_nums)} images):\n {_nums}')



    def __getitem__(self, index):
        # if self.multilabel_targets and len(self.label_names) != 1:
            # raise ValueError('multilabel_targets=False requires a single label_name')


        index %= len(self.meta)  # Wrap around to support epoch_multiplier
        # subdir_path = self.subdir_paths[index]
        mrow = self.meta.iloc[index]
        img_num = mrow['num']
        subdir_path = self.root_path / f'{img_num}'

        inp_path = subdir_path / f'{img_num}.tif'
        if not inp_path.exists():
            inp_path = subdir_path / f'{img_num}.TIF'
        inp = mimread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = mimread(label_path).astype(np.int64)
                if self.invert_labels:
                    label = (label == 0).astype(np.int64)
                if self.enable_partial_inversion_hack and int(img_num) < 55:  # TODO: Investigate why labels are inverted although images look fine
                    label = (label == 0).astype(np.int64)
            else:  # If label is missing, make it a full zero array
                label = np.zeros_like(inp[0], dtype=np.int64)
            labels.append(label)
        assert len(labels) > 0

        # Flat target filled with label indices
        target = np.zeros_like(labels[0], dtype=np.int64)
        # for c in range(len(self.label_names)):
        #     # Assign label index c to target at all locations where the c-th label is non-zero
        #     target[labels[c] != 0] = c

        target[labels[1] != 0] = 1

        if self.enable_binary_seg:  # Don't distinguish between foreground classes, just use one foreground class
            target[target > 0] = 1

        if self.enable_inputmask:  # Zero out input where target == 0 to make background invisible
            for c in range(inp.shape[0]):
                inp[c][target == 0] = 0

        if target.mean().item() > 0.2:
            print('Unusually high target mean in image number', img_num)

        # Mark regions to be ignored
        if self.ignore_far_background_distance > 0 and mrow['scond'] == 'HEK-1xTmEnc-BC2-Tag':
            dilated_foreground = sm.binary_dilation(target, selem=self.ifbd_disk)
            far_background = ~dilated_foreground
            target[far_background] = -1

        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, target = self.transform(inp, target)
                break
            except transforms._DropSample:
                pass
        if np.any(self.offset):
            off = self.offset
            target = target[off[0]:-off[0], off[1]:-off[1]]
        sample = {
            'inp': torch.as_tensor(inp.astype(self.inp_dtype)),
            'target': torch.as_tensor(target.astype(self.target_dtype)),
            'fname': f'{subdir_path.name} ({mrow["scond"]})',
        }
        return sample

    def __len__(self):
        return len(self.meta) * self.epoch_multiplier





### From here on this is legacy code for older dataset versions





class ZTifDirData2d(data.Dataset):
    """Using a special TIF file directory structure for segmentation data loading.

    Version for mxqtsegtrain2.py.
    For training on both 1xMmMT3 and 2xMmMT3."""
    def __init__(
            self,
            # data_root: str,
            label_names: Sequence[str],
            valid_nums: Sequence[int],
            descr_sheet = (os.path.expanduser('/wholebrain/scratch/mdraw/tum/Single-table_database/Image_annotation_for_ML_single_table.xlsx'), 'Image_origin_information'),
            meta_filter = lambda x: x,
            train: bool = True,
            transform=transforms.Identity(),
            offset: Optional[Sequence[int]] = (0, 0),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            invert_labels=False,  # Fixes inverted TIF loading
            enable_inputmask: bool = False,
            enable_binary_seg: bool = False,
            enable_partial_inversion_hack: bool = False,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        # self.data_root = data_root
        self.label_names = label_names
        self.meta_filter = meta_filter
        self.train = train
        self.transform = transform
        self.offset = offset
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.invert_labels = invert_labels
        self.enable_inputmask = enable_inputmask
        self.epoch_multiplier = epoch_multiplier
        self.valid_nums = valid_nums
        self.enable_binary_seg = enable_binary_seg
        self.enable_partial_inversion_hack = enable_partial_inversion_hack


        sheet = pd.read_excel(descr_sheet[0], sheet_name=descr_sheet[1])
        self.sheet = sheet
        try:
            meta = sheet.rename(columns={'Image abbreviation': 'num'}).astype({'num': int})
        except KeyError:  # Unnamed column in single-table format
            meta = sheet.rename(columns={' ': 'num'}).astype({'num': int})

        meta = self.meta_filter(meta)

        if self.train:
            logger.info('\nTraining data:')
            meta = meta[~meta['num'].isin(self.valid_nums)]
        else:
            logger.info('\nValidation data:')
            meta = meta[meta['num'].isin(self.valid_nums)]

        self.meta = meta

        # self.root_path = Path(data_root).expanduser()
        self.root_path = Path(descr_sheet[0]).parent

        self.image_numbers = self.meta['num'].to_list()

        _mxnums = meta[meta['MxEnc']]['num'].to_list()
        _qtnums = meta[meta['QtEnc']]['num'].to_list()

        logger.info(f'MxEnc ({len(_mxnums)} images):\n {_mxnums}')
        logger.info(f'QtEnc ({len(_qtnums)} images):\n  {_qtnums}')



    def __getitem__(self, index):
        # if self.multilabel_targets and len(self.label_names) != 1:
            # raise ValueError('multilabel_targets=False requires a single label_name')


        index %= len(self.meta)  # Wrap around to support epoch_multiplier
        # subdir_path = self.subdir_paths[index]
        mrow = self.meta.iloc[index]
        img_num = mrow['num']
        subdir_path = self.root_path / f'{img_num}'

        inp_path = subdir_path / f'{img_num}.tif'
        if not inp_path.exists():
            inp_path = subdir_path / f'{img_num}.TIF'
        inp = mimread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = mimread(label_path).astype(np.int64)
                if self.invert_labels:
                    label = (label == 0).astype(np.int64)
                if self.enable_partial_inversion_hack and int(img_num) < 55:  # TODO: Investigate why labels are inverted although images look fine
                    label = (label == 0).astype(np.int64)
            else:  # If label is missing, make it a full zero array
                label = np.zeros_like(inp[0], dtype=np.int64)
            labels.append(label)
        assert len(labels) > 0

        # Flat target filled with label indices
        target = np.zeros_like(labels[0], dtype=np.int64)
        # for c in range(len(self.label_names)):
        #     # Assign label index c to target at all locations where the c-th label is non-zero
        #     target[labels[c] != 0] = c

        enctype_name = ''
        if mrow['MxEnc']:
            target[labels[1] != 0] = 1
            enctype_name = 'MxEnc'
        elif mrow['QtEnc']:
            target[labels[1] != 0] = 2
            enctype_name = 'QtEnc'
        else:
            raise ValueError(mrow)

        x_name = ''
        if mrow['1xMmMT3']:
            x_name = '1xMmMT3'
        elif mrow['2xMmMT3']:
            x_name = '2xMmMT3'
        else:
            raise ValueError(mrow)

        if self.enable_binary_seg:  # Don't distinguish between foreground classes, just use one foreground class
            target[target > 0] = 1

        if self.enable_inputmask:  # Zero out input where target == 0 to make background invisible
            for c in range(inp.shape[0]):
                inp[c][target == 0] = 0

        if target.mean().item() > 0.2:
            print('Unusually high target mean in image number', img_num)

        # Distinguish between MxEnc and QtEnc (quick and dirty, TODO improve this)
        # Background: 0, MxEnc: 1, QtEnc: 2
        # if mrow['QtEnc']:
        #     target[target == 1] = 2
        # elif:
        #     pass # Keep target labels at 1

        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, target = self.transform(inp, target)
                break
            except transforms._DropSample:
                pass
        if np.any(self.offset):
            off = self.offset
            target = target[off[0]:-off[0], off[1]:-off[1]]
        sample = {
            'inp': torch.as_tensor(inp.astype(self.inp_dtype)),
            'target': torch.as_tensor(target.astype(self.target_dtype)),
            'fname': f'{subdir_path.name} ({x_name} {enctype_name})',
        }
        return sample

    def __len__(self):
        return len(self.meta) * self.epoch_multiplier


class YTifDirData2d(data.Dataset):
    """Using a special TIF file directory structure for segmentation data loading.
    
    Version for mxqtsegtrain2.py"""
    def __init__(
            self,
            # data_root: str,
            label_names: Sequence[str],
            valid_nums: Sequence[int],
            descr_sheet = (os.path.expanduser('/wholebrain/scratch/mdraw/tum/Single-table_database/Image_annotation_for_ML_single_table.xlsx'), 'Image_origin_information'),
            meta_filter = lambda x: x,
            train: bool = True,
            transform=transforms.Identity(),
            offset: Optional[Sequence[int]] = (0, 0),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            invert_labels=False,  # Fixes inverted TIF loading
            enable_inputmask: bool = False,
            enable_binary_seg: bool = False,
            enable_partial_inversion_hack: bool = False,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        # self.data_root = data_root
        self.label_names = label_names
        self.meta_filter = meta_filter
        self.train = train
        self.transform = transform
        self.offset = offset
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.invert_labels = invert_labels
        self.enable_inputmask = enable_inputmask
        self.epoch_multiplier = epoch_multiplier
        self.valid_nums = valid_nums
        self.enable_binary_seg = enable_binary_seg
        self.enable_partial_inversion_hack = enable_partial_inversion_hack


        sheet = pd.read_excel(descr_sheet[0], sheet_name=descr_sheet[1])
        self.sheet = sheet
        try:
            meta = sheet.rename(columns={'Image abbreviation': 'num'}).astype({'num': int})
        except KeyError:  # Unnamed column in single-table format
            meta = sheet.rename(columns={' ': 'num'}).astype({'num': int})

        # Temporary filters, TODO: refactor

        meta = meta.loc[meta['1xMmMT3']]
        meta = meta.loc[meta['Host organism'] == 'Drosophila']
        meta = meta.loc[meta['Modality'] == 'TEM']

        # meta = meta[['num', 'MxEnc', 'QtEnc', '1xMmMT3', '2xMmMT3']]


        meta = self.meta_filter(meta)

        if self.train:
            logger.info('\nTraining data:')
            meta = meta[~meta['num'].isin(self.valid_nums)]
        else:
            logger.info('\nValidation data:')
            meta = meta[meta['num'].isin(self.valid_nums)]

        self.meta = meta

        # self.root_path = Path(data_root).expanduser()
        self.root_path = Path(descr_sheet[0]).parent

        self.image_numbers = self.meta['num'].to_list()

        _mxnums = meta[meta['MxEnc']]['num'].to_list()
        _qtnums = meta[meta['QtEnc']]['num'].to_list()

        logger.info(f'MxEnc ({len(_mxnums)} images):\n {_mxnums}')
        logger.info(f'QtEnc ({len(_qtnums)} images):\n  {_qtnums}')



    def __getitem__(self, index):
        # if self.multilabel_targets and len(self.label_names) != 1:
            # raise ValueError('multilabel_targets=False requires a single label_name')


        index %= len(self.meta)  # Wrap around to support epoch_multiplier
        # subdir_path = self.subdir_paths[index]
        mrow = self.meta.iloc[index]
        img_num = mrow['num']
        subdir_path = self.root_path / f'{img_num}'

        inp_path = subdir_path / f'{img_num}.tif'
        if not inp_path.exists():
            inp_path = subdir_path / f'{img_num}.TIF'
        inp = mimread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = mimread(label_path).astype(np.int64)
                if self.invert_labels:
                    label = (label == 0).astype(np.int64)
                if self.enable_partial_inversion_hack and int(img_num) < 60:  # TODO: Investigate why labels are inverted although images look fine
                    label = (label == 0).astype(np.int64)
            else:  # If label is missing, make it a full zero array
                label = np.zeros_like(inp[0], dtype=np.int64)
            labels.append(label)
        assert len(labels) > 0

        # Flat target filled with label indices
        target = np.zeros_like(labels[0], dtype=np.int64)
        # for c in range(len(self.label_names)):
        #     # Assign label index c to target at all locations where the c-th label is non-zero
        #     target[labels[c] != 0] = c

        enctype_name = ''
        if mrow['MxEnc']:
            target[labels[1] != 0] = 1
            enctype_name = 'MxEnc'
        elif mrow['QtEnc']:
            target[labels[1] != 0] = 2
            enctype_name = 'QtEnc'
        else:
            raise ValueError(mrow)

        if self.enable_binary_seg:  # Don't distinguish between foreground classes, just use one foreground class
            target[target > 0] = 1

        if self.enable_inputmask:  # Zero out input where target == 0 to make background invisible
            for c in range(inp.shape[0]):
                inp[c][target == 0] = 0


        # Distinguish between MxEnc and QtEnc (quick and dirty, TODO improve this)
        # Background: 0, MxEnc: 1, QtEnc: 2
        # if mrow['QtEnc']:
        #     target[target == 1] = 2
        # elif:
        #     pass # Keep target labels at 1

        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, target = self.transform(inp, target)
                break
            except transforms._DropSample:
                pass
        if np.any(self.offset):
            off = self.offset
            target = target[off[0]:-off[0], off[1]:-off[1]]
        sample = {
            'inp': torch.as_tensor(inp.astype(self.inp_dtype)),
            'target': torch.as_tensor(target.astype(self.target_dtype)),
            'fname': f'{subdir_path.name} ({enctype_name})',
        }
        return sample

    def __len__(self):
        return len(self.meta) * self.epoch_multiplier


class XTifDirData2d(data.Dataset):
    """Using a special TIF file directory structure for segmentation data loading.
    
    Version for mxqtsegtrain.py"""
    def __init__(
            self,
            # data_root: str,
            label_names: Sequence[str],
            descr_sheet = (os.path.expanduser('~/tumdata2/Image_annotation_for_ML.xlsx'), 'Image_origin_information'),
            meta_filter = lambda x: x,
            train: bool = True,
            transform=transforms.Identity(),
            offset: Optional[Sequence[int]] = (0, 0),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            invert_labels=False,  # Fixes inverted TIF loading
            enable_inputmask: bool = False,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        # self.data_root = data_root
        self.label_names = label_names
        self.meta_filter = meta_filter
        self.train = train
        self.transform = transform
        self.offset = offset
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.invert_labels = invert_labels
        self.enable_inputmask = enable_inputmask
        self.epoch_multiplier = epoch_multiplier

        sheet = pd.read_excel(descr_sheet[0], sheet_name=descr_sheet[1])
        self.sheet = sheet
        meta = sheet.rename(columns={'Image abbreviation': 'num'}).astype({'num': int})

        # Temporary filters, TODO: refactor

        # Only use 1xMmMT3 or 2xMmMT3
        meta = meta.loc[meta[MMMT3_TYPE]]
        # meta = meta[['num', 'MxEnc', 'QtEnc', '1xMmMT3', '2xMmMT3']]
        meta = self.meta_filter(meta)

        if self.train:
            logger.info('\nTraining data:')
            meta = meta[~meta['num'].isin(VALID_NUMS)]
        else:
            logger.info('\nValidation data:')
            meta = meta[meta['num'].isin(VALID_NUMS)]

        self.meta = meta

        # self.root_path = Path(data_root).expanduser()
        self.root_path = Path(descr_sheet[0]).parent

        self.image_numbers = self.meta['num'].to_list()

        _mxnums = meta[meta['MxEnc']]['num'].to_list()
        _qtnums = meta[meta['QtEnc']]['num'].to_list()

        logger.info(f'MxEnc:\n {_mxnums}')
        logger.info(f'QtEnc:\n  {_qtnums}')



    def __getitem__(self, index):
        # if self.multilabel_targets and len(self.label_names) != 1:
            # raise ValueError('multilabel_targets=False requires a single label_name')

        # TODO: Add qt vs mx label

        index %= len(self.meta)  # Wrap around to support epoch_multiplier
        # subdir_path = self.subdir_paths[index]
        mrow = self.meta.iloc[index]
        img_num = mrow['num']
        subdir_path = self.root_path / f'{img_num}'

        inp_path = subdir_path / f'{img_num}.tif'
        if not inp_path.exists():
            inp_path = subdir_path / f'{img_num}.TIF'
        inp = mimread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = mimread(label_path).astype(np.int64)
                if self.invert_labels:
                    label = (label == 0).astype(np.int64)
            else:  # If label is missing, make it a full zero array
                label = np.zeros_like(inp[0], dtype=np.int64)
            labels.append(label)
        assert len(labels) > 0

        # Flat target filled with label indices
        target = np.zeros_like(labels[0], dtype=np.int64)
        # for c in range(len(self.label_names)):
        #     # Assign label index c to target at all locations where the c-th label is non-zero
        #     target[labels[c] != 0] = c

        if mrow['MxEnc']:
            target[labels[1] != 0] = 1
        elif mrow['QtEnc']:
            target[labels[1] != 0] = 2
        else:
            raise ValueError(mrow)

        if self.enable_inputmask:  # Zero out input where target == 0 to make background invisible
            for c in range(inp.shape[0]):
                inp[c][target == 0] = 0


        # Distinguish between MxEnc and QtEnc (quick and dirty, TODO improve this)
        # Background: 0, MxEnc: 1, QtEnc: 2
        # if mrow['QtEnc']:
        #     target[target == 1] = 2
        # elif:
        #     pass # Keep target labels at 1

        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, target = self.transform(inp, target)
                break
            except transforms._DropSample:
                pass
        if np.any(self.offset):
            off = self.offset
            target = target[off[0]:-off[0], off[1]:-off[1]]
        sample = {
            'inp': torch.as_tensor(inp.astype(self.inp_dtype)),
            'target': torch.as_tensor(target.astype(self.target_dtype)),
            'fname': subdir_path.name,
        }
        return sample

    def __len__(self):
        return len(self.meta) * self.epoch_multiplier



class TifDirData2d(data.Dataset):
    """Using a special TIF file directory structure for segmentation data loading"""
    def __init__(
            self,
            data_root: str,
            label_names: Sequence[str],
            image_numbers: Optional[Sequence[Union[int, str]]] = None,
            multilabel_targets: bool = True,
            transform=transforms.Identity(),
            offset: Optional[Sequence[int]] = (0, 0),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            invert_labels=False,  # Fixes inverted TIF loading
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        self.data_root = data_root
        self.image_numbers = image_numbers
        self.label_names = label_names
        self.multilabel_targets = multilabel_targets
        self.transform = transform
        self.offset = offset
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.invert_labels = invert_labels
        self.epoch_multiplier = epoch_multiplier

        self.root_path = Path(data_root).expanduser()
        if image_numbers is None:
            self.subdir_paths = [p for p in self.root_path.iterdir() if p.is_dir()]
        else:
            self.subdir_paths = []
            image_numbers_str_set = {str(num) for num in image_numbers}
            for p in self.root_path.iterdir():
                if p.is_dir() and p.name in image_numbers_str_set:
                    self.subdir_paths.append(p)



    def __getitem__(self, index):
        # if self.multilabel_targets and len(self.label_names) != 1:
            # raise ValueError('multilabel_targets=False requires a single label_name')

        index %= len(self.subdir_paths)  # Wrap around to support epoch_multiplier
        subdir_path = self.subdir_paths[index]
        img_num = subdir_path.name

        inp_path = subdir_path / f'{img_num}.tif'
        if not inp_path.exists():
            inp_path = subdir_path / f'{img_num}.TIF'
        inp = mimread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = mimread(label_path).astype(np.int64)
                if self.invert_labels:
                    label = (label == 0).astype(np.int64)
            else:  # If label is missing, make it a full zero array
                label = np.zeros_like(inp[0], dtype=np.int64)
            labels.append(label)
        assert len(labels) > 0

        if self.multilabel_targets:
            target = np.stack(labels)
        else:  # Flat target filled with label indices
            target = np.zeros_like(labels[0], dtype=np.int64)
            for c in range(len(self.label_names)):
                # Assign label index c to target at all locations where the c-th label is non-zero
                target[labels[c] != 0] = c

        # _target = target.copy()
        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, target = self.transform(inp, target)
                break
            except transforms._DropSample:
                pass
        if np.any(self.offset):
            off = self.offset
            target = target[off[0]:-off[0], off[1]:-off[1]]
        sample = {
            'inp': torch.as_tensor(inp.astype(self.inp_dtype)),
            'target': torch.as_tensor(target.astype(self.target_dtype)),
            'fname': subdir_path.name,
        }
        return sample

    def __len__(self):
        return len(self.subdir_paths) * self.epoch_multiplier
