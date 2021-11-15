"""
PyTorch Dataset classes for loading encapsulin segmentation datasets.
"""


import glob
import logging
import os
from os.path import expanduser
from typing import Tuple, Dict, Optional, Union, Sequence, Any, List, Callable
from pathlib import Path

import pandas as pd
import imageio
import numpy as np
import torch
from torch.utils import data

from elektronn3.data import transforms

logger = logging.getLogger('elektronn3log')


# Codes: BG: 0, MxEnc: 1, QtEnc: 2


MMMT3_TYPE = '1xMmMT3'
# MMMT3_TYPE = '2xMmMT3'

if MMMT3_TYPE == '1xMmMT3':
    VALID_NUMS = [22, 32, 42, 52, 54, 62]
    #            [qt, mx, qt, mx, qt, mx]
elif MMMT3_TYPE == '2xMmMT3':
    VALID_NUMS = [70, 75, 80, 85]
    #            [qt, qt, mx, mx]

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
        inp = imageio.imread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = imageio.imread(label_path).astype(np.int64)
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




MXENC_NUMS = list(range(31, 40 + 1)) + list(range(51, 53 + 1))
QTENC_NUMS = list(range(16, 30 + 1)) + list(range(41, 50 + 1)) + [54]


class __XTifDirData2d(data.Dataset):
    """Using a special TIF file directory structure for segmentation data loading.
    
    Version for mxqtsegtrain.py"""
    def __init__(
            self,
            data_root: str,
            label_names: Sequence[str],
            descr_sheet = (os.path.expanduser('~/tumdata2/Image_annotation_for_ML.xlsx'), 'Image_origin_information'),
            image_numbers: Optional[Sequence[Union[int, str]]] = None,
            transform=transforms.Identity(),
            offset: Optional[Sequence[int]] = (0, 0),
            inp_dtype=np.float32,
            target_dtype=np.int64,
            invert_labels=False,  # Fixes inverted TIF loading
            enable_inputmask: bool = False,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        self.data_root = data_root
        self.image_numbers = image_numbers
        self.label_names = label_names
        self.transform = transform
        self.offset = offset
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.invert_labels = invert_labels
        self.enable_inputmask = enable_inputmask
        self.epoch_multiplier = epoch_multiplier

        sheet = pd.read_excel(descr_sheet[0], sheet_name=descr_sheet[1])
        sheet = sheet.astype({'Image abbreviation': int})
        self.sheet = sheet
        metadata = sheet.copy()
        metadata = metadata[['Image abbreviation', 'MxEnc', 'QtEnc', '1xMmMT3', '2xMmMT3']]
        self.metadata = metadata
        # TODO

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

        # TODO: Add qt vs mx label

        index %= len(self.subdir_paths)  # Wrap around to support epoch_multiplier
        subdir_path = self.subdir_paths[index]
        img_num = subdir_path.name

        inp_path = subdir_path / f'{img_num}.tif'
        if not inp_path.exists():
            inp_path = subdir_path / f'{img_num}.TIF'
        inp = imageio.imread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = imageio.imread(label_path).astype(np.int64)
                if self.invert_labels:
                    label = (label == 0).astype(np.int64)
            else:  # If label is missing, make it a full zero array
                label = np.zeros_like(inp[0], dtype=np.int64)
            labels.append(label)
        assert len(labels) > 0

        # Flat target filled with label indices
        target = np.zeros_like(labels[0], dtype=np.int64)
        for c in range(len(self.label_names)):
            # Assign label index c to target at all locations where the c-th label is non-zero
            target[labels[c] != 0] = c

        if self.enable_inputmask:  # Zero out input where target == 0 to make background invisible
            for c in range(inp.shape[0]):
                inp[c][target == 0] = 0


        # Distinguish between MxEnc and QtEnc (quick and dirty, TODO improve this)
        # Background: 0, MxEnc: 1, QtEnc: 2
        if int(img_num) in QTENC_NUMS:
            target[target == 1] = 2
        else:
            pass # Keep target labels at 1
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
        inp = imageio.imread(inp_path).astype(self.inp_dtype)
        if inp.ndim == 2:  # (H, W)
            inp = inp[None]  # (C=1, H, W)


        labels = []
        for label_name in self.label_names:
            label_path = subdir_path / f'{img_num}_{label_name}.tif'
            if label_path.exists():
                label = imageio.imread(label_path).astype(np.int64)
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