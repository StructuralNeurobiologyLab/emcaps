#!/usr/bin/env python3

# TODO: Only use ignoring on TmEnc


"""
2D semantic segmentation training script
"""

import argparse
import datetime
import logging
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Dict
import os
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import pandas as pd
import yaml

# Don't move this stuff, it needs to be run this early to work
import elektronn3
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.utils import data

import hydra
from omegaconf import OmegaConf, DictConfig
# from hydra.core.config_store import ConfigStore


elektronn3.select_mpl_backend('Agg')
logger = logging.getLogger('emcaps-segtrain')

from elektronn3.training import Trainer, Backup, SWA
from elektronn3.training import metrics
from elektronn3.data import transforms
from elektronn3.models.unet import UNet
from elektronn3.modules.loss import CombinedLoss, DiceLoss

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import albumentations

from tqdm import tqdm

from emcaps.training.tifdirdata import EncSegData
from emcaps import utils


# @dataclass
# class TrainingConf:
#     data_root: str = '/wholebrain/scratch/mdraw/tum/Single-table_database'
#     dsel: str = '1x'
#     horg: Literal['HEK cell culture', 'Drosophila'] = 'HEK cell culture'

#     disable_cuda: bool = False
#     n: Optional[str] = None
#     max_steps: int = 80_000
#     seed: int = 0
#     deterministic: bool = False
#     save_root: str = '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_hek4_bin'



# cs = ConfigStore.instance()
# cs.store(name='training_conf', node=TrainingConf)


# @hydra.main(config_path='conf', config_name='training_conf')
# def main(cfg: DictConfig) -> None:
#     global conf
#     conf = cfg
#     print(OmegaConf.to_yaml(conf))


# if __name__ == "__main__":
#     main()


# print(conf.db)

# conf = OmegaConf.create()



@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg: DictConfig) -> None:

    # parser = argparse.ArgumentParser(description='Train a network.')
    # parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
    # parser.add_argument(
    #     '-m', '--max-steps', type=int, default=300_001,
    #     help='Maximum number of training steps to perform.'
    # )
    # parser.add_argument(
    #     '-r', '--resume', metavar='PATH',
    #     help='Path to pretrained model state dict from which to resume training.'
    # )
    # parser.add_argument('--seed', type=int, default=0, help='Base seed for all RNGs.')
    # parser.add_argument(
    #     '--deterministic', action='store_true',
    #     help='Run in fully deterministic mode (at the cost of execution speed).'
    # )
    # parser.add_argument('-c', '--constraintype', default=None, help='Constrain training and validation to only one encapsulin type (via v5name, e.g. `-c 1M-Qt`).')
    # args = parser.parse_args()

    # conf = args # TODO

    # Set up all RNG seeds, set level of determinism
    random_seed = cfg.segtrain.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

    device = torch.device('cuda')
    print(f'Running on device: {device}')

    # 0: background
    # 1: membranes
    # 2: encapsulins
    # 3: nuclear_membrane
    # 4: nuclear_region
    # 5: cytoplasmic_region


    SHEET_NAME = 0  # index of sheet

    # class_groups_to_include = [
    #     'simple_hek',
    #     'dro',
    #     'mice',
    #     'qttm',
    #     'multi',
    # ]

    # if cfg.segtrain.constraintype is None:
    #     included = []
    #     for cgrp in class_groups_to_include:
    #         cgrp_classes = utils.CLASS_GROUPS[cgrp]
    #         logger.info(f'Including class group {cgrp}, containing classes {cgrp_classes}')
    #         included.extend(utils.CLASS_GROUPS[cgrp])
    #     DATA_SELECTION_V5NAMES = included
    # else:
    #     DATA_SELECTION_V5NAMES = [cfg.segtrain.constraintype]


    # IGNORE_FAR_BACKGROUND_DISTANCE = 16
    IGNORE_FAR_BACKGROUND_DISTANCE = 0

    BG_WEIGHT = 0.2

    INPUTMASK = False

    INVERT_LABELS = False
    # INVERT_LABELS = True

    # USE_GRAY_AUG = False
    USE_GRAY_AUG = True

    DILATE_TARGETS_BY = 0


    # data_root = Path('/wholebrain/scratch/mdraw/tum/Single-table_database').expanduser()
    # data_root = Path(cfg.data_root).expanduser()

    label_names = ['=ZEROS=', 'encapsulins']
    target_dtype = np.int64

    out_channels = 2
    model = UNet(
        out_channels=out_channels,
        n_blocks=5,
        start_filts=64,
        activation='relu',
        normalization='batch',
        dim=2
    ).to(device)

    # USER PATHS
    sr_suffix = ''
    # if ONLY_QTTM:
    #     sr_suffix = f'{sr_suffix}_onlyqttm'
    # save_root = Path(f'/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v13{sr_suffix}').expanduser()
    save_root = Path(cfg.segtrain.save_root).expanduser()


    max_steps = cfg.segtrain.max_steps
    lr = cfg.segtrain.lr
    lr_stepsize = cfg.segtrain.lr_stepsize
    lr_dec = cfg.segtrain.lr_dec
    batch_size = cfg.segtrain.batch_size

    # TODO: https://github.com/Project-MONAI/MONAI/blob/384c7181e3730988fe5318e9592f4d65c12af843/monai/transforms/croppad/array.py#L831

    # Transformations to be applied to samples before feeding them to the network
    common_transforms = [
        transforms.RandomCrop((512, 512)),
        transforms.Normalize(mean=cfg.dataset_mean, std=cfg.dataset_std, inplace=False),
        transforms.RandomFlip(ndim_spatial=2),
    ]

    train_transform = common_transforms + [
        # transforms.RandomCrop((512, 512)),
        transforms.AlbuSeg2d(albumentations.ShiftScaleRotate(
            p=0.9, rotate_limit=180, shift_limit=0.0625, scale_limit=0.1, interpolation=2
        )),  # interpolation=2 means cubic interpolation (-> cv2.CUBIC constant).
        # transforms.ElasticTransform(prob=0.5, sigma=2, alpha=5),
        transforms.RandomCrop((384, 384)),
    ]
    if USE_GRAY_AUG:
        train_transform.extend([
            transforms.AdditiveGaussianNoise(prob=0.3, sigma=0.1),
            transforms.RandomGammaCorrection(prob=0.3, gamma_std=0.1),
            transforms.RandomBrightnessContrast(prob=0.3, brightness_std=0.1, contrast_std=0.1),
        ])

    valid_transform = common_transforms + []


    train_transform = transforms.Compose(train_transform)
    valid_transform = transforms.Compose(valid_transform)


    train_dataset = EncSegData(
        descr_sheet=(cfg.sheet_path, SHEET_NAME),
        data_group=cfg.data_group,
        # valid_nums=valid_image_numbers,  # read from table
        train=True,
        data_path=cfg.isplit_data_path,
        label_names=label_names,
        transform=train_transform,
        target_dtype=target_dtype,
        invert_labels=INVERT_LABELS,
        enable_inputmask=INPUTMASK,
        ignore_far_background_distance=IGNORE_FAR_BACKGROUND_DISTANCE,
        dilate_targets_by=DILATE_TARGETS_BY,
        epoch_multiplier=200,
    )

    valid_dataset = EncSegData(
        descr_sheet=(cfg.sheet_path, SHEET_NAME),
        data_group=cfg.data_group,
        # valid_nums=valid_image_numbers,  # read from table
        train=False,
        data_path=cfg.isplit_data_path,
        label_names=label_names,
        transform=valid_transform,
        target_dtype=target_dtype,
        invert_labels=INVERT_LABELS,
        enable_inputmask=INPUTMASK,
        ignore_far_background_distance=IGNORE_FAR_BACKGROUND_DISTANCE,
        dilate_targets_by=DILATE_TARGETS_BY,
        epoch_multiplier=10,
    )

    logger.info(f'Selected data_group: {cfg.data_group}')
    logger.info(f'Including images {list(train_dataset.meta.num.unique())}')


    # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=5e-5,
        lr=lr,
        amsgrad=True
    )
    optimizer = SWA(optimizer)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)


    # Validation metrics

    valid_metrics = {}
    for evaluator in [metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.DSC, metrics.IoU]:
        valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()  # Mean metrics
        for c in range(out_channels):
            valid_metrics[f'val_{evaluator.name}_c{c}'] = evaluator(c)


    class_weights = torch.tensor([BG_WEIGHT, 1.0]).to(device)
    ce = CrossEntropyLoss(weight=class_weights).to(device)
    gdl = DiceLoss(apply_softmax=True, weight=class_weights.to(device))
    criterion = CombinedLoss([ce, gdl], device=device)

    inference_kwargs = {
        'apply_softmax': True,
        'transform': valid_transform,
    }

    exp_name = cfg.segtrain.exp_name
    if exp_name is None:
        exp_name = ''
    timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    exp_name = f'{exp_name}__{model.__class__.__name__ + "__" + timestamp}'
    exp_name = f'{cfg.data_group}_{exp_name}'



    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=8,
        save_root=save_root,
        exp_name=exp_name,
        inference_kwargs=inference_kwargs,
        save_jit='script',
        schedulers={"lr": lr_sched},
        valid_metrics=valid_metrics,
        out_channels=out_channels,
        mixed_precision=True,
        extra_save_steps=list(range(40_000, max_steps + 1, 40_000)),
    )



    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__,
                save_path=trainer.save_path).archive_backup()

    # Start training
    trainer.run(max_steps)


if __name__ == '__main__':
    main()