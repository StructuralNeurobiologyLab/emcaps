#!/usr/bin/env python3


# TODO

# https://github.com/lucidrains/vit-pytorch#cct
# --> https://arxiv.org/abs/2104.05704

#

########


"""
Demo of a 2D semantic segmentation on TUM ML data v2, distinguishing QtEnc from MxEnc particles on patch basis.

Uses a patch dataset that can be created by inference/patchifyseg.py

"""

import datetime
import random
import logging

import torch
from torch import nn
from torch import optim
import numpy as np
import hydra
from omegaconf import DictConfig

# Don't move this stuff, it needs to be run this early to work
import elektronn3

elektronn3.select_mpl_backend('Agg')

from elektronn3.training import Trainer, Backup
from elektronn3.training import metrics
from elektronn3.data import transforms

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import albumentations

from emcaps.training.tifdirdata import EncPatchData
from emcaps.models.effnetv2 import effnetv2_m
from emcaps import utils


@hydra.main(version_base='1.2', config_path='../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Set up all RNG seeds, set level of determinism
    random_seed = cfg.patchtrain.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases
    device = torch.device('cuda')
    print(f'Running on device: {device}')

    ERASE_DISK_MASK_RADIUS = 0
    ERASE_MASK_BG = True
    NEGATIVE_SAMPLING = False

    if NEGATIVE_SAMPLING:
        assert not ERASE_MASK_BG

    SHEET_NAME = 0  # index of sheet
    descr_sheet = (cfg.patchtrain.patch_ds_sheet, SHEET_NAME)

    in_channels = 1
    out_channels = 8

    # model = effnetv2_s(in_c=1, num_classes=out_channels).to(device)
    model = effnetv2_m(in_c=in_channels, num_classes=out_channels).to(device)
    # model = effnetv2_l(in_c=1, num_classes=out_channels).to(device)


    # USER PATHS
    save_root = cfg.patchtrain.save_root

    max_steps = cfg.patchtrain.max_steps
    lr = cfg.patchtrain.lr
    lr_stepsize = cfg.patchtrain.lr_stepsize
    lr_dec = cfg.patchtrain.lr_dec
    batch_size = cfg.patchtrain.batch_size

    # Transformations to be applied to samples before feeding them to the network
    common_transforms = [
        transforms.Normalize(mean=cfg.dataset_mean, std=cfg.dataset_std, inplace=False),
        transforms.RandomFlip(ndim_spatial=2),
    ]

    train_transform = common_transforms + [
        transforms.AlbuSeg2d(albumentations.ShiftScaleRotate(
            p=0.9, rotate_limit=180, shift_limit=0.0, scale_limit=0.02, interpolation=2
        )),  # interpolation=2 means cubic interpolation (-> cv2.CUBIC constant).
        # transforms.ElasticTransform(prob=0.5, sigma=2, alpha=5),
        # transforms.AdditiveGaussianNoise(prob=0.9, sigma=0.1),
        # transforms.RandomGammaCorrection(prob=0.9, gamma_std=0.2),
        # transforms.RandomBrightnessContrast(prob=0.9, brightness_std=0.125, contrast_std=0.125),
    ]

    valid_transform = common_transforms + []

    train_transform = transforms.Compose(train_transform)
    valid_transform = transforms.Compose(valid_transform)

    # Specify data set
    train_dataset = EncPatchData(
        descr_sheet=descr_sheet,
        train=True,
        transform=train_transform,
        epoch_multiplier=5 if NEGATIVE_SAMPLING else 50,
        erase_mask_bg=ERASE_MASK_BG,
        erase_disk_mask_radius=ERASE_DISK_MASK_RADIUS,
    )

    valid_dataset = EncPatchData(
        descr_sheet=descr_sheet,
        train=False,
        transform=valid_transform,
        epoch_multiplier=4 if NEGATIVE_SAMPLING else 10,
        erase_mask_bg=ERASE_MASK_BG,
        erase_disk_mask_radius=ERASE_DISK_MASK_RADIUS,
    )

    # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=5e-5,
        lr=lr,
        amsgrad=True
    )
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

    # Validation metrics
    # TODO: Live confusion matrix generation and tracking would be cool
    valid_metrics = {}
    for evaluator in [metrics.Accuracy, metrics.Precision, metrics.Recall]:
        for c in range(out_channels):
            valid_metrics[f'val_{evaluator.name}_{utils.CLASS_NAMES[c]}'] = evaluator(c)
        valid_metrics[f'val_{evaluator.name}_mean'] = evaluator()

    criterion = nn.CrossEntropyLoss().to(device)

    inference_kwargs = {
        'apply_softmax': True,
        'transform': valid_transform,
    }

    exp_name = cfg.segtrain.exp_name
    if exp_name is None:
        exp_name = ''
    timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    exp_name = f'{exp_name}__{model.__class__.__name__ + "__" + timestamp}'
    exp_name = f'tr-{cfg.tr_group}_{exp_name}'

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
        extra_save_steps=list(range(10_000, max_steps + 1, 10_000)),
    )

    # Archiving training script, src folder, env info
    Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    # Start training
    trainer.run(max_steps)

if __name__ == '__main__':
    main()
