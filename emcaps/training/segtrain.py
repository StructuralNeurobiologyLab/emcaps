#!/usr/bin/env python3

# TODO: Only use ignoring on TmEnc


"""
2D semantic segmentation training script
"""

import datetime
import logging
import random
from pathlib import Path

# Don't move this stuff, it needs to be run this early to work
import elektronn3
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

# from hydra.core.config_store import ConfigStore

elektronn3.select_mpl_backend('Agg')
logger = logging.getLogger('emcaps-segtrain')

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import albumentations

from elektronn3.data import transforms
from elektronn3.models.unet import UNet
from elektronn3.modules.loss import CombinedLoss, DiceLoss
from elektronn3.training import SWA, Backup, Trainer, metrics

from emcaps.training.tifdirdata import EncSegData




@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # Set up all RNG seeds
    random_seed = cfg.segtrain.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases
    device = torch.device('cuda')
    print(f'Running on device: {device}')

    SHEET_NAME = 0  # index of sheet
    BG_WEIGHT = 0.2
    INPUTMASK = False

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

    save_root = Path(cfg.segtrain.save_root).expanduser()

    max_steps = cfg.segtrain.max_steps
    lr = cfg.segtrain.lr
    lr_stepsize = cfg.segtrain.lr_stepsize
    lr_dec = cfg.segtrain.lr_dec
    batch_size = cfg.segtrain.batch_size

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
    train_transform.extend([  # non-geometric grayscale augmentations
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
        train=True,
        data_path=cfg.isplit_data_path,
        label_names=label_names,
        transform=train_transform,
        target_dtype=target_dtype,
        enable_inputmask=INPUTMASK,
        epoch_multiplier=200,
    )

    valid_dataset = EncSegData(
        descr_sheet=(cfg.sheet_path, SHEET_NAME),
        data_group=cfg.data_group,
        train=False,
        data_path=cfg.isplit_data_path,
        label_names=label_names,
        transform=valid_transform,
        target_dtype=target_dtype,
        enable_inputmask=INPUTMASK,
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
    Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    # Start training
    trainer.run(max_steps)


if __name__ == '__main__':
    main()