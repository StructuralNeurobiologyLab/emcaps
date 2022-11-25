"""
Script for cropping patches from the raw data with segmentation masks:

A trained neural network model (from segtrain.py) is used to predict
segmentation masks and masks are used to crop 28x28 pixel patches centered
around particle centroids
"""
import os
from pathlib import Path
from typing import NamedTuple
import shutil
import logging

import numpy as np
import imageio.v3 as iio
import tqdm
import pandas as pd
import hydra
from omegaconf import DictConfig

from PIL import Image, ImageDraw

from scipy import ndimage
from skimage import morphology as sm
from skimage import measure
import torch.backends.cudnn

from elektronn3.inference import Predictor
from elektronn3.data import transforms

from emcaps.utils.patch_utils import measure_outer_disk_radius, concentric_average, concentric_max
from emcaps import utils
from emcaps.utils import inference_utils as iu


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



class PatchMeta(NamedTuple):
    # patch_id: int  # Auto-assigned as row number index from pandas
    patch_fname: str
    img_num: int
    enctype: str
    dataset_name: str
    centroid_x: int
    centroid_y: int
    corner_x: int
    corner_y: int
    train: bool
    validation: bool
    radius2: float = np.nan
    radius2_dilated: float = np.nan
    area: float = np.nan
    area_dilated: float = np.nan
    circularity: float = np.nan


torch.backends.cudnn.benchmark = True

np.random.seed(0)



@hydra.main(version_base='1.2', config_path='../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    pre_predict_transform = transforms.Compose([
        transforms.Normalize(mean=cfg.dataset_mean, std=cfg.dataset_std)
    ])
    thresh = cfg.patchifyseg.thresh

    # TODO: Move these constants to cfg
    N_EVAL_SAMPLES = 30
    EC_REGION_RADIUS = cfg.patchifyseg.ec_region_radius
    EC_MIN_AREA = cfg.minsize
    EC_MAX_AREA = (2 * EC_REGION_RADIUS)**2
    USE_GT = cfg.patchifyseg.use_gt
    DILATE_MASKS_BY = cfg.patchifyseg.dilate_masks_by
    MIN_CIRCULARITY = cfg.patchifyseg.min_circularity
    ALL_VALIDATION = cfg.patchifyseg.all_validation

    # Add 1 to high region coordinate in order to arrive at an odd number of pixels in each dimension
    EC_REGION_ODD_PLUS1 = 1

    class_groups_to_include = [
        'simple_hek',
        'dro',
        'mice',
        'qttm',
        'multi',
    ]

    # root_path = Path('/wholebrain/scratch/mdraw/tum/Single-table_database/')
    sheet_path = Path(cfg.sheet_path)
    isplitdata_root = Path(cfg.isplit_data_path)

    img_paths = []
    for lp in isplitdata_root.rglob(f'*_{cfg.label_name}.png'):
        # Indirectly find img paths via label paths: raw can be always found by stripping the "_encapsulins" substring
        img_path = lp.with_stem(lp.stem.removesuffix(f'_{cfg.label_name}'))
        # Only include images that can be found in cfg.tr_group, to stay consistent with segmentation training/validation
        if utils.is_in_data_group(path_or_num=img_path, group_name=cfg.tr_group, sheet_path=sheet_path):
            img_paths.append(img_path)

    patch_out_path: str = os.path.expanduser(cfg.patchifyseg.patch_out_path)
    segmenter = cfg.patchifyseg.segmenter

    # Create output directories
    for p in [patch_out_path, f'{patch_out_path}/raw', f'{patch_out_path}/mask', f'{patch_out_path}/samples', f'{patch_out_path}/nobg', f'{patch_out_path}/cavg']:
        os.makedirs(p, exist_ok=True)

    # Set up logging
    logger = logging.getLogger('emcaps-patchifyseg')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{patch_out_path}/patchify.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    if segmenter == 'auto':
        segmenter = f'unet_{cfg.tr_group}_{cfg.v}'
        logger.info(f'Using default segmenter {segmenter} based on other config values')
        segmenter_model = iu.get_model(segmenter)
    elif segmenter == 'randomizer':
        logger.info('Using randomizer test model')
        segmenter_model = iu.Randomizer()  # Produce random outputs
    else:
        logger.info(f'Using segmenter {segmenter}')
        segmenter_model = iu.get_model(segmenter)

    predictor = Predictor(
        model=segmenter_model,
        device=None,
        float16=True,
        transform=pre_predict_transform,
        augmentations=cfg.patchifyseg.tta_num,
        apply_softmax=True,
    )


    logger.info(f'Using data from {isplitdata_root}')
    logger.info(f'Using meta spreadsheet {sheet_path}')
    logger.info(f'Writing outputs to {patch_out_path}')

    included = []
    for cgrp in class_groups_to_include:
        cgrp_classes = utils.CLASS_GROUPS[cgrp]
        logger.info(f'Including class group {cgrp}, containing classes {cgrp_classes}')
        included.extend(utils.CLASS_GROUPS[cgrp])
    DATA_SELECTION = included

    patchmeta = []

    patch_id = 0  # Incremented below for each patch written to disk

    for img_path in tqdm.tqdm(img_paths, position=0, desc='Images', dynamic_ncols=True):
        logger.debug(str(img_path))

        imgmeta = utils.get_meta_row(img_path, sheet_path=sheet_path)

        inp = np.array(iio.imread(img_path), dtype=np.float32)[None][None]  # (N=1, C=1, H, W)
        raw = inp[0][0]
        if USE_GT:
            label_path = img_path.with_name(f'{img_path.stem}_{cfg.label_name}.png')
            label = iio.imread(label_path).astype(np.int64)
            mask = label
        else:
            out = predictor.predict(inp)
            out = out.numpy()

            assert out.shape[1] == 2
            cout = out[0, 1]
            cout = (cout * 255.).astype(np.uint8)
            mask = cout > thresh

        mask = ndimage.binary_fill_holes(mask).astype(mask.dtype)

        img_num = imgmeta.num
        dataset_name = imgmeta.get('Dataset Name', '')

        if ALL_VALIDATION:
            is_validation = True
        else:
            is_validation = '_val' in img_path.stem
        role = 'val' if is_validation else 'trn'
        is_train = not is_validation

        cc, n_comps = ndimage.label(mask)

        rprops = measure.regionprops(cc, raw)

        for rp in tqdm.tqdm(rprops, position=1, leave=False, desc='Patches'):
            centroid = np.round(rp.centroid).astype(np.int64)  # Note: This centroid is in the global coordinate frame
            if rp.area < EC_MIN_AREA or rp.area > EC_MAX_AREA:
                logger.info(f'Skipping: area size {rp.area} not within [{EC_MIN_AREA}, {EC_MAX_AREA}]')
                continue  # Too small or too big (-> background component?) to be a normal particle
            circularity = np.nan
            if MIN_CIRCULARITY > 0:
                circularity = iu.calculate_circularity(rp.perimeter, rp.area)
                if circularity < MIN_CIRCULARITY:
                    logger.info(f'Skipping: circularity {circularity} below {MIN_CIRCULARITY}')
                    continue  # Not circular enough (probably a false merger)
                circularity = np.round(circularity, 2)  # Round for more readable logging

            lo = centroid - EC_REGION_RADIUS
            hi = centroid + EC_REGION_RADIUS + EC_REGION_ODD_PLUS1
            if np.any(lo < 0) or np.any(hi > raw.shape):
                logger.info(f'Skipping: region touches border')
                continue  # Too close to image border

            xslice = slice(lo[0], hi[0])
            yslice = slice(lo[1], hi[1])

            # Get enctype for specific position (for supporting multi-class images)
            enctype = utils.get_isplit_enctype(path=img_path, sheet_path=sheet_path, pos=tuple(centroid), isplitdata_root=isplitdata_root, role=role)

            if enctype == '?':
                logger.info(f'Skipping patch, can\'t determine local enctype: image {img_num=}, {role=}, pos={centroid}')
                continue

            raw_patch = raw[xslice, yslice]
            # mask_patch = mask[xslice, yslice]
            # For some reason mask[xslice, yslice] does not always contain nonzero values, but cc at the same slice does.
            # So we rebuild the mask at the region slice by comparing cc to 0
            mask_patch = cc[xslice, yslice] > 0

            # Eliminate coinciding masks from other particles that can overlap with this region (this can happen because we slice the mask_patch from the global mask)
            _mask_patch_cc, _ = ndimage.label(mask_patch)
            # Assuming convex particles, the center pixel is always on the actual mask region of interest.
            _local_center = np.round(np.array(mask_patch.shape) / 2).astype(np.int64)
            _mask_patch_centroid_label = _mask_patch_cc[tuple(_local_center)]
            # All mask_patch pixels that don't share the same cc label as the centroid pixel are set to 0
            mask_patch[_mask_patch_cc != _mask_patch_centroid_label] = 0

            if mask_patch.sum() == 0:
                # No positive pixel in mask -> skip this one
                logger.info(f'Skipping: no particle mask in region')
                # TODO: Why does this happen although we're iterating over regionprops from mask?
                # (Only happens if using `mask_patch = mask[xslice, yslice]`. Workaround: `Use mask_patch = cc[xslice, yslice] > 0`)
                continue

            area = int(np.sum(mask_patch))
            radius2 = np.round(measure_outer_disk_radius(mask_patch, discrete=False), 1)

            # Enlarge masks because we don't want to risk losing perimeter regions
            if DILATE_MASKS_BY > 0:
                disk = sm.disk(DILATE_MASKS_BY)
                # mask_patch = ndimage.binary_dilation(mask_patch, iterations=DILATE_MASKS_BY)
                mask_patch = sm.binary_dilation(mask_patch, footprint=disk)

            # Measure again after mask dilation
            area_dilated = int(np.sum(mask_patch))
            radius2_dilated = np.round(measure_outer_disk_radius(mask_patch, discrete=False), 1)

            # Raw patch with background erased via mask
            nobg_patch = raw_patch.copy()
            nobg_patch[mask_patch == 0] = 0

            # Concentric average image
            cavg_patch = concentric_average(raw_patch)

            raw_patch_fname = f'{patch_out_path}/raw/raw_patch_{patch_id:06d}.png'
            mask_patch_fname = f'{patch_out_path}/mask/mask_patch_{patch_id:06d}.png'
            nobg_patch_fname = f'{patch_out_path}/nobg/nobg_patch_{patch_id:06d}.png'
            cavg_patch_fname = f'{patch_out_path}/cavg/cavg_patch_{patch_id:06d}.png'

            patchmeta.append(PatchMeta(
                # patch_id=patch_id,
                patch_fname=os.path.basename(raw_patch_fname),
                img_num=img_num,
                dataset_name=dataset_name,
                enctype=enctype,
                centroid_y=centroid[0],
                centroid_x=centroid[1],
                corner_y=lo[0],
                corner_x=lo[1],
                train=is_train,
                validation=is_validation,
                radius2=radius2,
                radius2_dilated=radius2_dilated,
                area=area,
                area_dilated=area_dilated,
                circularity=circularity,
            ))

            iio.imwrite(raw_patch_fname, raw_patch.astype(np.uint8))
            iio.imwrite(mask_patch_fname, mask_patch.astype(np.uint8) * 255)
            iio.imwrite(nobg_patch_fname, nobg_patch.astype(np.uint8))
            iio.imwrite(cavg_patch_fname, cavg_patch.astype(np.uint8))
            patch_id += 1


    patchmeta = pd.DataFrame(
        patchmeta,
        columns=PatchMeta._fields,
    )
    patchmeta = patchmeta.convert_dtypes()
    patchmeta = patchmeta.astype({'img_num': int})  # Int64
    patchmeta.to_excel(f'{patch_out_path}/patchmeta.xlsx', index_label='patch_id')

    # individual_enctypes = patchmeta.enctype.unique()
    individual_enctypes = utils.CLASS_GROUPS['simple_hek']
    samples = []
    eval_samples = {}

    for role in ['train', 'validation']:
        n_samples = {}
        # Find condition with smallest number of patches
        min_n_samples = 1_000_000  # Unexpected high value for initialization
        min_scond = None
        # # Gather selected enctypes in simple form (without host)
        for scond in individual_enctypes:
            # if DRO_MODE:
            #     scond = scond.replace('DRO-', '')  # drop DRO because we want to treat DRO the same as HEK here  #DRO
            matching_patches = patchmeta[(patchmeta['enctype'] == scond) & patchmeta[role]]
            n_samples[scond] = len(matching_patches)
            print(f'({role}, {scond}) n_samples: {n_samples[scond]}')
            if n_samples[scond] <= min_n_samples:
                min_n_samples = n_samples[scond]
                min_scond = scond
        print(f'({role}) min_n_samples: {min_n_samples}, condition {min_scond}')

        # Sample min_scond patches each to create a balanced dataset
        scond_samples = {}
        for scond in individual_enctypes:
            # if DRO_MODE:
            #     scond = scond.replace('DRO-', '')  # drop DRO because we want to treat DRO the same as HEK here  #DRO
            # scond_samples[scond] = patchmeta[patchmeta['enctype'] == scond].sample(min_n_samples)
            matching_patches = patchmeta[(patchmeta['enctype'] == scond) & patchmeta[role]]
            scond_samples = matching_patches.sample(min_n_samples)
            samples.append(scond_samples)

            if role == 'validation':
                eval_samples[scond] = matching_patches.sample(N_EVAL_SAMPLES)

    samples = pd.concat(samples)

    all_samples = samples
    all_samples = all_samples.convert_dtypes()
    all_samples.to_excel(f'{patch_out_path}/patchmeta_traintest.xlsx', index_label='patch_id')

    print('Done with all_samples')


    shuffled_samples = pd.concat(eval_samples.values())
    shuffled_samples = shuffled_samples.sample(frac=1)  # Shuffle
    shuffled_samples.reset_index(inplace=True, drop=True)  # TODO: Avoid dropping index completely
    # # samples = samples[['patch_fname', 'enctype']]
    shuffled_samples.to_excel(f'{patch_out_path}/samples_gt.xlsx', index_label='patch_id')
    shuffled_samples[['enctype']].to_excel(f'{patch_out_path}/samples_blind.xlsx', index_label='patch_id')
    imgs = []
    _kind = 'nobg'  #  or 'raw'
    for entry in shuffled_samples.itertuples():
        srcpath = f'{patch_out_path}/{_kind}/{entry.patch_fname.replace("raw", _kind)}'
        shutil.copyfile(srcpath, f'{patch_out_path}/samples/{entry.Index:03d}.png')
        imgs.append(Image.open(srcpath).resize((28*4, 28*4), Image.Resampling.NEAREST))

    text_color = 255 if _kind == 'nobg' else 0

    # grid = image_grid(imgs, len(DATA_SELECTION), N_EVAL_SAMPLES, text_color=text_color)
    grid = image_grid(imgs, len(individual_enctypes) * 2, N_EVAL_SAMPLES // 2, text_color=text_color)
    grid.save(f'{patch_out_path}/samples_grid.png')


if __name__ == '__main__':
    main()
