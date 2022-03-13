"""
Script for cropping patches from the raw data with segmentation masks:

A trained neural network model (from mxqtsegtrain2.py) is used to predict
segmentation masks and masks are used to crop 28x28 pixel patches centered
around particle centroids
"""
import os
from enum import Enum, auto
from pathlib import Path
from os.path import expanduser as eu
from random import shuffle
from re import S
from typing import NamedTuple
import shutil

import gzip
from unittest.mock import patch
import zstandard
import numpy as np
import imageio
import skimage
import torch
import tqdm
import pandas as pd
import yaml

from PIL import Image, ImageDraw

from scipy import ndimage
from skimage import morphology as sm
from skimage.color import label2rgb
from skimage import measure
from sklearn import metrics as sme

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from elektronn3.inference import Predictor
from elektronn3.data import transforms
from elektronn3.models.unet import UNet

# torch.backends.cudnn.benchmark = True

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


np.random.seed(0)

# Keep this in sync with training normalization
dataset_mean = (128.0,)
dataset_std = (128.0,)


invert_labels = True  # Workaround: Fixes inverted TIF loading


pre_predict_transform = transforms.Compose([
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
])

thresh = 127

N_EVAL_SAMPLES = 30

# NEGATIVE_SAMPLING = True
NEGATIVE_SAMPLING = False
N_NEG_PATCHES_PER_IMAGE = 100

# EC_REGION_RADIUS = 14
EC_REGION_RADIUS = 18

EC_MIN_AREA = 200
EC_MAX_AREA = (2 * EC_REGION_RADIUS)**2

# DROSOPHILA_TEM_ONLY = False
DROSOPHILA_TEM_ONLY = True

# USE_GT = False
USE_GT = True

FILL_HOLES = True
DILATE_MASKS_BY = 3

sheet = pd.read_excel(
    os.path.expanduser('~/tum/Single-table_database/Image_annotation_for_ML_single_table.xlsx'),
    sheet_name='all_metadata'
)
# valid_image_numbers = [
#     40, 60, 114,  # MxEnc
#     43, 106, 109,  # QtEnc
# ]

meta = sheet.copy()
meta = meta.rename(columns={' Image': 'num'})
meta = meta.loc[meta['num'] >= 16]
meta = meta.rename(columns={'Short experimental condition': 'scond'})
meta = meta.convert_dtypes()


DATA_SELECTION = [
    # 'DRO_1xMT3-MxEnc-Flag-NLS',
    # 'DRO_1xMT3-QtEnc-Flag-NLS',
    'HEK_1xMT3-MxEnc-Flag',
    'HEK_1xMT3-QtEnc-Flag',
    'HEK-2xMT3-MxEnc-Flag',
    'HEK-2xMT3-QtEnc-Flag',
    'HEK-3xMT3-QtEnc-Flag',
    'HEK-1xTmEnc-BC2-Tag',  # -> bad, requires extra model?
]

# Load mapping from class names to class IDs
class_info_path = './class_info.yaml'
with open(class_info_path) as f:
    class_info = yaml.load(f, Loader=yaml.FullLoader)
class_ids = class_info['class_ids']
class_names = {v: k for k, v in class_ids.items()}

# Load train/validation split
valid_split_path = './valid_split.yaml'
with open(valid_split_path) as f:
    valid_image_dict = yaml.load(f, Loader=yaml.FullLoader)

valid_image_numbers = []
for condition in DATA_SELECTION:
    valid_image_numbers.extend(valid_image_dict[condition])

# image_numbers = valid_image_numbers  # validation images, held out from training data


image_numbers = meta['num'].to_list()

conditions = meta['scond'].unique()
for condition in conditions:
    _nums = meta.loc[meta['scond'] == condition]['num'].to_list()
    print(f'{condition}:\t({len(_nums)} images):\n {_nums}')


root_path = Path('~/tum/Single-table_database/').expanduser()

img_paths = []
for img_num in image_numbers:

    subdir_path = root_path / f'{img_num}'

    inp_path = subdir_path / f'{img_num}.tif'
    if not inp_path.exists():
        inp_path = subdir_path / f'{img_num}.TIF'
    inp = imageio.imread(inp_path).astype(np.float32)
    img_paths.append(inp_path)


def get_meta_row(path):
    if not isinstance(path, Path):
        path = Path(path)
    row = meta[meta['num'] == int(path.stem)]
    return row

def get_enctype(path):
    row = get_meta_row(path)
    return row.scond.item()

def is_for_validation(path):
    row = get_meta_row(path)
    return row.Validation.item()


# TODO: Migrate to scratch storage?

# patch_out_path = os.path.expanduser('~/tum/patches_v4_uni')
# if USE_GT:
#     # patch_out_path = os.path.expanduser('~/tum/patches_v4_uni__from_gt_notmenc')
#     patch_out_path = os.path.expanduser('~/tum/patches_v4_uni__from_gt_a')

patch_out_path = os.path.expanduser('/wholebrain/scratch/mdraw/tum/patches_v4_uni')
if USE_GT:
    patch_out_path = os.path.expanduser('/wholebrain/scratch/mdraw/tum/patches_v4_uni__from_gt_notmenc')
    # patch_out_path = os.path.expanduser('/wholebrain/scratch/mdraw/tum/patches_v4_uni__from_gt_a')



model_paths = eul([
    f'~/tum/mxqtsegtrain2_trainings_uni_v4/GDL_CE_B_GA___UNet__22-02-26_02-02-13/model_step150000.pt'
    # f'~/tum/mxqtsegtrain2_trainings_uni/GDL_CE_B_GA_nb5__UNet__22-02-23_02-32-41/model_step80000.pt'
    # f'~/tum/mxqtsegtrain2_trainings_uni/GDL_CE_B_GA___UNet__22-02-21_05-30-56/model_step40000.pt',
])

for p in [patch_out_path, f'{patch_out_path}/raw', f'{patch_out_path}/mask', f'{patch_out_path}/samples', f'{patch_out_path}/nobg']:
    os.makedirs(p, exist_ok=True)


class PatchMeta(NamedTuple):
    # patch_id: int
    patch_fname: str
    img_num: int
    enctype: str
    centroid_x: int
    centroid_y: int
    corner_x: int
    corner_y: int
    train: bool
    validation: bool

patchmeta = []

patch_id = 0  # Incremented below for each patch written to disk

for model_path in model_paths:
    modelname = os.path.basename(os.path.dirname(model_path))

    apply_softmax = True
    predictor = Predictor(
        model=model_path,
        device='cuda',
        float16=True,
        transform=pre_predict_transform,
        # verbose=True,
        augmentations=3 if apply_softmax else None,
        apply_softmax=apply_softmax,
    )

    for img_path in tqdm.tqdm(img_paths, position=0, desc='Image'):
        enctype = get_enctype(img_path)
        img_path = Path(img_path)
        _img_path = Path(img_path)


        if not img_path.is_file():
            img_path = img_path.with_suffix('.TIF')
        img_path = str(img_path)
        inp = np.array(imageio.imread(img_path), dtype=np.float32)[None][None]  # (N=1, C=1, H, W)
        raw = inp[0][0]
        if USE_GT and is_for_validation(img_path):

            label_path = _img_path.with_name(f'{_img_path.stem}_encapsulins.tif')
            label = imageio.imread(label_path).astype(np.int64)
            if int(_img_path.stem) < 55:  # TODO: Investigate why labels are inverted although images look fine
                label = (label == 0).astype(np.int64)
            mask = label
        else:
            out = predictor.predict(inp)
            out = out.numpy()

            assert out.shape[1] == 2
            cout = out[0, 1]
            cout = (cout * 255.).astype(np.uint8)
            mask = cout > thresh

        if DILATE_MASKS_BY > 0:
            mask = ndimage.binary_dilation(mask, iterations=DILATE_MASKS_BY)

        if FILL_HOLES:
            mask = ndimage.binary_fill_holes(mask).astype(mask.dtype)

        img_num = int(os.path.splitext(os.path.basename(img_path))[0])

        is_validation = img_num in valid_image_numbers
        is_train = not is_validation

        if NEGATIVE_SAMPLING:
            n_patches_from_this_image = 0
            while n_patches_from_this_image < N_NEG_PATCHES_PER_IMAGE:
                shx, shy = mask.shape
                sh = np.array(mask.shape)
                lo = np.random.randint(0, sh - 2 * EC_REGION_RADIUS, 2)
                hi = lo + 2 * EC_REGION_RADIUS
                centroid = lo + EC_REGION_RADIUS

                xslice = slice(lo[0], hi[0])
                yslice = slice(lo[1], hi[1])

                mask_patch = mask[xslice, yslice]
                if np.count_nonzero(mask_patch) > 0:
                    # Skip if any pixel in this patch overlaps with an encapsulin particle mask
                    continue
                raw_patch = raw[xslice, yslice]

                raw_patch_fname = f'{patch_out_path}/raw/raw_patch_{patch_id:06d}.tif'


                patchmeta.append(PatchMeta(
                    # patch_id=patch_id,
                    patch_fname=os.path.basename(raw_patch_fname),
                    img_num=img_num,
                    enctype=enctype,
                    centroid_y=centroid[0],
                    centroid_x=centroid[1],
                    corner_y=lo[0],
                    corner_x=lo[1],
                    train=is_train,
                    validation=is_validation,
                ))

                imageio.imwrite(raw_patch_fname, raw_patch.astype(np.uint8))
                patch_id += 1

                n_patches_from_this_image += 1

        else:  # Normal positive sampling
            cc, n_comps = ndimage.label(mask)

            rprops = measure.regionprops(cc, raw)


            for rp in tqdm.tqdm(rprops, position=1, leave=False, desc='Patches'):
                centroid = np.round(rp.centroid).astype(np.int64)
                if rp.area < EC_MIN_AREA or rp.area > EC_MAX_AREA:
                    continue  # Too small or too big (-> background component?) to be a normal particle
                lo = centroid - EC_REGION_RADIUS
                hi = centroid + EC_REGION_RADIUS
                if np.any(lo < 0) or np.any(hi > raw.shape):
                    continue  # Too close to image border

                xslice = slice(lo[0], hi[0])
                yslice = slice(lo[1], hi[1])

                raw_patch = raw[xslice, yslice]
                mask_patch = mask[xslice, yslice]
                # Raw patch with background erased via mask
                nobg_patch = raw_patch.copy()
                nobg_patch[mask_patch == 0] = 0

                # raw_patch_fname = f'{patch_out_path}/raw/raw_{enctype}_{patch_id:06d}.tif'
                # mask_patch_fname = f'{patch_out_path}/mask/mask{enctype}_{patch_id:06d}.tif'
                raw_patch_fname = f'{patch_out_path}/raw/raw_patch_{patch_id:06d}.tif'
                mask_patch_fname = f'{patch_out_path}/mask/mask_patch_{patch_id:06d}.tif'
                nobg_patch_fname = f'{patch_out_path}/nobg/nobg_patch_{patch_id:06d}.tif'

                patchmeta.append(PatchMeta(
                    # patch_id=patch_id,
                    patch_fname=os.path.basename(raw_patch_fname),
                    img_num=img_num,
                    enctype=enctype,
                    centroid_y=centroid[0],
                    centroid_x=centroid[1],
                    corner_y=lo[0],
                    corner_x=lo[1],
                    train=is_train,
                    validation=is_validation,
                ))

                imageio.imwrite(raw_patch_fname, raw_patch.astype(np.uint8))
                imageio.imwrite(mask_patch_fname, mask_patch.astype(np.uint8) * 255)
                imageio.imwrite(nobg_patch_fname, nobg_patch.astype(np.uint8) * 255)
                patch_id += 1


patchmeta = pd.DataFrame(
    patchmeta,
    columns=PatchMeta._fields,
)
patchmeta = patchmeta.convert_dtypes()
patchmeta = patchmeta.astype({'img_num': int})  # Int64
patchmeta.to_excel(f'{patch_out_path}/patchmeta.xlsx', index_label='patch_id')

samples = []

eval_samples = {}

for role in ['train', 'validation']:
    n_samples = {}
    # Find condition with smallest number of patches
    min_n_samples = 1_000_000  # Unexpected high value for initialization
    min_scond = None
    for scond in DATA_SELECTION:
        matching_patches = patchmeta[(patchmeta['enctype'] == scond) & patchmeta[role]]
        n_samples[scond] = len(matching_patches)
        print(f'({role}, {scond}) n_samples: {n_samples[scond]}')
        if n_samples[scond] <= min_n_samples:
            min_n_samples = n_samples[scond]
            min_scond = scond
    print(f'({role}) min_n_samples: {min_n_samples}, condition {min_scond}')

    # Sample min_scond patches each to create a balanced dataset
    scond_samples = {}
    for scond in DATA_SELECTION:
        # scond_samples[scond] = patchmeta[patchmeta['enctype'] == scond].sample(min_n_samples)
        matching_patches = patchmeta[(patchmeta['enctype'] == scond) & patchmeta[role]]
        scond_samples = matching_patches.sample(min_n_samples)
        samples.append(scond_samples)

        if role == 'validation':
            eval_samples[scond] = matching_patches.sample(N_EVAL_SAMPLES)
    # import IPython ; IPython.embed()
    # samples[role] = pd.concat(scond_samples.values())

samples = pd.concat(samples)

all_samples = samples
all_samples = all_samples.convert_dtypes()
all_samples.to_excel(f'{patch_out_path}/patchmeta_traintest.xlsx', index_label='patch_id')


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
    shutil.copyfile(srcpath, f'{patch_out_path}/samples/{entry.Index:03d}.tif')
    imgs.append(Image.open(srcpath).resize((28*4, 28*4), Image.NEAREST))

text_color = 255 if _kind == 'nobg' else 0

grid = image_grid(imgs, len(DATA_SELECTION), N_EVAL_SAMPLES, text_color=text_color)
grid.save(f'{patch_out_path}/samples_grid.png')



import IPython; IPython.embed(); exit()
