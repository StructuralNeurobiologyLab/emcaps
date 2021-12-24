"""
Script for cropping patches from the raw data with segmentation masks:

A trained neural network model (from tumtrain2d.py) is used to predict
segmentation masks and masks are used to crop 28x28 pixel patches centered
around particle centroids
"""
import os
from enum import Enum, auto
from pathlib import Path
from os.path import expanduser as eu
from typing import NamedTuple
import shutil

import gzip
import zstandard
import numpy as np
import imageio
import skimage
import torch
import tqdm
import pandas as pd
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


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('L', size=(cols * w, rows * h))
    
    for i, img in enumerate(imgs):
        drw = ImageDraw.Draw(img)
        drw.text((0, 0), f'{i:02d}')
        grid.paste(img, box=(i % cols * w, i // cols * h))
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

# NEGATIVE_SAMPLING = True
NEGATIVE_SAMPLING = False
N_NEG_PATCHES_PER_IMAGE = 100

EC_REGION_RADIUS = 14
EC_MIN_AREA = 200
EC_MAX_AREA = (2 * EC_REGION_RADIUS)**2

# DROSOPHILA_TEM_ONLY = False
DROSOPHILA_TEM_ONLY = True

USE_GT = False
# USE_GT = True


if DROSOPHILA_TEM_ONLY:
    sheet = pd.read_excel(
        os.path.expanduser('~/tum/Single-table_database/Image_annotation_for_ML_single_table.xlsx'),
        sheet_name='Image_origin_information'
    )
    valid_image_numbers = [
        40, 60, 114,  # MxEnc
        43, 106, 109,  # QtEnc
    ]
    try:
        sheet = sheet.rename(columns={'Image abbreviation': 'num'}).astype({'num': int})
    except KeyError:  # Unnamed column in single-table format
        sheet = sheet.rename(columns={' ': 'num'}).astype({'num': int})
    sheet = sheet.loc[sheet['1xMmMT3']]
    sheet = sheet.loc[sheet['Host organism'] == 'Drosophila']
    sheet = sheet.loc[sheet['Modality'] == 'TEM']
    # sheet = sheet[sheet['num'].isin(valid_image_numbers)]

    image_numbers = sheet['num'].to_list()

    _mxnums = sheet[sheet['MxEnc']]['num'].to_list()
    _qtnums = sheet[sheet['QtEnc']]['num'].to_list()

    print(f'MxEnc ({len(_mxnums)} images):\n {_mxnums}')
    print(f'QtEnc ({len(_qtnums)} images):\n  {_qtnums}')

    root_path = Path('~/tum/Single-table_database/').expanduser()

    img_paths = []
    for img_num in image_numbers:

        subdir_path = root_path / f'{img_num}'

        inp_path = subdir_path / f'{img_num}.tif'
        if not inp_path.exists():
            inp_path = subdir_path / f'{img_num}.TIF'
        inp = imageio.imread(inp_path).astype(np.float32)
        img_paths.append(inp_path)

    def get_enctype(path):
        if not isinstance(path, Path):
            path = Path(path)
        row = sheet.loc[sheet['num'] == int(path.stem)]
        if row['QtEnc'].item():
            return 'qt'
        elif row['MxEnc'].item():
            return 'mx'
        else:
            raise ValueError(row)

    patch_out_path = os.path.expanduser('~/tum/patches_v3_drosophila_only_tem')
    if USE_GT:
        patch_out_path = os.path.expanduser('~/tum/patches_v3_drosophila_only_tem__from_gt')

    model_variant = '57k.pt'

    model_paths = eul([
        f'~/tum/binary_mxqtsegtrain2_trainings/GDL_CE_B_GA___UNet__21-12-14_17-32-22/model_{model_variant}'
        # f'~/tumtrainings/15to54_encapsulins__UNet__21-09-16_03-10-26/model_{model_variant}',
    ])

else:

    # image_numbers = range(16, 54 + 1)
    # MXENC_NUMS = list(range(31, 40 + 1)) + list(range(51, 53 + 1))
    # QTENC_NUMS = list(range(16, 30 + 1)) + list(range(41, 50 + 1)) + [54]


    # Data v2: Include all 1xMmMT3 data (exclude only 1..15)

    image_numbers = range(16, 69 + 1)

    MXENC_NUMS = list(range(31, 40 + 1)) + list(range(51, 53 + 1)) + list(range(60, 69 + 1))
    QTENC_NUMS = list(range(16, 30 + 1)) + list(range(41, 50 + 1)) + list(range(54, 59 + 1))

    img_paths = eul([
        f'~/tumdata2/{i}/{i}.tif' for i in image_numbers
    ])

    def get_enctype(path: str) -> str:
        imgnum = int(os.path.basename(path)[:-4])
        if imgnum in MXENC_NUMS:
            return 'qt'
        elif imgnum in QTENC_NUMS:
            return 'mx'
        else:
            raise ValueError(f'Image {path} not found in any list')


    patch_out_path = os.path.expanduser('~/tum/patches_v2')
    if NEGATIVE_SAMPLING:
        patch_out_path = os.path.expanduser('~/tum/patches_v2neg')
    
    model_variant = 'final.pt'

    model_paths = eul([
        f'~/tumtrainings/15to54_encapsulins__UNet__21-09-16_03-10-26/model_{model_variant}',
        # f'~/tumtrainings/D_15to54_encapsulins__UNet__21-09-16_04-02-24/model_{model_variant}',
        # f'~/tumtrainings/M___UNet__21-09-13_04-32-52/model_{model_variant}',
    ])


for p in [patch_out_path, f'{patch_out_path}/raw', f'{patch_out_path}/mask', f'{patch_out_path}/samples']:
    os.makedirs(p, exist_ok=True)



masked_raw_regions = {'mx': [], 'qt': []}
masked_feat_regions = {'mx': [], 'qt': []}


MX = 0
QT = 1
ECODE = {'mx': 0, 'qt': 1}






outputs = {
    'mx': {
        'mask': [],
        'feat': [],
        'raw': [],
    },
    'qt': {
        'mask': [],
        'feat': [],
        'raw': [],
    },
}

rps = {'qt': [], 'mx': []}
lrdata = []

# meta = pd.DataFrame({
#     'id': pd.Series(dtype=pd.Int64Dtype),
#     'patch_fname': pd.Series(dtype=pd.StringDtype),
#     'enctype': pd.Series(dtype=pd.StringDtype),
#     'mmmt': pd.Series(dtype=pd.StringDtype),
#     'img_num': pd.Series(dtype=pd.StringDtype),
#     'centroid_x': pd.Series(dtype=pd.UInt32Dtype),
#     'centroid_y': pd.Series(dtype=pd.UInt32Dtype),
#     'corner_x': pd.Series(dtype=pd.UInt32Dtype),
#     'corner_y': pd.Series(dtype=pd.UInt32Dtype),
# })

class PatchMeta(NamedTuple):
    # patch_id: int
    patch_fname: str
    img_num: int
    enctype: str
    mmmt: str
    centroid_x: int
    centroid_y: int
    corner_x: int
    corner_y: int

meta = []

patch_id = 0  # Incremented below for each patch written to disk

for model_path in model_paths:
    modelname = os.path.basename(os.path.dirname(model_path))
    is_distmap = modelname.startswith('D_')  # Distance transform regression, no output activation
    is_multi = modelname.startswith('M_')  # Multilabel model, trained with sigmoid output activation

    apply_softmax = not (is_distmap or is_multi)


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
        if USE_GT:

            label_path = _img_path.with_name(f'{_img_path.stem}_encapsulins.tif')
            label = imageio.imread(label_path).astype(np.int64)
            if int(_img_path.stem) < 60:  # TODO: Investigate why labels are inverted although images look fine
                label = (label == 0).astype(np.int64)
            mask = label
        else:
            out = predictor.predict(inp)
            out = out.numpy()

            assert out.shape[1] == 2
            cout = out[0, 1]
            cout = (cout * 255.).astype(np.uint8)
            mask = cout > thresh

        img_num = int(os.path.splitext(os.path.basename(img_path))[0])

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

                raw_patch_fname = f'{patch_out_path}/raw/raw_patch_{patch_id:05d}.tif'


                mmmt = '1xMmMT3'  # TODO: Determine this from sheet when applicable

                meta.append(PatchMeta(
                    # patch_id=patch_id,
                    patch_fname=os.path.basename(raw_patch_fname),
                    img_num=img_num,
                    enctype=enctype,
                    mmmt=mmmt,
                    centroid_y=centroid[0],
                    centroid_x=centroid[1],
                    corner_y=lo[0],
                    corner_x=lo[1]
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

                # raw_patch_fname = f'{patch_out_path}/raw/raw_{enctype}_{patch_id:06d}.tif'
                # mask_patch_fname = f'{patch_out_path}/mask/mask{enctype}_{patch_id:06d}.tif'
                raw_patch_fname = f'{patch_out_path}/raw/raw_patch_{patch_id:05d}.tif'
                mask_patch_fname = f'{patch_out_path}/mask/mask_patch_{patch_id:05d}.tif'

                mmmt = '1xMmMT3'  # TODO: Determine this from sheet when applicable

                meta.append(PatchMeta(
                    # patch_id=patch_id,
                    patch_fname=os.path.basename(raw_patch_fname),
                    img_num=img_num,
                    enctype=enctype,
                    mmmt=mmmt,
                    centroid_y=centroid[0],
                    centroid_x=centroid[1],
                    corner_y=lo[0],
                    corner_x=lo[1]
                ))

                imageio.imwrite(raw_patch_fname, raw_patch.astype(np.uint8))
                imageio.imwrite(mask_patch_fname, mask_patch.astype(np.uint8) * 255)
                patch_id += 1


meta = pd.DataFrame(
    meta,
    columns=PatchMeta._fields,
)
meta = meta.convert_dtypes()
meta = meta.astype({'img_num': int})  # Int64
meta.to_excel(f'{patch_out_path}/patchmeta.xlsx', index_label='patch_id')


# Sample random images from qt and mx subsets
n_samples = 96
mx_samples = meta[meta.enctype == 'mx'].sample(n_samples // 2)
qt_samples = meta[meta.enctype == 'qt'].sample(n_samples // 2)
mixed_samples = pd.concat((mx_samples, qt_samples))
shuffled_samples = mixed_samples.sample(frac=1)  # Sampling with frac=1 shuffles rows
shuffled_samples.reset_index(inplace=True, drop=True)  # TODO: Avoid dropping index completely
# samples = samples[['patch_fname', 'enctype']]
shuffled_samples.to_excel(f'{patch_out_path}/samples_gt.xlsx', index_label='patch_id')
shuffled_samples[['enctype']].to_excel(f'{patch_out_path}/samples_blind.xlsx', index_label='patch_id')
imgs = []
for entry in shuffled_samples.itertuples():
    shutil.copyfile(f'{patch_out_path}/raw/{entry.patch_fname}', f'{patch_out_path}/samples/{entry.Index:02d}.tif')
    imgs.append(Image.open(f'{patch_out_path}/raw/{entry.patch_fname}').resize((28*4, 28*4), Image.NEAREST))

grid = image_grid(imgs, 8, 12)
grid.save(f'{patch_out_path}/samples_grid.png')


# Try to build a balanced patch dataset with train/test split

meta['train'] = True
meta['test'] = False

meta.iloc[mixed_samples.index, meta.columns.get_loc('test')] = True
meta.iloc[mixed_samples.index, meta.columns.get_loc('train')] = False

if meta[meta.enctype == 'mx'].shape[0] >= meta[meta.enctype == 'qt'].shape[0]:
    # There are more MxEnc examples than QtEnc samples
    # -> Do not train on all available mx patches to better balance classes
    _n_qt_train_samples = meta[(meta.enctype == 'qt') & (meta.test == False)].shape[0]
    _mx_train_samples = meta[meta.enctype == 'mx'].sample(_n_qt_train_samples)
    _mx_exclude_idxs = set(meta[meta.enctype == 'mx'].index) - set(_mx_train_samples.index)
    meta.loc[meta.index.isin(_mx_exclude_idxs), 'train'] = False

else:
    # There are more QtEnc examples than MxEnc samples
    # -> Do not train on all available qt patches to better balance classes
    _n_mx_train_samples = meta[(meta.enctype == 'mx') & (meta.test == False)].shape[0]
    _qt_train_samples = meta[meta.enctype == 'qt'].sample(_n_mx_train_samples)
    _qt_exclude_idxs = set(meta[meta.enctype == 'qt'].index) - set(_qt_train_samples.index)
    meta.loc[meta.index.isin(_qt_exclude_idxs), 'train'] = False

meta = meta.convert_dtypes()

meta.to_excel(f'{patch_out_path}/patchmeta_traintest.xlsx', index_label='patch_id')

import IPython; IPython.embed(); exit()
