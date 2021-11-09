"""
Encapsulin classification analysis script.

Requires output of eclassify_compute.py in `ec_out_path`.


Uses segmentation results to crop 28x28 pixel patches that each have
one encapsulin particle (connected component) in the center.
Patches are saved to `patch_out_path`. The encapsulin type (QtEnc vs MxEnc)
is reflected in the file names.

Additionally, the patches are analysed to find if there are unsupervised
clustering methods with which the different types can be distinguished:

- Feeding flat patch pixel intensity vectors into 3D UMAP
- Plotting mean vs std of the patch pixel intensities
- Plotting eccentricity vs area of patch pixel intensities

Each experiment shows patches of the different types in different colors:
MxEnc in red, QtEnc in blue.

Currently works just on the initial version of the dataset, including just images
from number 16 to 54.
"""

import itertools
import os
import pickle
import time
from enum import Enum, auto
from pathlib import Path
from os.path import expanduser as eu

import gzip
# import zstandard
import numpy as np
import imageio
import skimage
import torch
import tqdm
import pandas as pd

from scipy import ndimage
from skimage import morphology as sm
from skimage import measure
from skimage.color import label2rgb
from sklearn import metrics as sme

import matplotlib.pyplot as plt

import seaborn as sns
from babyplots import Babyplot


import umap


sns.set_theme()


# torch.backends.cudnn.benchmark = True

def eul(paths):
    """Shortcut for expanding all user paths in a list"""
    return [os.path.expanduser(p) for p in paths]



# Keep this in sync with training normalization
dataset_mean = (128.0,)
dataset_std = (128.0,)


invert_labels = True  # Workaround: Fixes inverted TIF loading



EC_MIN_AREA = 200
EC_REGION_RADIUS = 14

multi_label_names = {
    0: 'background',
    1: 'membranes',
    2: 'encapsulins',
    3: 'nuclear_membrane',
    4: 'nuclear_region',
    5: 'cytoplasmic_region',
}

# class EncType:
#     QT = auto()
#     MX = auto()

MX = 0
QT = 1
ECODE = {'mx': 0, 'qt': 1}

MXENC_NUMS = list(range(31, 40 + 1)) + list(range(51, 53 + 1))
QTENC_NUMS = list(range(16, 30 + 1)) + list(range(41, 50 + 1)) + [54]


def get_enctype(path: str) -> str:
    imgnum = int(os.path.basename(path)[:-4])
    if imgnum in MXENC_NUMS:
        return 'qt'
    elif imgnum in QTENC_NUMS:
        return 'mx'
    else:
        raise ValueError(f'Image {path} not found in any list')


def emb_to_rgb_pca(emb):
    emb = torch.as_tensor(emb)
    emb_flat = emb.swapaxes(1, -1).reshape((-1, emb.shape[1]))
    # Get top 3 principal components
    u, s, v = torch.pca_lowrank(emb_flat, 3)

    emb_projected_pca = torch.tensordot(emb, v, ([1], [0]))
    emb_projected_pca = emb_projected_pca[0]  # Remove batch dimension -> (H, W, C=3)
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
    epp = emb_projected_pca.numpy()
    # Normalized projection
    nepp = (epp - epp.min()) / epp.ptp()
    ax.imshow(nepp)
    plt.imsave('emb.png', nepp)
    plt.show()
    return emb_projected_pca


# image_numbers = [22, 32, 42]
image_numbers = range(16, 54 + 1)

img_paths = eul([
    f'~/tumdata/{i}/{i}.tif' for i in image_numbers
])

label_name = 'encapsulins'

# results_path = os.path.expanduser('~/tumresults')


masked_raw_regions = {'mx': [], 'qt': []}
masked_feat_regions = {'mx': [], 'qt': []}

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

# ec_out_path = os.path.expanduser('~/tumresults/ec_outputs_lite.pkl.zst')
# ec_out_path = os.path.expanduser('~/tumresults/ec_outputs.pkl.zst')

ec_out_path = os.path.expanduser('~/tumecls')

vis_path = os.path.expanduser('~/tumvis')

patch_out_path = os.path.expanduser('~/tumpatches')

for p in [vis_path, patch_out_path]:
    os.makedirs(p, exist_ok=True)

# with zstandard.open(ec_out_path, 'rb') as f:
#     ec_outputs = pickle.load(f)

# mx = ec_outputs['mx']
# qt = ec_outputs['qt']


def normalize_zero_one_no_target(inp):
    eps = 1e-10
    inp_min = inp.min()
    inp_max = inp.max()
    inp = (inp - inp_min) / (inp_max - inp_min + eps)
    # assert inp.min() >= 0.0 and inp.max() <= 1.0
    return inp


def emb_to_rgb_umap(emb):  # todo: run on sep. process on cpu, takes ~20 s
    start_time = time.time()
    while emb.ndim < 4:
        emb = emb[None]
    emb_flat = emb.swapaxes(1, -1).reshape((-1, emb.shape[1]))
    emb_umap_flat = umap.UMAP(n_components=3).fit_transform(emb_flat)  # todo: use cuml not umap
    import IPython ; IPython.embed(); raise SystemExit
    emb_umap_reshaped = emb_umap_flat.reshape(
        (emb.shape[0], emb.shape[2], emb.shape[3], 3)
    ).swapaxes(-1, 1)
    emb_umap_normalized = normalize_zero_one_no_target(emb_umap_reshaped)

    print(f"umap took {time.time() - start_time}")
    import IPython ; IPython.embed(); raise SystemExit

    # fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
    # ax.imshow(emb_umap_normalized[0].swapaxes(0, -1))
    # ax.imshow(normalize_zero_one_no_target(emb_pca(emb, d=3))[0].swapaxes(0, -1))
    # plt.show()
    return emb_umap_normalized

def var(x):
    return np.var(x)

# Raw region props
rps = {
    'mx': [],
    'qt': [],
}
# Feature region props
fps = {
    'mx': [],
    'qt': [],
}

# raw_patches = {
#     'mx': [],
#     'qt': [],
# }
# feat_patches = {
#     'mx': [],
#     'qt': [],
# }
raw_patches = []
feat_avgs = []
ecodes = []
enctypes = []

geom = {
    'mx': [],
    'qt': [],
}

for num in tqdm.tqdm(image_numbers):
    fpath = f'{ec_out_path}/{num}.npz'
    npz = np.load(fpath)
    raw = npz['raw']
    mask = npz['mask']
    feat = npz['feat']
    enctype = str(npz['enctype'])

    cc, n_comps = ndimage.label(mask)

    rprops = measure.regionprops(cc, raw, extra_properties=[var])
    rprops = [rp for rp in rprops if rp.area >= EC_MIN_AREA]

    # TODO: SHAPES
    # fprops = measure.regionprops(cc, feat, extra_properties=[var])
    # fprops = [fp for fp in fprops if fp.area >= EC_MIN_AREA]

    roi_slices = []
    for rp in rprops:
        centroid = np.round(rp.centroid).astype(np.int64)
        if rp.area < EC_MIN_AREA:
            continue  # Too small to be a normal particle
        lo = centroid - EC_REGION_RADIUS
        hi = centroid + EC_REGION_RADIUS
        if np.any(lo < 0) or np.any(hi > raw.shape):
            continue  # Too close to image border

        xslice = slice(lo[0], hi[0])
        yslice = slice(lo[1], hi[1])
        roi_slices.append((xslice, yslice))

        geom[enctype].append((rp.area, rp.eccentricity))
        
        # plt.scatter(rp.eccentricity, rp.area, c=ECODE[enctype], label=ECODE[enctype], cmap='Spectral')

    # raw_patches = []
    for slc in roi_slices:
        raw_patch = raw[slc]
        feat_patch = feat[0, :, slc[0], slc[1]]  # (N, C, H, W) -> (C, h, w)
        # Use global average pooling (mean over all spatial axes) to arrive at nonspatial output vector
        feat_avg = np.mean(feat_patch, axis=(1, 2))  # (C, h, w) -> (C,)
        # raw_patches[enctype].append(raw_patch)
        # feat_avgs[enctype].append(feat_patch)
        ecode = ECODE[enctype]
        raw_patches.append(raw_patch)
        feat_avgs.append(feat_avg)
        ecodes.append(ecode)
        enctypes.append(enctype)


print('Writing patches...')
for i, (p, t) in enumerate(zip(raw_patches, enctypes)):
    imageio.imwrite(f'{patch_out_path}/{t}_{i:04d}.tif', p.astype(np.uint8))

raw_patches = np.array(raw_patches)
feat_avgs = np.array(feat_avgs)
ecodes = np.array(ecodes)

raw_patches_flat = raw_patches.reshape(raw_patches.shape[0], -1)



print('UMAP...')

sweep_n_neigbors = [5, 15, 25]
sweep_min_dist = [0.0, 0.1, 0.5]
kinds = ['raw', 'feat']

for kind, n_neighbors, min_dist in itertools.product(kinds, sweep_n_neigbors, sweep_min_dist):
    if kind == 'raw':
        inp = raw_patches_flat
    elif kind == 'feat':
        inp = feat_avgs
    reducer = umap.UMAP(n_components=3)
    um = reducer.fit_transform(inp)
    # plt.scatter(urp[:, 0], urp[:, 1], c=ecodes, cmap='Spectral')

    bp = Babyplot()
    bp.add_plot(um.tolist(), "shapeCloud", "categories", ecodes.tolist(), {
            "shape": "sphere",
            "colorScale": "Spectral",
            "showAxes": [True, True, True],
            "axisLabels": ["x", "y", "z"]
    })
    bp.save_as_html(f'{vis_path}/{kind}_umap_n{n_neighbors}_d{min_dist:.1f}.html', True)

# import IPython ; IPython.embed(); raise SystemExit

import plotly.express as px


# fig3d = px.scatter_3d(umr, x=0, y=1, z=2, color=ecodes)
# fig3d.update_traces(marker_size=5)
# fig3d.show()

print('Mean vs. std...')

# Plot mean against variance of masked raw patches
means = np.mean(raw_patches_flat, axis=1)
stds = np.std(raw_patches_flat, axis=1)
fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
ax.scatter(means, stds, c=ecodes, cmap='Spectral', label=ecodes)
ax.set_title('Mean, std of raw pixel intensities')
ax.set_xlabel('mean')
ax.set_ylabel('std')
# plt.show()
plt.savefig(f'{vis_path}/raw_mean_std.png')

m = np.array(geom['mx'])
q = np.array(geom['qt'])

fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
ax.scatter(*m.T, c='red')
ax.scatter(*q.T, c='blue')
ax.set_title('Geometric properties of masked regions')
ax.set_xlabel('area')
ax.set_ylabel('eccentricity')
# plt.show()
plt.savefig(f'{vis_path}/geom.png')


import IPython ; IPython.embed(); raise SystemExit


# for enctype, contents in ec_outputs.items():
#     masks = contents['mask']
#     feats = contents['feat']
#     raws = contents['raw']

#     for i, (mask, feat, raw) in tqdm.tqdm(enumerate(zip(masks, feats, raws)), total=len(masks)):
#         mask = mask.astype(np.float32)
#         feat = feat.astype(np.float32)

#         cc, n_comps = ndimage.label(mask)
#         rp = measure.regionprops(cc, raw, extra_properties=[var])
#         # Exclude background
#         rp = rp[1:]
#         rps[enctype].extend(rp)
#         # fp = measure.regionprops(cc, feat, extra_properties=[var])
#         # fps[enctype].extend(fp)

#         # import IPython ; IPython.embed(); raise SystemExit

# # rdata = pd.DataFrame(columns=['enctype', 'var', 'mean'])
# lrdata = []

# fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
# for enctype in rps.keys():
#     for rp in rps[enctype]:
#         var = rp.var
#         mean = rp.mean_intensity

#         lrdata.append((enctype, var, mean))

# rdata = pd.DataFrame(lrdata, columns=['enctype', 'var', 'mean'])

# sns.scatterplot(data=rdata, x='var', y='mean', hue='enctype')
# plt.show()
# plt.savefig('rvarmean.png')

# import IPython ; IPython.embed(); raise SystemExit


# # mask_coords = np.argwhere(mask)
# # raw_pixs = raw_img[mask_coords]
# # feat_pixs = feat[mask_coords]

# # import IPython ; IPython.embed(); raise SystemExit

# # TODO
# # Get coordinates 
# # np.argwhere()
# # find objects: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html
# # cc, n_comps = ndimage.label(mask)
# # import IPython ; IPython.embed(); raise SystemExit
# # for i in range(n_comps):
# #     pass


# # masked_raw = raw_img.copy()
# # masked_raw[mask == 0] = 0
# # masked_raw_regions[enctype].append(masked_raw)
# # masked_feat = feat.copy()
# # masked_feat[mask == 0] = 0

