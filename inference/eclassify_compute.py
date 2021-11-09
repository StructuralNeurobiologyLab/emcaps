"""
Script for the first step of ecapsulin classification (using GPU):

A trained neural network model (from tumtrain2d.py) is used to predict
segmentation masks and save raw data, segmentations, neural network feature
vectors and encapsulin type information into .npz files in the directory
`ec_results_path`.
"""

import os
from enum import Enum, auto
from pathlib import Path
from os.path import expanduser as eu

import gzip
import zstandard
import numpy as np
import imageio
import skimage
import torch
import tqdm
import pandas as pd

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



# Keep this in sync with training normalization
dataset_mean = (128.0,)
dataset_std = (128.0,)


invert_labels = True  # Workaround: Fixes inverted TIF loading


pre_predict_transform = transforms.Compose([
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
])

thresh = 127

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
    # emb.shape: batch, emb_dim, x, y, z (or z,y,x?)
    emb_flat = emb.swapaxes(1, -1).reshape((-1, emb.shape[1]))
    U, S, V = torch.pca_lowrank(emb_flat)

    V_3 = V[:, :3]
    emb_projected_pca = torch.tensordot(emb, V_3, ([1], [0]))
    emb_projected_pca = emb_projected_pca.moveaxis(-1, 1)
    # alternative #1: (emb.swapaxes(1, -1) @ (V_3).swapaxes(1, -1))[0]
    # alternative #2:
    # emb_projected_flat = torch.matmul(emb_flat, V[:, :3])
    # emb_projected_reshaped =   emb_projected_flat.reshape(  (emb.shape[0],  emb.shape[2], emb.shape[3], emb.shape[4], 3)).swapaxes(1,2).swapaxes(2,3)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
    import IPython; IPython.embed(); raise SystemExit
    ax.imshow(emb_projected_pca)
    plt.imsave('emb.png')
    plt.show()

    # emb_middle = get_middle(emb)
    # emb_middle_flat= #todo: tsne is expensive -> just on plotted (middle) slice
    # emb_tsne_flat = TSNE(n_components=3, learning_rate="auto").fit_transform(emb_flat) # todo: use tsnecuda package
    # emb_tsne_reshaped = emb_tsne_flat.reshape((emb.shape[0], emb.shape[2], emb.shape[3], emb.shape[4], 3)
    #                     ).swapaxes(1, 2).swapaxes(2, 3)
    return emb_projected_pca  # , emb_tsne_reshaped


# image_numbers = [22, 32, 42]
image_numbers = range(16, 54 + 1)

img_paths = eul([
    f'~/tumdata/{i}/{i}.tif' for i in image_numbers
])

label_name = 'encapsulins'

results_path = os.path.expanduser('~/tumresults')

ec_results_path = os.path.expanduser('~/tumecls')

model_variant = 'final.pt'
# model_variant = 'best.pt'
# model_variant = ''


model_paths = eul([
    f'~/tumtrainings/15to54_encapsulins__UNet__21-09-16_03-10-26/model_{model_variant}',
    # f'~/tumtrainings/D_15to54_encapsulins__UNet__21-09-16_04-02-24/model_{model_variant}',
    # f'~/tumtrainings/M___UNet__21-09-13_04-32-52/model_{model_variant}',
])

class FeatureExtractorModel(torch.nn.Module):
    def __init__(self, basemodel: UNet) -> None:
        super().__init__()
        self.basemodel = basemodel

    def forward(self, x):
        """Modified forward that returns the feature maps from the penultimate layer"""
        encoder_outs = []

        # Encoder pathway, save outputs for merging
        i = 0  # Can't enumerate because of https://github.com/pytorch/pytorch/issues/16123
        for module in self.basemodel.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
            i += 1

        # Decoding by UpConv and merging with saved outputs of encoder
        i = 0
        for module in self.basemodel.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
            i += 1

        # Skip final conv layer, return feature before classification layer instead
        feat = x
        ## x = self.basemodel.conv_final(x)
        return feat

masked_raw_regions = {'mx': [], 'qt': []}
masked_feat_regions = {'mx': [], 'qt': []}

def var(x):
    return np.var(x)

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

for model_path in model_paths:
    modelname = os.path.basename(os.path.dirname(model_path))
    is_distmap = modelname.startswith('D_')  # Distance transform regression, no output activation
    is_multi = modelname.startswith('M_')  # Multilabel model, trained with sigmoid output activation

    apply_softmax = not (is_distmap or is_multi)


    predictor = Predictor(
        model=model_path,
        device='cuda',
        # float16=True,
        transform=pre_predict_transform,
        # verbose=True,
        augmentations=3 if apply_softmax else None,
        apply_softmax=apply_softmax,
    )
    # First initialize with normal model...
    feature_extractor = Predictor(
        model=model_path,
        device='cuda',
        # float16=True,
        transform=pre_predict_transform,
        # verbose=True,
        # augmentations=3,
        apply_softmax=False,
    )
    # ... then replace model with feature extractor wrapper
    feature_extractor.model = FeatureExtractorModel(feature_extractor.model)

    for img_path in tqdm.tqdm(img_paths):
        enctype = get_enctype(img_path)
        inp = np.array(imageio.imread(img_path), dtype=np.float32)[None][None]  # (N=1, C=1, H, W)
        raw = inp[0][0]
        out = predictor.predict(inp)
        out = out.numpy()
        # TODO: Avoid redundant computation
        feat_th = feature_extractor.predict(inp)
        # emb_proj_pca = emb_to_rgb_pca(feat_th)
        feat = feat_th.numpy()
        basename = os.path.splitext(os.path.basename(img_path))[0]

        assert out.shape[1] == 2
        cout = out[0, 1]
        cout = (cout * 255.).astype(np.uint8)
        mask = cout > thresh

        ccfeats = []
        ccraws = []

        # Compress to save space
        # feat = feat.astype(np.float16)
        for c in range(feat.shape[1]):
            feat[0, c][mask == 0] = 0
        # raw = raw.astype(np.float16)
        raw[mask == 0] = 0

        # outputs[enctype]['mask'].append(mask)
        # outputs[enctype]['feat'].append(feat)
        # outputs[enctype]['raw'].append(raw)



        cc, n_comps = ndimage.label(mask)
        rp = measure.regionprops(cc, raw, extra_properties=[var])
        # Exclude background
        rp = rp[1:]
        rps[enctype].extend(rp)


        # patches = []
        # obj_slices = ndimage.find_objects(cc, max_label=n_comps)
        # for slc in obj_slices:
            # patch = 

        # mask_coords = np.argwhere(mask)
        # raw_pixs = raw_img[mask_coords]
        # feat_pixs = feat[mask_coords]
        
        # import IPython ; IPython.embed(); raise SystemExit

        # TODO
        # Get coordinates 
        # np.argwhere()
        # find objects: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html
        # cc, n_comps = ndimage.label(mask)



        # import IPython ; IPython.embed(); raise SystemExit
        # for i in range(n_comps):
        #     pass


        # masked_raw = raw_img.copy()
        # masked_raw[mask == 0] = 0
        # masked_raw_regions[enctype].append(masked_raw)
        # masked_feat = feat.copy()
        # masked_feat[mask == 0] = 0

        outfname = f'{ec_results_path}/{basename}.npz'
        np.savez_compressed(outfname, raw=raw, mask=mask, feat=feat, enctype=enctype)



#TODO MPI:
#- Cut out individual Encapsulin particles, centered on centroid and cropped so they are normalized in appearance -> Feed directly into UMAP, TSNE

# rdata = pd.DataFrame(lrdata, columns=['enctype', 'var', 'mean'])

# import IPython ; IPython.embed(); raise SystemExit

# rdata.to_hdf('rdata.h5', key='rdata', mode='w')

# import pickle
# print('Writing file')
# with zstandard.open('ec_outputs.pkl.zst', 'wb') as f:
    # pickle.dump(outputs, f)

# for enctype in outputs.keys():
#     for k in outputs[enctype].keys():
#         fname = f'{ec_results_path}/'

import IPython; IPython.embed(); exit()
