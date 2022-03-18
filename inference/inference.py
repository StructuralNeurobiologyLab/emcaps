"""Inference script for segmentation prediction and evaluation, based on
raw images and a model trained with `tumtrain2d.py`."""


import os
from pathlib import Path
from os.path import expanduser as eu

import numpy as np
import imageio
import skimage
import torch
import yaml

from skimage import morphology as sm
from skimage.color import label2rgb
from sklearn import metrics as sme

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from elektronn3.inference import Predictor
from elektronn3.data import transforms

# torch.backends.cudnn.benchmark = True

def eul(paths):
    """Shortcut for expanding all user paths in a list"""
    return [os.path.expanduser(p) for p in paths]



# Keep this in sync with training normalization
dataset_mean = (128.0,)
dataset_std = (128.0,)


# TODO: Rewrite as invert_labels = False and flip meaning of ENABLE_TERRIBLE_INVERSION_HACK so it triggers on num < 60 (see YTifDirData2d implementation)

invert_labels = True  # Workaround: Fixes inverted TIF loading

ENABLE_TERRIBLE_INVERSION_HACK = True  # Another hacky inversion fix
# ENABLE_TERRIBLE_INVERSION_HACK = False

pre_predict_transform = transforms.Compose([
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
])

thresh = 127
dt_thresh = 0.00
multi_thresh = 100

multi_label_names = {
    0: 'background',
    1: 'membranes',
    2: 'encapsulins',
    3: 'nuclear_membrane',
    4: 'nuclear_region',
    5: 'cytoplasmic_region',
}

# DATA_SOURCE = 'tumdata_v1'
# DATA_SOURCE = 'droso_tem'
# DATA_SOURCE = 'hek-only'
DATA_SOURCE = 'uni'  # universal model for all conditions


if DATA_SOURCE == 'tumdata_v1':
    # image_numbers = [22, 32, 42]  # validation images, held out from training data
    image_numbers = range(16, 70 + 1)  # all images from 1xMmMT3 subset (also including training data!), exluding first 15 images

    img_paths = eul([
        f'~/tumdata/{i}/{i}.tif' for i in image_numbers
    ])
    results_path = Path('~/tum/results_tumdata_v1').expanduser()
elif DATA_SOURCE == 'hek-only':
    image_numbers = [22]  # validation images, held out from training data

    img_paths = eul([
        f'~/tumdata/{i}/{i}.tif' for i in image_numbers
    ])
    results_path = Path('~/tum/results_hek_only').expanduser()
elif DATA_SOURCE == 'droso_tem':
    img_paths = []
    data_root = Path('~/tum/Drosophila-only-TEM-only_annotation/').expanduser()
    for dir in data_root.iterdir():
        if dir.is_dir():  # Exlude actual files
            img_path = dir / f'{dir.name}.tif'
            if not img_path.exists():
                img_path = img_path.with_suffix('.TIF')
            if not img_path.exists() and dir.name == '94':  # Handle strange name in sample 94
                img_path = img_path = dir / 'EM2021_33a_R2Box26_02_05_51_88.tif'

            img_path = str(img_path)  # Code below expects strings
            img_paths.append(img_path)
    # results_path = Path('~/tum/results_droso_tem_oldmodel').expanduser()
    results_path = Path('~/tum/results_droso_tem_new_binary_model_gdl_ce_ga_63k').expanduser()
elif DATA_SOURCE == 'uni':
    # Select validation images from all experimental conditions
    results_path = Path('/wholebrain/scratch/mdraw/tum/results_uni_model_v4_training').expanduser()
    data_root = Path('/wholebrain/scratch/mdraw/tum/Single-table_database/').expanduser()

    #v3
    # valid_image_dict = {
    #     'DRO_1xMT3-MxEnc-Flag-NLS': [40, 60, 114],
    #     'DRO_1xMT3-QtEnc-Flag-NLS': [43, 106, 109],
    #     'HEK_1xMT3-QtEnc-Flag':     [26],
    #     'HEK_1xMT3-MxEnc-Flag':     [66],
    #     'HEK-2xMT3-QtEnc-Flag':     [125],
    #     'HEK-2xMT3-MxEnc-Flag':     [142],
    #     'HEK-3xMT3-QtEnc-Flag':     [131],
    #     'HEK-1xTmEnc-BC2-Tag':      [139],
    # }
    valid_split_path = './valid_split.yaml'
    with open(valid_split_path) as f:
        valid_image_dict = yaml.load(f, Loader=yaml.FullLoader)
    DATA_SELECTION = [
        'HEK_1xMT3-QtEnc-Flag',
        'DRO_1xMT3-MxEnc-Flag-NLS',
        'DRO_1xMT3-QtEnc-Flag-NLS',
        'HEK_1xMT3-MxEnc-Flag',
        'HEK-2xMT3-QtEnc-Flag',
        'HEK-2xMT3-MxEnc-Flag',
        'HEK-3xMT3-QtEnc-Flag',
        'HEK-1xTmEnc-BC2-Tag',  # -> bad, requires extra model?
    ]

    valid_image_numbers = []
    for condition in DATA_SELECTION:
        valid_image_numbers.extend(valid_image_dict[condition])

    image_numbers = valid_image_numbers  # validation images, held out from training data

    # image_numbers = [139, 147, 148, 149, 150, 151, 152]  # extra tmenc set

    # Training set
    image_numbers = set(list(range(16, 138 + 1)) + list(range(141, 152 + 1)))
    image_numbers = image_numbers - set(valid_image_numbers)
    print('Selected image numbers:', image_numbers)

    img_paths = [
        str(data_root / f'{i}/{i}.tif') for i in image_numbers
    ]

    # for p in [results_path / scond for scond in DATA_SELECTION]:
    #     os.makedirs(p, exist_ok=True)
else:
    raise ValueError(f'{DATA_SOURCE=}')

DESIRED_OUTPUTS = [
    'raw',
    'thresh',
    'lab',
    'overlays',
    'error_maps',
    'probmaps',  # TODO
    'metrics',
]

label_name = 'encapsulins'


for p in [results_path]:
    os.makedirs(p, exist_ok=True)


model_paths = eul([
    # f'/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_uni_v4/GDL_CE_B_GA_tm_only__UNet__22-02-28_20-13-52/model_final.pt'  # TmEnc only
    f'/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_uni_v4/GDL_CE_B_GA___UNet__22-02-26_02-02-13/model_step150000.pt'  # universal

    # f'/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_uni/GDL_CE_B_GA_nb5__UNet__22-02-23_02-32-41/model_step80000.pt'
    # f'/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_uni/GDL_CE_B_GA___UNet__22-02-21_05-30-56/model_step70000.pt',
    # f'~/tum/binary_mxqtsegtrain2_trainings/GDL_CE_B_GA___UNet__21-12-14_17-32-22/model_63k.pt',
    # f'~/tum/binary_mxqtsegtrain2_trainings/B___UNet__21-12-10_03-08-01/model_{model_variant}',
    # f'~/tumtrainings/15to54_encapsulins__UNet__21-09-16_03-10-26/model_{model_variant}',
    # f'~/tumtrainings/D_15to54_encapsulins__UNet__21-09-16_04-02-24/model_{model_variant}',
    # f'~/tumtrainings/M___UNet__21-09-13_04-32-52/model_{model_variant}',
])


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
        verbose=True,
        augmentations=3 if apply_softmax else None,
        apply_softmax=apply_softmax,
    )
    m_targets = []
    m_probs = []
    m_preds = []
    for img_path in img_paths:
        inp = np.array(imageio.imread(img_path), dtype=np.float32)[None][None]  # (N=1, C=1, H, W)
        out = predictor.predict(inp)
        if is_multi:
            out.sigmoid_()
        out = out.numpy()
        basename = os.path.splitext(os.path.basename(img_path))[0]

        if out.shape[1] in {1, 2}:
            if out.shape[1] == 2:  # Binary segmentation -> only export channel 1
                cout = out[0, 1]
                cout = (cout * 255.).astype(np.uint8)
                cout = cout > thresh
                kind = f'thresh{thresh}'
            elif out.shape[1] == 1:  # Distance transform regression
                cout = out[0, 0]
                cout = cout <= dt_thresh
                kind = f'dtthresh{dt_thresh}'

            # Postprocessing:
            # cout = sm.remove_small_holes(cout, 2000)
            # cout = sm.remove_small_objects(cout, 100)

            # Make imageio.imwrite-able
            cout = cout.astype(np.uint8) * 255

            # out_path = eu(f'{results_path}/{basename}_{modelname}_{kind}.png')
            out_path = eu(f'{results_path}/{basename}_{kind}.png')
            print(f'Writing inference result to {out_path}')
            if 'thresh' in DESIRED_OUTPUTS:
                imageio.imwrite(out_path, cout)

            if is_distmap:
                dmap_path = eu(f'{results_path}/{basename}_{modelname}_distmap.png')
                # dmap = ((out[0, 0] + 1.) * 128.).astype(np.uint8)
                # imageio.imwrite(dmap_path, dmap)
                dmap = out[0, 0]
                plt.figure(figsize=(8, 8), tight_layout=True)
                plt.imshow(dmap)
                plt.colorbar()
                plt.savefig(dmap_path)

            # Write raw and gt labels
            lab_path = f'{img_path[:-4]}_{label_name}.tif'
            if not Path(lab_path).exists():
                lab_path = f'{img_path[:-4]}_{label_name}.TIF'
            lab_img = np.array(imageio.imread(lab_path))
            if invert_labels:
                lab_img = (lab_img == 0).astype(np.int64)
            if ENABLE_TERRIBLE_INVERSION_HACK and int(basename) >= 55:
                lab_img = (lab_img == 0).astype(np.int64)

            # lab_img = sm.binary_erosion(lab_img, sm.selem.disk(5)).astype(lab_img.dtype)  # Model was trained with this target transform. Change this if training changes!
            lab_img = ((lab_img > 0) * 255).astype(np.uint8)  # Binarize (binary training specific!)

            raw_img = imageio.imread(img_path)
            if 'raw' in DESIRED_OUTPUTS:
                imageio.imwrite(eu(f'{results_path}/{basename}_raw.jpg'), raw_img)
            if 'lab' in DESIRED_OUTPUTS:
                imageio.imwrite(eu(f'{results_path}/{basename}_lab.png'), lab_img)

            if 'overlays' in DESIRED_OUTPUTS:
                # Create overlay images
                lab_overlay = label2rgb(lab_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['red'])
                pred_overlay = label2rgb(cout > 0, raw_img, bg_label=0, alpha=0.5, colors=['green'])
                # Redraw raw image onto overlays where they were blended with 0, to restore original brightness
                raw_img_01 = raw_img.astype(np.float64) / 255.
                lab_overlay[lab_img == 0, :] = raw_img_01[lab_img == 0, None]
                pred_overlay[cout == 0, :] = raw_img_01[cout == 0, None]
                # Convert from [0, 1] float to [0, 255] uint8 for imageio
                lab_overlay = (lab_overlay * 255.).astype(np.uint8)
                pred_overlay = (pred_overlay * 255.).astype(np.uint8)

                imageio.imwrite(eu(f'{results_path}/{basename}_overlay_lab.jpg'), lab_overlay)
                imageio.imwrite(eu(f'{results_path}/{basename}_overlay_pred.jpg'), pred_overlay)

            if 'error_maps' in DESIRED_OUTPUTS:
                # Create error image
                error_img = lab_img != cout
                error_img = (error_img.astype(np.uint8)) * 255
                imageio.imwrite(eu(f'{results_path}/{basename}_error.png'), error_img)

                # Create false positive (fp) image
                fp_error_img = (lab_img == 0) & (cout > 0)
                fp_error_img = (fp_error_img.astype(np.uint8)) * 255
                imageio.imwrite(eu(f'{results_path}/{basename}_fp_error.png'), fp_error_img)
                # Create false positive (fp) image overlay
                fp_overlay = label2rgb(fp_error_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['magenta'])
                fp_overlay[fp_error_img == 0, :] = raw_img_01[fp_error_img == 0, None]
                fp_overlay = (fp_overlay * 255.).astype(np.uint8)
                imageio.imwrite(eu(f'{results_path}/{basename}_fp_error_overlay.jpg'), fp_overlay)

                # Create false negative (fn) image
                fn_error_img = (lab_img > 0) & (cout == 0)
                fn_error_img = (fn_error_img.astype(np.uint8)) * 255
                imageio.imwrite(eu(f'{results_path}/{basename}_fn_error.png'), fn_error_img)
                # Create false negative (fn) image overlay
                fn_overlay = label2rgb(fn_error_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['magenta'])
                fn_overlay[fn_error_img == 0, :] = raw_img_01[fn_error_img == 0, None]
                fn_overlay = (fn_overlay * 255.).astype(np.uint8)
                imageio.imwrite(eu(f'{results_path}/{basename}_fn_error_overlay.jpg'), fn_overlay)


            m_targets.append((lab_img > 0))#.reshape(-1))
            m_preds.append((cout > 0))#.reshape(-1))
            if is_distmap:
                m_probs.append(out[0, 0])#.reshape(-1))
            else:
                m_probs.append(out[0, 1])#.reshape(-1))

        elif out.shape[1] > 2:  # Export each channel separately
            raw_img = imageio.imread(img_path)
            raw_img_01 = raw_img.astype(np.float64) / 255.
            if 'raw' in DESIRED_OUTPUTS:
                imageio.imwrite(eu(f'{results_path}/{basename}_raw.jpg'), raw_img)
            export_channels = range(out.shape[1])
            for c in export_channels:
                # Probmaps
                cout = out[0, c]
                cout = (cout * 255.).astype(np.uint8)
                out_path = eu(f'{results_path}/{basename}_c{c}_{modelname}.png')
                if 'probmaps' in DESIRED_OUTPUTS:
                    imageio.imwrite(out_path, cout)

                # Probmaps thresholded
                out_thresh_path = eu(f'{results_path}/{basename}_c{c}_thresh{multi_thresh}_{modelname}.png')
                cout_thresh_bin = cout >= multi_thresh
                cout_thresh = (cout_thresh_bin.astype(np.uint8) * 255)
                if 'thresh' in DESIRED_OUTPUTS:
                    imageio.imwrite(out_thresh_path, cout_thresh)

                if 'overlays' in DESIRED_OUTPUTS:
                    # Create overlay images
                    # First get GT label image
                    lab_path = f'{img_path[:-4]}_{multi_label_names[c]}.tif'
                    if not Path(lab_path).exists():
                        lab_path = f'{img_path[:-4]}_{multi_label_names[c]}.TIF'
                    if os.path.exists(lab_path):  # Not all labels are always available
                        lab_img = np.array(imageio.imread(lab_path))
                        if invert_labels:
                            lab_img = (lab_img == 0).astype(np.int64)
                        if ENABLE_TERRIBLE_INVERSION_HACK and int(basename) >= 60:
                            lab_img = (lab_img == 0).astype(np.int64)
                    else:
                        lab_img = np.zeros_like(raw_img)
                    lab_img = ((lab_img > 0) * 255).astype(np.uint8)
                    lab_overlay = label2rgb(lab_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['red'])
                    lab_overlay[lab_img == 0, :] = raw_img_01[lab_img == 0, None]
                    lab_overlay = (lab_overlay * 255.).astype(np.uint8)

                    pred_overlay = label2rgb(cout_thresh_bin, raw_img, bg_label=0, alpha=0.5, colors=['green'])
                    pred_overlay[cout_thresh_bin == 0, :] = raw_img_01[cout_thresh_bin == 0, None]
                    pred_overlay = (pred_overlay * 255.).astype(np.uint8)

                    imageio.imwrite(eu(f'{results_path}/{basename}_overlay_{multi_label_names[c]}_lab.jpg'), lab_overlay)
                    imageio.imwrite(eu(f'{results_path}/{basename}_overlay_{multi_label_names[c]}_pred.jpg'), pred_overlay)

            if 'argmax' in DESIRED_OUTPUTS:
                # Argmax of channel probs
                pred = np.argmax(out, 1)[0]
                # plab = skimage.color.label2rgb(pred, bg_label=0)
                plab = skimage.color.label2rgb(pred, colors=['red', 'green', 'blue', 'purple', 'brown', 'magenta'], bg_label=0)
                out_path = eu(f'{results_path}/{basename}_argmax_{modelname}.jpg')
                imageio.imwrite(out_path, plab)

    if 'metrics' in DESIRED_OUTPUTS and not is_multi:  # TODO: Make this work with multilabel
        # Calculate pixelwise precision and recall
        m_targets = np.concatenate(m_targets, axis=None)
        m_probs = np.concatenate(m_probs, axis=None)
        m_preds = np.concatenate(m_preds, axis=None)
        precision = sme.precision_score(m_targets, m_preds)
        recall = sme.recall_score(m_targets, m_preds)
        # Plot pixelwise PR curve
        p, r, t = sme.precision_recall_curve(m_targets, m_probs)
        plt.figure(figsize=(3, 3))
        np.savez_compressed('prdata.npy', p,r,t)
        plt.plot(r, p)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.minorticks_on()
        plt.grid(True, 'both')

        # Get index of pr-curve's threshold that's nearest to the one used in practice for this segmentation
        _i = np.abs(t - thresh/255).argmin()
        plt.scatter(r[_i], p[_i])
        plt.annotate(f'(r={r[_i]:.2f}, p={p[_i]:.2f})', (r[_i] - 0.6, p[_i] - 0.2))

        plt.tight_layout()
        plt.savefig(eu(f'{results_path}/prcurve.pdf'), dpi=300)
        
        # with open(eu(f'{results_path}/{modelname}_{kind}_pr.txt'), 'w') as f:
        with open(eu(f'{results_path}/info.txt'), 'w') as f:
            f.write(
    f"""
    model: {model_path}
    thresh: {thresh if not is_distmap else dt_thresh}

    precision: {precision}
    recall: {recall}
    """
            )

import IPython; IPython.embed(); exit()
