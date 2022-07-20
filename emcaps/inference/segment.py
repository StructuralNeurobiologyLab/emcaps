"""Inference script for segmentation prediction and evaluation, based on
raw images and a model trained with `segtrain.py`."""


import os
from pathlib import Path
from os.path import expanduser as eu

import numpy as np
import imageio.v3 as iio
import skimage
import torch
import yaml

from skimage import morphology as sm
from skimage import measure
from skimage.color import label2rgb
from sklearn import metrics as sme
from scipy import ndimage

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


from elektronn3.inference import Predictor
from elektronn3.data import transforms


from emcaps.utils import get_old_enctype, get_v5_enctype, OLDNAMES_TO_V5NAMES, clean_int, ensure_not_inverted, get_meta


# torch.backends.cudnn.benchmark = True


def main():


    def eul(paths):
        """Shortcut for expanding all user paths in a list"""
        return [os.path.expanduser(p) for p in paths]


    # Keep this in sync with training normalization
    dataset_mean = (128.0,)
    dataset_std = (128.0,)


    pre_predict_transform = transforms.Compose([
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    ENABLE_ENCTYPE_SUBDIRS = False
    ZERO_LABELS = True

    EVAL_ON_DRO = False


    import argparse
    parser = argparse.ArgumentParser(description='Run inference with a trained network.')
    parser.add_argument('srcpath', help='Path to input file', default=None)
    parser.add_argument('-t', default=False, action='store_true', help='enable tiled inference')
    parser.add_argument('-a', type=int, default=0, choices=[0, 1, 2], help='Number of test-time augmentations to use')
    args = parser.parse_args()

    tta_num = args.a
    srcpath = os.path.expanduser(args.srcpath) if args.srcpath is not None else None

    """
    for ETYPE in '1M-Mx' '1M-Qt' '2M-Mx' '2M-Qt' '3M-Qt' '1M-Tm'
        python -m inference.inference -c $ETYPE
    end
    """

    thresh = 127
    dt_thresh = 0.00

    MINSIZE = 150

    img_paths = [srcpath]
    results_root = Path(img_paths[0]).parent


    DESIRED_OUTPUTS = [
        'raw',
        'thresh',
        # 'lab',
        'overlays',
        # 'error_maps',
        'probmaps',
        # 'metrics',
    ]

    label_name = 'encapsulins'



    for p in [results_root]:
        p.mkdir(exist_ok=True)


    model_path = './unet_v7_all.pts'

    modelname = os.path.basename(os.path.dirname(model_path))

    if args.t:
        tile_shape = (448, 448)
        overlap_shape = (32, 32)
        # TODO
        out_shape = np.array(iio.imread(img_paths[0])).shape
        out_shape = (2, *out_shape)
    else:
        tile_shape = None
        overlap_shape = None
        out_shape = None

    apply_softmax = True
    predictor = Predictor(
        model=model_path,
        device=None,
        tile_shape=tile_shape,
        overlap_shape=overlap_shape,
        out_shape=out_shape,
        float16=True,
        transform=pre_predict_transform,
        verbose=True,
        augmentations=tta_num,
        apply_softmax=apply_softmax,
    )
    m_targets = []
    m_probs = []
    m_preds = []
    for img_path in img_paths:
        inp = np.array(iio.imread(img_path), dtype=np.float32)[None][None]  # (N=1, C=1, H, W)
        # TODO: N=1 ?
        out = predictor.predict(inp)
        out = out.numpy()
        basename = os.path.splitext(os.path.basename(img_path))[0]

        if ENABLE_ENCTYPE_SUBDIRS:
            # enctype = get_old_enctype(img_path)
            enctype = get_v5_enctype(img_path)
            results_path = results_root / enctype
            results_path.mkdir(exist_ok=True)
        else:
            results_path = results_root

        if out.shape[1] in {1, 2}:
            if out.shape[1] == 2:  # Binary segmentation -> only export channel 1
                cout = out[0, 1]
                cout = (cout * 255.).astype(np.uint8)
                cout = cout > thresh
                # kind = f'thresh{thresh}'
                kind = f'thresh'
            elif out.shape[1] == 1:  # Distance transform regression
                cout = out[0, 0]
                cout = cout <= dt_thresh
                kind = f'dtthresh{dt_thresh}'

            # Postprocessing:
            cout = sm.remove_small_holes(cout, 2000)
            cout = sm.remove_small_objects(cout, MINSIZE)

            # cc, n_comps = ndimage.label(cout)

            # rprops = measure.regionprops(cc, inp[0, 0])



            # Make iio.imwrite-able
            cout = cout.astype(np.uint8) * 255

            # out_path = eu(f'{results_path}/{basename}_{modelname}_{kind}.tif')
            out_path = eu(f'{results_path}/{basename}_{kind}.tif')
            print(f'Writing inference result to {out_path}')
            if 'thresh' in DESIRED_OUTPUTS:
                iio.imwrite(out_path, cout)

            if 'probmaps' in DESIRED_OUTPUTS:
                probmap = (out[0, 1] * 255.).astype(np.uint8)
                probmap_path = eu(f'{results_path}/{basename}_probmap.tif')
                iio.imwrite(probmap_path, probmap)

            raw_img = iio.imread(img_path)

            # Write raw and gt labels
            if ZERO_LABELS:
                lab_img = np.zeros_like(raw_img, dtype=np.uint8)
            else:
                lab_path = f'{str(img_path)[:-4]}_{label_name}.tif'
                if not Path(lab_path).exists():
                    lab_path = f'{img_path[:-4]}_{label_name}.TIF'
                lab_img = np.array(iio.imread(lab_path))
                lab_img = ensure_not_inverted(lab_img > 0, verbose=True, error=False)[0].astype(np.int64)

                # lab_img = sm.binary_erosion(lab_img, sm.selem.disk(5)).astype(lab_img.dtype)  # Model was trained with this target transform. Change this if training changes!
                lab_img = ((lab_img > 0) * 255).astype(np.uint8)  # Binarize (binary training specific!)

            if 'raw' in DESIRED_OUTPUTS:
                iio.imwrite(eu(f'{results_path}/{basename}_raw.tif'), raw_img)
            if 'lab' in DESIRED_OUTPUTS:
                iio.imwrite(eu(f'{results_path}/{basename}_lab.tif'), lab_img)

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

                if not ZERO_LABELS:
                    iio.imwrite(eu(f'{results_path}/{basename}_overlay_lab.tif'), lab_overlay)
                iio.imwrite(eu(f'{results_path}/{basename}_overlay_pred.tif'), pred_overlay)

            if 'error_maps' in DESIRED_OUTPUTS:
                # Create error image
                error_img = lab_img != cout
                error_img = (error_img.astype(np.uint8)) * 255
                iio.imwrite(eu(f'{results_path}/{basename}_error.tif'), error_img)

                # Create false positive (fp) image
                fp_error_img = (lab_img == 0) & (cout > 0)
                fp_error_img = (fp_error_img.astype(np.uint8)) * 255
                iio.imwrite(eu(f'{results_path}/{basename}_fp_error.tif'), fp_error_img)
                # Create false positive (fp) image overlay
                fp_overlay = label2rgb(fp_error_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['magenta'])
                fp_overlay[fp_error_img == 0, :] = raw_img_01[fp_error_img == 0, None]
                fp_overlay = (fp_overlay * 255.).astype(np.uint8)
                iio.imwrite(eu(f'{results_path}/{basename}_fp_error_overlay.tif'), fp_overlay)

                # Create false negative (fn) image
                fn_error_img = (lab_img > 0) & (cout == 0)
                fn_error_img = (fn_error_img.astype(np.uint8)) * 255
                iio.imwrite(eu(f'{results_path}/{basename}_fn_error.tif'), fn_error_img)
                # Create false negative (fn) image overlay
                fn_overlay = label2rgb(fn_error_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['magenta'])
                fn_overlay[fn_error_img == 0, :] = raw_img_01[fn_error_img == 0, None]
                fn_overlay = (fn_overlay * 255.).astype(np.uint8)
                iio.imwrite(eu(f'{results_path}/{basename}_fn_error_overlay.tif'), fn_overlay)


            m_targets.append((lab_img > 0))#.reshape(-1))
            m_preds.append((cout > 0))#.reshape(-1))
            m_probs.append(out[0, 1])#.reshape(-1))

            if 'argmax' in DESIRED_OUTPUTS:
                # Argmax of channel probs
                pred = np.argmax(out, 1)[0]
                # plab = skimage.color.label2rgb(pred, bg_label=0)
                plab = skimage.color.label2rgb(pred, colors=['red', 'green', 'blue', 'purple', 'brown', 'magenta'], bg_label=0)
                out_path = eu(f'{results_path}/{basename}_argmax_{modelname}.tif')
                iio.imwrite(out_path, plab)

    if 'metrics' in DESIRED_OUTPUTS:
        # Calculate pixelwise precision and recall
        m_targets = np.concatenate(m_targets, axis=None)
        m_probs = np.concatenate(m_probs, axis=None)
        m_preds = np.concatenate(m_preds, axis=None)
        iou = sme.jaccard_score(m_targets, m_preds)  # iou == jaccard score
        dsc = sme.f1_score(m_targets, m_preds)  # dsc == f1 score
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
        # plt.annotate(f'(r={r[_i]:.2f}, p={p[_i]:.2f})', (r[_i] - 0.6, p[_i] - 0.2))

        plt.tight_layout()
        plt.savefig(eu(f'{results_root}/prcurve.pdf'), dpi=300)
        
        with open(eu(f'{results_root}/info.txt'), 'w') as f:
            f.write(
    f"""Output description:
- X_raw.tif: raw image (image number X from the shared dataset)
- X_probmap.tif: raw softmax pseudo-probability outputs (before thresholding).
- X_thresh.tif: binary segmentation map, obtained by neural network with standard threshold 127/255 (i.e. ~ 50% confidence)
- X_overlay_lab.tif: given GT label annotations, overlayed on raw image
- X_overlay_pred.tif: prediction by the neural network, overlayed on raw image
- X_fn_error.tif: map of false negative predictions w.r.t. GT labels
- X_fp_error.tif: map of false positive predictions w.r.t. GT labels
- X_fn_error_overlay.tif: map of false negative predictions w.r.t. GT labels, overlayed on raw image
- X_fp_error_overlay.tif: map of false positive predictions w.r.t. GT labels, overlayed on raw image


Model info:
- model: {model_path}
- thresh: {thresh}

- IoU: {iou * 100:.1f}%
- DSC: {dsc * 100:.1f}%
- precision: {precision * 100:.1f}%
- recall: {recall * 100:.1f}%
"""
)



if __name__ == '__main__':
    main()
