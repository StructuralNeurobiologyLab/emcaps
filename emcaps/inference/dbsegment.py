"""Inference script for segmentation prediction and evaluation, based on
raw images and a model trained with `segtrain.py`."""


# TODO: Reconfigure to do one example batch processing run based on single table database sheet

import os
from pathlib import Path
from os.path import expanduser as eu

import numpy as np
import imageio.v3 as iio
import skimage
import torch
import yaml
import pandas as pd

from skimage import morphology as sm
from skimage.color import label2rgb
from sklearn import metrics as sme

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


from elektronn3.inference import Predictor
from elektronn3.data import transforms

from emcaps import utils
from emcaps.utils import get_old_enctype, get_v5_enctype, OLDNAMES_TO_V5NAMES, clean_int, ensure_not_inverted, get_meta


# torch.backends.cudnn.benchmark = True


def produce_metrics(thresh, results_root, model_path, data_selection, m_targets, m_preds, m_probs):
    # Calculate pixelwise precision and recall
    # if m_targets.max() < 1:
    #     import IPython ; IPython.embed(); raise SystemExit
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
    # np.savez_compressed('prdata.npy', p,r,t)
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
    plt.close()

    with open(eu(f'{results_root}/info.txt'), 'w') as f:
        f.write(
        f"""Output description:
- X_raw.jpg: raw image (image number X from the shared dataset)
- X_probmap.jpg: raw softmax pseudo-probability outputs (before thresholding).
- X_thresh.png: binary segmentation map, obtained by neural network with standard threshold 127/255 (i.e. ~ 50% confidence)
- X_overlay_lab.jpg: given GT label annotations, overlayed on raw image
- X_overlay_pred.jpg: prediction by the neural network, overlayed on raw image
- X_fn_error.png: map of false negative predictions w.r.t. GT labels
- X_fp_error.png: map of false positive predictions w.r.t. GT labels
- X_fn_error_overlay.jpg: map of false negative predictions w.r.t. GT labels, overlayed on raw image
- X_fp_error_overlay.jpg: map of false positive predictions w.r.t. GT labels, overlayed on raw image
Info:
- data: {data_selection}
- model: {model_path}
- thresh: {thresh}
- IoU: {iou * 100:.1f}%
- DSC: {dsc * 100:.1f}%
- precision: {precision * 100:.1f}%
- recall: {recall * 100:.1f}%
"""
    )
    metrics_dict = {'dsc': dsc, 'iou': iou, 'precision': precision, 'recall': recall}
    return metrics_dict


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

    ENABLE_ENCTYPE_SUBDIRS = True
    ZERO_LABELS = False


    import argparse
    parser = argparse.ArgumentParser(description='Run inference with a trained network.')
    parser.add_argument('--srcpath', help='Path to input file', default=None)
    parser.add_argument('--model', help='Path to model file', default=None)
    parser.add_argument('-s', help='Traing setting (for auto model selection)', default=None)
    parser.add_argument('-t', default=False, action='store_true', help='enable tiled inference')
    parser.add_argument('-c', '--constraintype', default=None, help='Constrain inference to only one encapsulin type (via v5name, e.g. `-c 1M-Qt`).')
    parser.add_argument('-e', '--use-expert', default=False, action='store_true', help='If true, use expert models for each enc type. Else, use common model')
    parser.add_argument('-a', type=int, default=2, choices=[0, 1, 2], help='Number of test-time augmentations to use')
    args = parser.parse_args()

    selected_enctype = args.constraintype
    use_expert = args.use_expert
    tta_num = args.a
    srcpath = None
    tr_setting = args.s

    """
    for ETYPE in '1M-Mx' '1M-Qt' '2M-Mx' '2M-Qt' '3M-Qt' '1M-Tm'
        python -m inference.inference -c $ETYPE
    end
    """

    thresh = 127
    dt_thresh = 0.00

    if tr_setting is None:
        tr_setting = 'qttm'

    if srcpath is None:
        results_root = Path(f'/wholebrain/scratch/mdraw/tum/results_seg_v10a_tr-{tr_setting}')
        # results_root = Path('/wholebrain/scratch/mdraw/tum/results_seg_v10_tr-hek')
        if selected_enctype is not None:
            msuffix = '_expert' if use_expert else ''
            results_root = Path(f'{str(results_root)}{msuffix}_{selected_enctype}')
        data_root = Path('/wholebrain/scratch/mdraw/tum/Single-table_database/')


        class_groups_to_include = [
            'simple_hek',
            'dro',
            'mice',
            'qttm',
            'multi',
        ]

        if selected_enctype is None:
            included = []
            for cgrp in class_groups_to_include:
                cgrp_classes = utils.CLASS_GROUPS[cgrp]
                print(f'Including class group {cgrp} containing classes {cgrp_classes}')
                included.extend(cgrp_classes)
            DATA_SELECTION_V5NAMES = included
        else:
            DATA_SELECTION_V5NAMES = [selected_enctype]


        def find_vx_val_images(isplitpath=None):
            """Find paths to all raw validation images of split vx"""
            if isplitpath is None:
                isplitpath = data_root / 'isplitdata_v10a'
            val_img_paths = []
            for p in isplitpath.rglob('*_val.png'):  # Look for all validation raw images recursively
                if get_v5_enctype(p) in DATA_SELECTION_V5NAMES:
                    val_img_paths.append(p)
            return val_img_paths

        img_paths = find_vx_val_images()

        for p in [results_root / scond for scond in DATA_SELECTION_V5NAMES]:
            os.makedirs(p, exist_ok=True)

        if len(DATA_SELECTION_V5NAMES) == 1:
            v5_enctype = DATA_SELECTION_V5NAMES[0]
            results_root = Path(f'{str(results_root)}_{v5_enctype}')

    else:
        img_paths = [srcpath]
        results_root = Path(img_paths[0]).parent


    DESIRED_OUTPUTS = [
        'raw',
        'thresh',
        'lab',
        'overlays',
        'error_maps',
        'probmaps',
        'metrics',
    ]

    label_name = 'encapsulins'


    class Randomizer(torch.nn.Module):
        """Fake module for producing correctly shaped random outputs in range [0, 1]"""
        def forward(self, x):
            return torch.rand(x.shape[0], 2, *x.shape[2:])

    if args.model is None:
        model_dict = {
            'all': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10/GA_all_dec98__UNet__22-09-24_03-24-36/model_step160000.pts',
            'hek': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10/GA_hek__UNet__22-09-24_03-29-06/model_step160000.pts',
            'dro': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10/GA_dro__UNet__22-09-24_03-26-07/model_step160000.pts',
            'mice': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10/MICE_2M-Qt_GA_mice__UNet__22-09-24_03-26-43/model_step240000.pts',

            'qttm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10/GA_qttm__UNet__22-09-24_03-27-16/model_step240000.pts',

            'onlytm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_onlytm/GA_onlytm__UNet__22-09-24_03-32-39/model_step160000.pts',
            'all_notm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_notm/GA_notm_all__UNet__22-09-24_03-33-19/model_step160000.pts',
            'hek_notm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_notm/GA_notm_hek__UNet__22-09-24_03-34-32/model_step160000.pts',
        }
        model_paths = [model_dict[tr_setting]]
        # model_paths = [Randomizer()]  # produce random outputs instead
    else:
        model_paths = [args.model]

    assert len(model_paths) == 1, 'Currently only one model is supported per inference run'
    for model_path in model_paths:
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

        if ENABLE_ENCTYPE_SUBDIRS:
            per_enctype_results = {}
            for enctype in DATA_SELECTION_V5NAMES:
                per_enctype_results[enctype] = {
                    'targets': [], 'preds': [], 'probs': []
                }

        assert len(img_paths) > 0
        for img_path in img_paths:
            inp = np.array(iio.imread(img_path), dtype=np.float32)[None][None]  # (N=1, C=1, H, W)
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
                # cout = sm.remove_small_holes(cout, 2000)
                # cout = sm.remove_small_objects(cout, 100)

                # Make iio.imwrite-able
                cout = cout.astype(np.uint8) * 255

                # out_path = eu(f'{results_path}/{basename}_{modelname}_{kind}.png')
                out_path = eu(f'{results_path}/{basename}_{kind}.png')
                print(f'Writing inference result to {out_path}')
                if 'thresh' in DESIRED_OUTPUTS:
                    iio.imwrite(out_path, cout)

                if 'probmaps' in DESIRED_OUTPUTS:
                    probmap = (out[0, 1] * 255.).astype(np.uint8)
                    probmap_path = eu(f'{results_path}/{basename}_probmap.jpg')
                    iio.imwrite(probmap_path, probmap)

                raw_img = iio.imread(img_path)

                # Write raw and gt labels
                if ZERO_LABELS:
                    lab_img = np.zeros_like(raw_img, dtype=np.uint8)
                else:
                    lab_path = f'{str(img_path)[:-4]}_{label_name}.png'
                    if not Path(lab_path).exists():
                        lab_path = f'{img_path[:-4]}_{label_name}.tif'
                    lab_img = np.array(iio.imread(lab_path))
                    lab_img = ensure_not_inverted(lab_img > 0, verbose=True, error=False)[0].astype(np.int64)

                    # lab_img = sm.binary_erosion(lab_img, sm.selem.disk(5)).astype(lab_img.dtype)  # Model was trained with this target transform. Change this if training changes!
                    lab_img = ((lab_img > 0) * 255).astype(np.uint8)  # Binarize (binary training specific!)

                if 'raw' in DESIRED_OUTPUTS:
                    iio.imwrite(eu(f'{results_path}/{basename}_raw.jpg'), raw_img)
                if 'lab' in DESIRED_OUTPUTS:
                    iio.imwrite(eu(f'{results_path}/{basename}_lab.png'), lab_img)

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
                        iio.imwrite(eu(f'{results_path}/{basename}_overlay_lab.jpg'), lab_overlay)
                    iio.imwrite(eu(f'{results_path}/{basename}_overlay_pred.jpg'), pred_overlay)

                if 'error_maps' in DESIRED_OUTPUTS:
                    # Create error image
                    error_img = lab_img != cout
                    error_img = (error_img.astype(np.uint8)) * 255
                    iio.imwrite(eu(f'{results_path}/{basename}_error.png'), error_img)

                    # Create false positive (fp) image
                    fp_error_img = (lab_img == 0) & (cout > 0)
                    fp_error_img = (fp_error_img.astype(np.uint8)) * 255
                    iio.imwrite(eu(f'{results_path}/{basename}_fp_error.png'), fp_error_img)
                    # Create false positive (fp) image overlay
                    fp_overlay = label2rgb(fp_error_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['magenta'])
                    fp_overlay[fp_error_img == 0, :] = raw_img_01[fp_error_img == 0, None]
                    fp_overlay = (fp_overlay * 255.).astype(np.uint8)
                    iio.imwrite(eu(f'{results_path}/{basename}_fp_error_overlay.jpg'), fp_overlay)

                    # Create false negative (fn) image
                    fn_error_img = (lab_img > 0) & (cout == 0)
                    fn_error_img = (fn_error_img.astype(np.uint8)) * 255
                    iio.imwrite(eu(f'{results_path}/{basename}_fn_error.png'), fn_error_img)
                    # Create false negative (fn) image overlay
                    fn_overlay = label2rgb(fn_error_img > 0, raw_img, bg_label=0, alpha=0.5, colors=['magenta'])
                    fn_overlay[fn_error_img == 0, :] = raw_img_01[fn_error_img == 0, None]
                    fn_overlay = (fn_overlay * 255.).astype(np.uint8)
                    iio.imwrite(eu(f'{results_path}/{basename}_fn_error_overlay.jpg'), fn_overlay)


                m_target = (lab_img > 0)#.reshape(-1)
                m_pred = (cout > 0)#.reshape(-1))
                m_prob = (out[0, 1])#.reshape(-1))

                if ENABLE_ENCTYPE_SUBDIRS:
                    per_enctype_results[enctype]['targets'].append(m_target)
                    per_enctype_results[enctype]['preds'].append(m_pred)
                    per_enctype_results[enctype]['probs'].append(m_prob)

                m_targets.append(m_target)
                m_preds.append(m_pred)
                m_probs.append(m_prob)

                if 'argmax' in DESIRED_OUTPUTS:
                    # Argmax of channel probs
                    pred = np.argmax(out, 1)[0]
                    # plab = skimage.color.label2rgb(pred, bg_label=0)
                    plab = skimage.color.label2rgb(pred, colors=['red', 'green', 'blue', 'purple', 'brown', 'magenta'], bg_label=0)
                    out_path = eu(f'{results_path}/{basename}_argmax_{modelname}.jpg')
                    iio.imwrite(out_path, plab)

        if 'metrics' in DESIRED_OUTPUTS:
            METRICS_KEYS = ['dsc', 'iou', 'precision', 'recall']
            global_metrics_dict = produce_metrics(
                thresh=thresh,
                results_root=results_root,
                model_path=model_path,
                data_selection=DATA_SELECTION_V5NAMES,
                m_targets=m_targets,
                m_preds=m_preds,
                m_probs=m_probs
            )
            dfdict = {'All': [global_metrics_dict[k] for k in METRICS_KEYS]}

            if ENABLE_ENCTYPE_SUBDIRS:
                for enctype in DATA_SELECTION_V5NAMES:
                    enctype_metrics_dict = produce_metrics(
                        thresh=thresh,
                        results_root=results_root / enctype,
                        model_path=model_path,
                        data_selection=[enctype],
                        m_targets=per_enctype_results[enctype]['targets'],
                        m_preds=per_enctype_results[enctype]['preds'],
                        m_probs=per_enctype_results[enctype]['probs']
                    )
                    per_enctype_results[enctype]['metrics'] = enctype_metrics_dict

                    em = per_enctype_results[enctype]['metrics']
                    assert list(em.keys()) == METRICS_KEYS 
                    dfdict[enctype] = [em[k] for k in METRICS_KEYS]

                # df_index = ['All'] + DATA_SELECTION_V5NAMES

                dfmetrics = pd.DataFrame(dfdict, index=METRICS_KEYS)
                dfmetrics.to_excel(results_root / 'metrics.xlsx')

if __name__ == '__main__':
    main()
