"""Inference script for segmentation prediction and evaluation, based on
raw images and a model trained with `segtrain.py`."""


# TODO: Reconfigure to do one example batch processing run based on single table database sheet

import os
from pathlib import Path
from os.path import expanduser as eu

import numpy as np
import hydra
from omegaconf import DictConfig
import imageio.v3 as iio
import skimage
import torch
import yaml
import pandas as pd
import torch.backends.cudnn

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
from emcaps.utils import inference_utils as iu

torch.backends.cudnn.benchmark = True


def produce_metrics(thresh, results_root, segmenter_path, classifier_path, data_selection, m_targets, m_preds, m_probs):
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
- segmenter: {segmenter_path}
- classifier: {classifier_path}
- thresh: {thresh}
- IoU: {iou * 100:.1f}%
- DSC: {dsc * 100:.1f}%
- precision: {precision * 100:.1f}%
- recall: {recall * 100:.1f}%
"""
    )
    metrics_dict = {'dsc': dsc, 'iou': iou, 'precision': precision, 'recall': recall}
    return metrics_dict


def find_vx_val_images(isplit_data_path, group_name, sheet_path):
    """Find paths to all raw validation images of split vx"""
    val_img_paths = []
    for p in isplit_data_path.rglob('*_val.png'):  # Look for all validation raw images recursively
        if utils.is_in_data_group(path_or_num=p, group_name=group_name, sheet_path=sheet_path):
            val_img_paths.append(p)
    return val_img_paths


@hydra.main(version_base=None, config_path='../../conf', config_name='config')
def main(cfg: DictConfig) -> None:

    pre_predict_transform = transforms.Compose([
        transforms.Normalize(mean=cfg.dataset_mean, std=cfg.dataset_std)
    ])

    ENABLE_ENCTYPE_SUBDIRS = True
    ZERO_LABELS = False

    tta_num = cfg.dbsegment.tta_num
    minsize = cfg.minsize
    segmenter_path = cfg.dbsegment.segmenter
    classifier_path = cfg.dbsegment.classifier

    """
    for ETYPE in '1M-Mx' '1M-Qt' '2M-Mx' '2M-Qt' '3M-Qt' '1M-Tm'
        python -m inference.inference -c $ETYPE
    end
    """

    thresh = 127
    dt_thresh = 0.00

    # allowed_classes_for_classification = utils.CLASS_GROUPS['simple_hek']
    allowed_classes_for_classification = [
        '1M-Mx',
        '1M-Qt',
        '2M-Mx',
        '2M-Qt',
        '3M-Qt',
        '1M-Tm',
    ]

    constraint_signature = ''  # Unconstrained
    if allowed_classes_for_classification != utils.CLASS_GROUPS['simple_hek']:
        constraint_signature = '_constrained'
        for ac in allowed_classes_for_classification:
            constraint_signature = f'{constraint_signature}_{ac}'

    results_root = Path(cfg.dbsegment.results_root)

    img_paths = find_vx_val_images(isplit_data_path=cfg.isplit_data_path, group_name=cfg.eval_group, sheet_path=cfg.sheet_path)

    class_groups_to_include = [
        'simple_hek',
        'dro',
        'mice',
        'qttm',
        'multi',
    ]
    included = []
    for cgrp in class_groups_to_include:
        cgrp_classes = utils.CLASS_GROUPS[cgrp]
        # logger.info(f'Including class group {cgrp}, containing classes {cgrp_classes}')
        included.extend(cgrp_classes)
    DATA_SELECTION_V5NAMES = included

    for p in [results_root / scond for scond in DATA_SELECTION_V5NAMES]:
        os.makedirs(p, exist_ok=True)

    if len(DATA_SELECTION_V5NAMES) == 1:
        v5_enctype = DATA_SELECTION_V5NAMES[0]
        results_root = Path(f'{str(results_root)}_{v5_enctype}')


    DESIRED_OUTPUTS = [
        'raw',
        'thresh',
        'lab',
        'overlays',
        'error_maps',
        'probmaps',
        'metrics',
        'cls_overlays'
    ]

    label_name = 'encapsulins'


    class Randomizer(torch.nn.Module):
        """Fake module for producing correctly shaped random outputs in range [0, 1]"""
        def forward(self, x):
            return torch.rand(x.shape[0], 2, *x.shape[2:])

    if segmenter_path is None:
        # segmenter_dict = {
        #     'all': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10b/GA_all_dec98__UNet__22-10-05_04-22-48/model_step240000.pts',
        #     'hek': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10b/GA_hek_dec98__UNet__22-10-05_04-24-22/model_step240000.pts',
        #     'dro': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10b/GA_dro__UNet__22-10-05_04-26-13/model_step240000.pts',
        #     'qttm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10b/GA_qttm__UNet__22-10-05_04-25-24/model_step240000.pts',

        #     'mice': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10/MICE_2M-Qt_GA_mice__UNet__22-09-24_03-26-43/model_step240000.pts',

        #     'onlytm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_onlytm/GA_onlytm__UNet__22-09-24_03-32-39/model_step40000.pts',
        #     'all_notm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_notm/GA_notm_all__UNet__22-09-24_03-33-19/model_step160000.pts',
        #     'hek_notm': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_notm/GA_notm_hek__UNet__22-09-24_03-34-32/model_step160000.pts',
        # }
        segmenter_dict = {
            'all': '/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v13/GA_lrdec99__UNet__22-10-15_20-29-15/model_step240000.pts'
        }
        tr_setting = 'all'
        segmenter_paths = [segmenter_dict[tr_setting]]
        # segmenter_paths = [Randomizer()]  # produce random outputs instead
    else:
        segmenter_paths = [segmenter_path]

    if classifier_path is None:
        classifier_path = '/cajal/nvmescratch/users/mdraw/tum/patch_trainings_v14_dr5__t100/erasemaskbg___EffNetV2__22-10-21_02-44-53/model_step120000.pts'

    if not 'cls_overlays' in DESIRED_OUTPUTS:
        # Classifier not required, so we disable it and don't reference it
        classifier_path = ''

    assert len(segmenter_paths) == 1, 'Currently only one model is supported per inference run'
    for segmenter_path in segmenter_paths:
        # segmentername = os.path.basename(os.path.dirname(segmenter_path))

        # No tiling necessary on normal images
        tile_shape = None
        overlap_shape = None
        out_shape = None

        apply_softmax = True
        predictor = Predictor(
            model=segmenter_path,
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
                cout = sm.remove_small_holes(cout, 2000)
                cout = sm.remove_small_objects(cout, minsize)

                # Make iio.imwrite-able
                cout = cout.astype(np.uint8) * 255

                # out_path = eu(f'{results_path}/{basename}_{segmentername}_{kind}.png')
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

                if 'cls_overlays' in DESIRED_OUTPUTS:
                    rprops, cls_relabeled = iu.compute_rprops(
                        image=raw_img,
                        lab=cout > 0,
                        classifier_variant=classifier_path,
                        return_relabeled_seg=True,
                        allowed_classes=allowed_classes_for_classification,
                        minsize=minsize
                    )
                    cls_ov = utils.render_skimage_overlay(img=raw_img, lab=cls_relabeled, colors=iu.skimage_color_cycle)
                    iio.imwrite(eu(f'{results_path}/{basename}_overlay_cls.jpg'), cls_ov)
                    cls = utils.render_skimage_overlay(img=None, lab=cls_relabeled, colors=iu.skimage_color_cycle)
                    iio.imwrite(eu(f'{results_path}/{basename}_cls.png'), cls)

                    iu.save_properties_to_xlsx(properties=rprops, xlsx_out_path=results_path / f'{basename}_cls_table.xlsx')

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
                segmenter_path=segmenter_path,
                classifier_path=classifier_path,
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
                        segmenter_path=segmenter_path,
                        classifier_path=classifier_path,
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
