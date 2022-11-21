"""Inference script for segmentation prediction and evaluation, based on
raw images and a model trained with `segtrain.py`."""


# TODO: Reconfigure to do one example batch processing run based on single table database sheet

import logging
import os
from pathlib import Path
from os.path import expanduser as eu
import random

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


logger = logging.getLogger('emcaps-segment')


def produce_metrics(thresh, results_root, segmenter_path, classifier_path, data_selection, m_targets, m_preds, m_probs):
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
- X_cls.jpg: Classifier-based recolorization of segmentation
- X_overlay_cls.jpg: Classifier-based recolorization of segmentation, overlayed on raw image
- X_cls_table.xlsx: Classification / region analysis results in tabular form

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


def is_empty(targets) -> bool:
    # No positive value found in any target -> metrics are undefined, so skip this group
    return len(targets) == 0 or np.concatenate(targets, axis=None).max() == 0

def find_vx_val_images(isplit_data_path: Path | str, group_name: str, sheet_path: Path | str):
    """Find paths to all raw validation images of split vx"""
    val_img_paths = []
    for p in Path(isplit_data_path).rglob('*_val.png'):  # Look for all validation raw images recursively
        if group_name == 'everything' or utils.is_in_data_group(path_or_num=p, group_name=group_name, sheet_path=sheet_path):
            val_img_paths.append(p)
    return val_img_paths


@hydra.main(version_base='1.2', config_path='../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    _hydra_cwd = hydra.core.hydra_config.HydraConfig.get()['run']['dir']
    logger.info(f'Writing logs and config to {_hydra_cwd}')
    # (Path(cfg.segment.results_root) / '.hydra_outputs').symlink_to(_hydra_cwd)

    tta_num = cfg.segment.tta_num
    minsize = cfg.minsize
    segmenter_path = cfg.segment.segmenter
    classifier_path = cfg.segment.classifier
    thresh = cfg.segment.thresh

    METRICS_KEYS = ['dsc', 'iou', 'precision', 'recall']

    # allowed_classes_for_classification = utils.CLASS_GROUPS['simple_hek']
    allowed_classes_for_classification = cfg.segment.constrain_classifier

    constraint_signature = ''  # Unconstrained
    if allowed_classes_for_classification != utils.CLASS_GROUPS['simple_hek']:
        constraint_signature = '_constrained'
        for ac in allowed_classes_for_classification:
            constraint_signature = f'{constraint_signature}_{ac}'

    pre_predict_transform = transforms.Compose([
        transforms.Normalize(mean=cfg.dataset_mean, std=cfg.dataset_std)
    ])

    results_root = Path(cfg.segment.results_root)

    inp_path = cfg.segment.inp_path
    if inp_path is None:
        # Use image database with metadata and human labels, enabling metrics calculation and error visualization etc.
        use_database = True 
        enable_zero_labels = False  # Expect labels to be available
        img_paths = find_vx_val_images(isplit_data_path=cfg.isplit_data_path, group_name=cfg.ev_group, sheet_path=cfg.sheet_path)

        all_dataset_names = utils.get_unique_entries_under(column_name='Dataset Name', sheet_path=cfg.sheet_path)
        all_image_types = utils.get_unique_entries_under(column_name='Image Type', sheet_path=cfg.sheet_path)
    else:
        # Custom path, can't use image database with metadata and human labels, so disable all quantitative eval features
        use_database = False
        enable_zero_labels = True  # Makes everything work without human labels
        inp_path = Path(inp_path).expanduser()
        if inp_path.is_file():
            img_paths = [inp_path]
            if cfg.segment.relative_out_path:
                results_root = Path(img_paths[0]).parent
        elif inp_path.is_dir():
            img_paths = list(inp_path.rglob('*'))
            if cfg.segment.relative_out_path:
                results_root = inp_path.parent / f'{inp_path.name}_seg_{cfg.tr_group}'
        else:
            raise FileNotFoundError(f'{inp_path} not found')


    results_root.mkdir(exist_ok=True)
    logger.info(f'Writing outputs to {results_root}')
    desired_outputs = cfg.segment.desired_outputs

    label_name = cfg.label_name

    if segmenter_path == 'auto':
        segmenter_path = f'unet_{cfg.tr_group}_{cfg.v}'
        logger.info(f'Using default segmenter {segmenter_path} based on other config values')
        segmenter_model = iu.get_model(segmenter_path)
    elif segmenter_path == 'randomizer':
        logger.info('Using randomizer test model')
        segmenter_model = iu.Randomizer()  # Produce random outputs
    else:
        logger.info(f'Using segmenter {segmenter_path}')
        segmenter_model = iu.get_model(segmenter_path)

    if not 'cls_overlays' in desired_outputs:
        # Classifier not required, so we disable it and don't reference it
        classifier_path = ''

    if classifier_path == 'auto':
        classifier_path = f'effnet_{cfg.tr_group}_{cfg.v}'
        logger.info(f'Using default classifier {classifier_path} based on other config values')
    elif classifier_path != '':
        logger.info(f'Using classifier {classifier_path}')

    apply_softmax = True
    predictor = Predictor(
        model=segmenter_model,
        device=None,
        float16=True,
        transform=pre_predict_transform,
        augmentations=tta_num,
        apply_softmax=apply_softmax,
    )

    dfdict = {mkey: {} for mkey in METRICS_KEYS}
    if use_database:
        # Base level: Initialize with 'All' aggregate key, populate more specific fields below
        per_group_results = {'All': {'targets': [], 'preds': [], 'probs': []}}
        # "Dataset Name"-level: Initialize with 'All' aggregate key, populate more specific fields below
        for dataset_name in all_dataset_names:
            per_group_results[dataset_name] = {'All': {'targets': [], 'preds': [], 'probs': []}}
            dfdict[dataset_name] = {mkey: {} for mkey in METRICS_KEYS}
            # "Image Type"-level
            for image_type in all_image_types:
                per_group_results[dataset_name][image_type] = {'targets': [], 'preds': [], 'probs': []}
                # dfdict[dataset_name][image_type] = {}

    # img_paths = random.sample(img_paths, 5)  # Uncomment to test a small sample
    assert len(img_paths) > 0
    for img_path in img_paths:
        inp = np.array(iio.imread(img_path), dtype=np.float32)[None][None]  # (N=1, C=1, H, W)
        out = predictor.predict(inp)
        out = out.numpy()
        basename = os.path.splitext(os.path.basename(img_path))[0]

        if use_database:
            dataset_name = utils.get_image_entry(img_path, column_name='Dataset Name', sheet_path=cfg.sheet_path)
            image_type = utils.get_image_entry(img_path, column_name='Image Type', sheet_path=cfg.sheet_path)
            results_path = results_root / dataset_name
            results_path.mkdir(exist_ok=True)
        else:
            results_path = results_root

        assert out.shape[1] == 2
        cout = out[0, 1]  # Binary segmentation -> only export channel 1
        cout = (cout * 255.).astype(np.uint8)
        cout = cout > thresh
        # kind = f'thresh{thresh}'
        kind = f'thresh'

        # Postprocessing:
        cout = sm.remove_small_holes(cout, 2000)
        cout = sm.remove_small_objects(cout, minsize)

        # Make iio.imwrite-able
        cout = cout.astype(np.uint8) * 255

        # out_path = eu(f'{results_path}/{basename}_{segmentername}_{kind}.png')
        out_path = eu(f'{results_path}/{basename}_{kind}.png')
        logger.info(f'Writing inference result to {out_path}')
        if 'thresh' in desired_outputs:
            iio.imwrite(out_path, cout)

        if 'probmaps' in desired_outputs:
            probmap = (out[0, 1] * 255.).astype(np.uint8)
            probmap_path = eu(f'{results_path}/{basename}_probmap.jpg')
            iio.imwrite(probmap_path, probmap)

        raw_img = iio.imread(img_path)

        # Write raw and gt labels
        if enable_zero_labels:
            # Make all-zero label image, to make handling label-free images easier below.
            # TODO: Remove the need for zero_labels
            lab_img = np.zeros_like(raw_img, dtype=np.uint8)
        else:
            lab_path = f'{str(img_path)[:-4]}_{label_name}.png'
            lab_img = np.array(iio.imread(lab_path))
            lab_img = ((lab_img > 0) * 255).astype(np.uint8)  # Binarize (binary training specific!)

        if 'raw' in desired_outputs:
            iio.imwrite(eu(f'{results_path}/{basename}_raw.jpg'), raw_img)
        if use_database and 'lab' in desired_outputs:
            iio.imwrite(eu(f'{results_path}/{basename}_lab.png'), lab_img)

        if 'overlays' in desired_outputs:
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

            if not enable_zero_labels:
                iio.imwrite(eu(f'{results_path}/{basename}_overlay_lab.jpg'), lab_overlay)
            iio.imwrite(eu(f'{results_path}/{basename}_overlay_pred.jpg'), pred_overlay)

        if 'cls_overlays' in desired_outputs:
            if classifier_path == '' or classifier_path is None:
                raise ValueError(f'classifier_path needs to be set for cls_overlays')
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

        if use_database and 'error_maps' in desired_outputs:
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

        if use_database:
            per_group_results[dataset_name][image_type]['targets'].append(m_target)
            per_group_results[dataset_name][image_type]['preds'].append(m_pred)
            per_group_results[dataset_name][image_type]['probs'].append(m_prob)
            # Aggregate over all image_types: Fill into 'All' bucket regardless of image_type
            per_group_results[dataset_name]['All']['targets'].append(m_target)
            per_group_results[dataset_name]['All']['preds'].append(m_pred)
            per_group_results[dataset_name]['All']['probs'].append(m_prob)
            # Aggregate over all dataset_names and all image_types: Fill everything into 'All' bucket
            per_group_results['All']['targets'].append(m_target)
            per_group_results['All']['preds'].append(m_pred)
            per_group_results['All']['probs'].append(m_prob)

        if 'argmax' in desired_outputs:
            # Argmax of channel probs
            pred = np.argmax(out, 1)[0]
            # plab = skimage.color.label2rgb(pred, bg_label=0)
            plab = skimage.color.label2rgb(pred, colors=['red', 'green', 'blue', 'purple', 'brown', 'magenta'], bg_label=0)
            out_path = eu(f'{results_path}/{basename}_argmax_{modelname}.jpg')
            iio.imwrite(out_path, plab)

    if use_database and 'metrics' in desired_outputs:
        # Initialize metric value storage
        dfdict = {}
        for mkey in METRICS_KEYS:
            dfdict[mkey] = {}
            for dataset_name in all_dataset_names:
                dfdict[mkey][dataset_name] = {}

        logger.info('Calculating global metrics...')
        # 1. Global metrics (All), aggregate over all images, regardless of dataset_name and image_type
        global_metrics_dict = produce_metrics(
            thresh=thresh,
            results_root=results_root,
            segmenter_path=segmenter_path,
            classifier_path=classifier_path,
            data_selection=all_dataset_names,
            m_targets=per_group_results['All']['targets'],
            m_preds=per_group_results['All']['preds'],
            m_probs=per_group_results['All']['probs']
        )
        assert list(global_metrics_dict.keys()) == METRICS_KEYS

        for mkey, mval in global_metrics_dict.items():
            dfdict[mkey]['All'] = mval

        # 2. Per dataset_name: first, aggregate over all images of one dataset_name, regardless of image_type
        for dataset_name in all_dataset_names:
            if is_empty(per_group_results[dataset_name]['All']['targets']):
                continue
            logger.info(f'Calculating metrics of group {dataset_name}...')
            dataset_name_metrics_dict = produce_metrics(
                thresh=thresh,
                results_root=results_root / dataset_name,
                segmenter_path=segmenter_path,
                classifier_path=classifier_path,
                data_selection=f'{dataset_name}',
                m_targets=per_group_results[dataset_name]['All']['targets'],
                m_preds=per_group_results[dataset_name]['All']['preds'],
                m_probs=per_group_results[dataset_name]['All']['probs']
            )
            for mkey, mval in dataset_name_metrics_dict.items():
                dfdict[mkey][dataset_name]['All'] = mval


            # 3. Per dataset_name and image_type (nested).
            for image_type in all_image_types:
                if is_empty(per_group_results[dataset_name][image_type]['targets']):
                    continue
                logger.info(f'Calculating metrics of group {dataset_name} > {image_type}...')
                image_type_metrics_dict = produce_metrics(
                    thresh=thresh,
                    results_root=results_root / dataset_name,
                    segmenter_path=segmenter_path,
                    classifier_path=classifier_path,
                    data_selection=f'{dataset_name} > {[image_type]}',
                    m_targets=per_group_results[dataset_name][image_type]['targets'],
                    m_preds=per_group_results[dataset_name][image_type]['preds'],
                    m_probs=per_group_results[dataset_name][image_type]['probs']
                )
                for mkey, mval in image_type_metrics_dict.items():
                    dfdict[mkey][dataset_name][image_type] = mval

        for mkey, mdict in dfdict.items():
            mdf = pd.DataFrame.from_dict(mdict)
            mdf = mdf.reindex(columns=sorted(mdf.columns))  # Sort alphabetically so "All" is first
            mdf = mdf.round(2)  # Round everything to 2 decimal places
            mdf.to_excel(results_root / f'metrics_{mkey}.xlsx')
            mdf.to_html(results_root / f'metrics_{mkey}.html')

if __name__ == '__main__':
    main()
