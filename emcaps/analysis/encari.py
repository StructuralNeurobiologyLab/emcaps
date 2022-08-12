"""
Experimental interactive visualization tool for Encapsulin classification.
Based on https://napari.org/tutorials/segmentation/annotate_segmentation.html
"""

# TODO:
# - Tiling prediction
# - TTA


from functools import lru_cache
from pathlib import Path
from typing import Literal

import argparse
import logging
import imageio.v3 as iio
import napari
import numpy as np
import torch
import tqdm
import yaml
import ubelt as ub
from magicgui import magic_factory, widgets
from napari.qt.threading import FunctionWorker, thread_worker, GeneratorWorker
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari.utils.notifications import show_info
from napari.utils.progress import progrange, progress
from scipy import ndimage
from skimage import data
from skimage.measure import label, regionprops_table, regionprops
from skimage.measure._regionprops import _props_to_dict
from skimage import morphology as sm
from skimage.segmentation import clear_border
from typing_extensions import Annotated

from emcaps.analysis.radial_patchlineplots import measure_outer_disk_radius

# Set up logging
logger = logging.getLogger('encari')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'/tmp/encari.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


# WARNING: This can quickly lead to OOM on systems with <= 8 GB RAM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

DTYPE = torch.float16 if 'cuda' in str(device) else torch.float32


def calculate_circularity(perimeter, area):
    """Calculate the circularity of the region

    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region

    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity


def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect

repo_root = Path(__file__).parents[2]


# Load mapping from class names to class IDs
class_info_path = repo_root / 'emcaps/class_info.yaml'
with open(class_info_path) as f:
    class_info = yaml.load(f, Loader=yaml.FullLoader)
CLASS_IDS = class_info['class_ids_v5']
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}

# load the image and segment it


# TODO: Path updates
# data_root = Path('~/tum/Single-table_database/').expanduser()
# image_raw = iio.imread(data_root / '129/129.tif')

# segmenter_path = Path('~/tum/ptsmodels/unet_gdl_v7_hek_160k.pts').expanduser()
# classifier_path = Path('~/tum/ptsmodels/effnet_m_v7_hek_80k.pts').expanduser()


# segmenter_path = Path('./unet_v7_all.pts')
# classifier_path = Path('./effnet_m_v7_hek_80k.pts')


segmenter_path = ub.grabdata('https://github.com/mdraw/model-test/releases/download/v7/unet_v7_all.pts')
# classifier_path = ub.grabdata('https://github.com/mdraw/model-test/releases/download/v7/effnet_m_v7_hek_80k.pts', hash_prefix='b8eb59038a')
classifier_path = ub.grabdata('https://github.com/mdraw/model-test/releases/download/v7/effnet_m_v7_hek_80k.pts')


segmenter_urls = {
    'unet_all': 'https://github.com/mdraw/model-test/releases/download/v7/unet_gdl_v7_all_160k.pts',
    'unet_hek': 'https://github.com/mdraw/model-test/releases/download/v7/unet_gdl_v7_hek_160k.pts',
    'unet_dro': 'https://github.com/mdraw/model-test/releases/download/v7/unet_gdl_v7_dro_160k.pts',
}
classifier_urls = {
    'effnet_m_hek': 'https://github.com/mdraw/model-test/releases/download/v7/effnet_m_v7_hek_80k.pts',
    'effnet_s_hek': 'https://github.com/mdraw/model-test/releases/download/v7/effnet_s_v7_hek_80k.pts',
    'effnet_m_dro': 'https://github.com/mdraw/model-test/releases/download/v7/effnet_m_v7_dro_80k.pts',
    'effnet_s_dro': 'https://github.com/mdraw/model-test/releases/download/v7/effnet_m_v7_dro_80k.pts',
}
model_urls = {**segmenter_urls, **classifier_urls}


def get_model(variant: str) -> torch.jit.ScriptModule:
    if variant not in model_urls.keys():
        raise ValueError(f'Model variant {variant} not found. Valid choices are: {list(model_urls.keys())}')
    url = model_urls[variant]
    local_path = ub.grabdata(url, appname='emcaps')
    model = load_torchscript_model(local_path)
    return model


def load_torchscript_model(path: str) -> torch.jit.ScriptModule:
    model = torch.jit.load(path, map_location=device).eval().to(DTYPE)
    model = torch.jit.optimize_for_inference(model)
    return model


def normalize(image: np.ndarray) -> np.ndarray:
    normalized = (image.astype(np.float32) - 128.) / 128.
    assert normalized.min() >= -1.
    assert normalized.max() <= 1.
    return normalized


# classifier_model = load_torchscript_model(classifier_path)
# seg_model = load_torchscript_model(segmenter_path)

# image_normalized = normalize(image_raw)


def segment(image: np.ndarray, thresh: float, segmenter_variant: str) -> np.ndarray:
    # return image > 0.9
    seg_model = get_model(segmenter_variant)
    img = torch.from_numpy(image)[None, None].to(DTYPE)
    with torch.inference_mode():
        out = seg_model(img)
        # pred = torch.argmax(out, dim=1)
        pred = out[0, 1] > thresh
        pred = pred.numpy().astype(np.int64)
    return pred


def calculate_padding(current_shape, target_shape):
    """Calculate optimal padding for np.pad() to do central padding to target shape.
    
    If necessary (odd shape difference), pad 1 pixel more before than after."""
    halfdiff = np.subtract(target_shape, current_shape) / 2  # Half shape difference (float)
    fd = np.floor(halfdiff).astype(int)
    cd = np.ceil(halfdiff).astype(int)
    padding = (
        (fd[0], cd[0]),
        (fd[1], cd[1])
    )
    return padding


# For regionprops extra_properties
def rp_classify(region_mask, img, patch_shape=(49, 49), dilate_masks_by=5):
    # Pad to standardized patch shape
    padding = calculate_padding(current_shape=img.shape, target_shape=patch_shape)
    # padding = padding.clip(dilate_masks_by, None)  # Ensure that padding is non-negative and the mask can be dilated
    img = np.pad(img, padding)
    region_mask = np.pad(region_mask, padding)

    if dilate_masks_by > 0:
        disk = sm.disk(dilate_masks_by)
        region_mask = sm.binary_dilation(region_mask, selem=disk)
    nobg = img.astype(np.float32)
    # nobg[~region_mask] = 0
    # Image is already normalized, so we have to normalize the 0 masking intensity as well here
    norm0 = normalize(np.zeros(()))
    nobg[~region_mask] = norm0

    import string
    import random
    # fn = '/tmp/' + ''.join(random.choice(string.ascii_lowercase) for i in range(16)) + '.png'
    # iio.imwrite(fn, np.uint8(nobg * 128 + 128))

    inp = torch.from_numpy(nobg)[None, None]
    with torch.inference_mode():
        out = classifier_model(inp)
        pred = torch.argmax(out, dim=1)[0].item()
    return pred


def assign_class_names(pred_ids):
    pred_class_names = [CLASS_NAMES[pred] for pred in pred_ids]
    return pred_class_names



class ImageError(Exception):
    pass

def check_image(img, normalized=False, shape=None):
    _min = 0
    _max = 255
    if normalized:
        _min = normalize(np.array(_min))
        _max = normalize(np.array(_max))
    if img.min() < _min or img.max() > _max:
        raise ImageError(f'{img.min()=}, {img.max()=} not within expected range [{_min}, {_max}]')
    if shape is not None and not np.all(np.array(img.shape) == np.array(shape)):
        raise ImageError(f'{img.shape=}, but expected {shape}')


# TODO: Support invalidation for low-confidence predictions
def classify_patch(patch, classifier_variant):

    inp = normalize(patch)
    check_image(inp, normalized=True)

    classifier_model = get_model(classifier_variant)

    import string
    import random
    # fn = '/tmp/' + ''.join(random.choice(string.ascii_lowercase) for i in range(16)) + '.png'
    # iio.imwrite(fn, np.uint8(inp * 128 + 128))

    inp = torch.from_numpy(inp)[None, None]
    with torch.inference_mode():
        out = classifier_model(inp)
        pred = torch.argmax(out, dim=1)[0].item()
    return pred


def compute_rprops(image, lab, classifier_variant, minsize=150, maxsize=None, noborder=False, min_circularity=0.8):

    # Code mainly redundant with / copied from patchifyseg. TODO: Refactor into shared function

    DILATE_MASKS_BY = 5
    EC_REGION_RADIUS = 24
    # Add 1 to high region coordinate in order to arrive at an odd number of pixels in each dimension
    EC_REGION_ODD_PLUS1 = 1
    PATCH_WIDTH = EC_REGION_RADIUS * 2 + EC_REGION_ODD_PLUS1
    PATCH_SHAPE = (PATCH_WIDTH, PATCH_WIDTH)
    EC_MAX_AREA = (2 * EC_REGION_RADIUS)**2 if maxsize is None else maxsize
    EC_MIN_AREA = minsize
    EC_MAX_AREA = maxsize
    MIN_CIRCULARITY = min_circularity
    raw = image

    check_image(raw, normalized=False)

    # remove artifacts connected to image border
    if noborder:
        lab = clear_border(lab)
    cleared = sm.remove_small_objects(lab, minsize)

    # label image regions
    cc, n_comps = ndimage.label(cleared)

    rprops = regionprops(cc, raw)

    # epropdict = {k: np.full((len(rprops),), np.nan) for k in extra_prop_names}

    epropdict = {
        'class_id': np.empty((len(rprops),), dtype=np.uint8),
        'class_name': ['?'] * len(rprops),
        'circularity': np.empty((len(rprops),), dtype=np.float32),
        'radius2': np.empty((len(rprops),), dtype=np.float32),
        'is_invalid': np.empty((len(rprops),), dtype=bool),
    }

    for i, rp in enumerate(tqdm.tqdm(rprops, position=1, leave=True, desc='Analyzing regions', dynamic_ncols=True)):
        is_invalid = False
        centroid = np.round(rp.centroid).astype(np.int64)  # Note: This centroid is in the global coordinate frame
        if rp.area < EC_MIN_AREA or rp.area > EC_MAX_AREA:
            logger.info(f'Skipping: area size {rp.area} not within [{EC_MIN_AREA}, {EC_MAX_AREA}]')
            is_invalid = True
            continue  # Too small or too big (-> background component?) to be a normal particle
        circularity = np.nan
        if MIN_CIRCULARITY > 0:
            circularity = calculate_circularity(rp.perimeter, rp.area)
            if circularity < MIN_CIRCULARITY:
                logger.info(f'Skipping: circularity {circularity} below {MIN_CIRCULARITY}')
                is_invalid = True
                continue  # Not circular enough (probably a false merger)
            circularity = np.round(circularity, 2)  # Round for more readable logging

        lo = centroid - EC_REGION_RADIUS
        hi = centroid + EC_REGION_RADIUS + EC_REGION_ODD_PLUS1
        if np.any(lo < 0) or np.any(hi > raw.shape):
            logger.info(f'Skipping: region touches border')
            is_invalid = True
            continue  # Too close to image border

        xslice = slice(lo[0], hi[0])
        yslice = slice(lo[1], hi[1])

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
            is_invalid = True
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

        check_image(nobg_patch, normalized=False, shape=PATCH_SHAPE)

        class_id = classify_patch(patch=nobg_patch, classifier_variant=classifier_variant)
        class_name = CLASS_NAMES[class_id]

        # # Attribute assignments don't stick for _props_to_dict() for some reason
        # rp.class_id = class_id
        # rp.class_name = class_name
        # rp.circularity = circularity
        # rp.radius2 = radius2

        epropdict['class_id'][i] = class_id
        epropdict['class_name'][i] = class_name
        epropdict['circularity'][i] = circularity
        epropdict['radius2'][i] = radius2
        epropdict['is_invalid'][i] = is_invalid

        # iio.imwrite('/tmp/nobg-{i:03d}.png', nobg_patch)


    # print(rprops)
    # print(epropdict)
    # Can only assign builtin props here
    propdict = _props_to_dict(
        rprops, properties=['label', 'bbox', 'perimeter', 'area', 'solidity']
    )

    propdict.update(epropdict)

    # TODO: Prune invalid rps
    is_invalid = propdict['is_invalid']
    num_invalid = int(np.sum(is_invalid))
    logger.info(f'Pruning {num_invalid} regions due to filter criteria...')
    for k in propdict.keys():
        propdict[k] = np.delete(propdict[k], is_invalid)
    return propdict


def compute_majority_class_name(class_preds):
    majority_class = np.argmax(np.bincount(class_preds))
    majority_class_name = assign_class_names([majority_class])[0]
    return majority_class_name


# TODO: Make tiling optional
@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Segmenting...'})
def make_seg_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    segmenter_variant: Annotated[str, {'choices': list(segmenter_urls.keys())}] = 'unet_all',
    threshold: Annotated[float, {"min": 0, "max": 1, "step": 0.1}] = 0.5,
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 50}] = 150,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide})
    def seg() -> LayerDataTuple:
        img_normalized = normalize(image)

        pred = segment(img_normalized, thresh=threshold, segmenter_variant=segmenter_variant)
        # pred = tiled_segment(img_normalized, thresh=threshold, pbar=pbar)

        # Postprocessing:
        pred = sm.remove_small_holes(pred, 2000)
        pred = sm.remove_small_objects(pred, minsize)

        meta = dict(
            name='segmentation'
        )
        # return a "LayerDataTuple"
        return (pred, meta, 'labels')

    # show progress bar and return worker
    pbar.show()
    return seg()


def handle_yield(rp):
    print(rp)


@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Analyzing regions...'})
def make_regions_widget_generator_experimental(
    pbar: widgets.ProgressBar,
    image: ImageData,
    labels: LabelsData,
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 50}] = 150,
    maxsize: Annotated[int, {"min": 1, "max": 2000, "step": 50}] = 1000,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide, 'yielded': handle_yield}, progress={'total': 50})
    def gen_rprops():
        img_normalized = normalize(image)
        instance_labels = cc_label(labels, minsize=minsize, maxsize=maxsize)

        props = regionprops(
            label_image=instance_labels,
            intensity_image=img_normalized,
            extra_properties=[rp_classify]
        )

        for rp in props:
            _ = rp.rp_classify
            # Can't update from here, need yield
            yield rp

    def regions() -> LayerDataTuple:
        # img_normalized = normalize(image)

        props = []
        # for rp in gen_rprops():
        #     props.append(rp)

        properties = _props_to_dict(props, properties=['label', 'bbox', 'perimeter', 'area', 'solidity', 'rp_classify'])

        # for rp in props:
        #     for k in properties.keys():
        #         properties[k].append(rp[k])
        # for k in properties.keys():
        #     properties[k] = np.array(properties[k])

        properties['pred_classname'] = assign_class_names(properties['rp_classify'])
        properties['circularity'] = circularity(
            properties['perimeter'], properties['area']
        )

        # iio.imwrite('/tmp/image.png', np.uint8(image * 128 + 128))
        print(properties)


        bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])
        text_parameters = {
            # 'text': 'id: {label:03d}, circularity: {circularity:.2f}\nclass: {pred_classname}',
            # 'text': 'id: {label:03d}\nclass: {pred_classname}',
            'text': '{rp_classify}',# '{pred_classname}',
            'size': 14,
            'color': 'blue',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }

        majority_class_name = compute_majority_class_name(class_preds=properties['class_id'])

        text_display =  f'Majority vote: {majority_class_name}'
        print(f'\n{text_display}')
        show_info(text_display)

        meta = dict(
            name='regions',
            # features={'class': properties['pred_classname'], 'majority_class_name': majority_class_name},
            edge_color='pred_classname',
            edge_color_cycle=['red', 'green', 'blue', 'purple', 'orange', 'magenta', 'cyan', 'yellow'],
            face_color='transparent',
            properties=properties,
            text=text_parameters,
            metadata={'majority_class_name': majority_class_name},
        )

        return (bbox_rects, meta, 'shapes')

    pbar.show()
    return regions()

# TODO: Actual progress indicator

@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Analyzing regions...'})
def make_regions_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    labels: LabelsData,
    classifier_variant: Annotated[str, {'choices': list(classifier_urls.keys())}] = 'effnet_s_hek',
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 50}] = 150,
    maxsize: Annotated[int, {"min": 1, "max": 2000, "step": 50}] = 1000,
    mincircularity: Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.1}] = 0.8,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide})
    def regions() -> LayerDataTuple:
        # img_normalized = normalize(image)

        properties = compute_rprops(
            image=image,
            lab=labels,
            classifier_variant=classifier_variant,
            minsize=minsize,
            maxsize=maxsize,
            min_circularity=mincircularity,
        )
        bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])
        text_parameters = {
            # 'text': 'id: {label:03d}, circularity: {circularity:.2f}\nclass: {pred_classname}',
            # 'text': 'id: {label:03d}\nclass: {pred_classname}',
            'text': '{class_name}',
            'size': 14,
            'color': 'blue',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }

        majority_class_name = compute_majority_class_name(class_preds=properties['class_id'])

        text_display =  f'Majority vote: {majority_class_name}'
        print(f'\n{text_display}')
        show_info(text_display)

        meta = dict(
            name='regions',
            # features={'class': properties['pred_classname'], 'majority_class_name': majority_class_name},
            edge_color='class_id',
            edge_color_cycle=['magenta', 'cyan', 'blue', 'purple', 'orange', 'green', 'red', 'yellow'],
            face_color='transparent',
            properties=properties,
            text=text_parameters,
            metadata={'majority_class_name': majority_class_name},
        )

        return (bbox_rects, meta, 'shapes')

    if labels is None:
        raise ValueError('Please select segmentation labels for region analysis')

    pbar.show()
    return regions()

def main():

    import argparse
    parser = argparse.ArgumentParser(description='Napari emcaps')
    parser.add_argument('paths', nargs='*', help='Path to input file(s)', default=None)
    args = parser.parse_args()
    ipaths = args.paths

    viewer = napari.Viewer(title='EMcapsulin demo')
    # viewer.add_image(image_raw[:600, :600].copy(), name='image')

    if ipaths and len(ipaths) == 1 and ipaths == 'test136'
        eip = Path('~/tum/Single-table_database/136/136.tif').expanduser()
        ilp = Path('~/tum/Single-table_database/136/136_encapsulins.tif').expanduser()
        eimg = iio.imread(eip)[600:1200, 600:1200].copy()
        elab = iio.imread(ilp)[600:1200, 600:1200].copy()
        viewer.add_image(eimg, name='img')
        viewer.add_labels(elab, name='lab')

    if ipaths and len(ipaths) > 0:
        img_path = Path(ipaths[0]).expanduser()
        img = iio.imread(img_path)
        viewer.add_image(img, name=img_path.name)
        print(img_path.stem)
    if ipaths and len(ipaths) > 1:
        lab_path = Path(ipaths[1]).expanduser()
        lab = iio.imread(lab_path)
        viewer.add_labels(lab, name=lab_path.name)


    viewer.window.add_dock_widget(make_seg_widget(), name='Segmentation', area='right')
    viewer.window.add_dock_widget(make_regions_widget(), name='Region analysis', area='right')

    napari.run()


if __name__ == '__main__':
    main()