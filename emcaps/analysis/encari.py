"""
Experimental interactive visualization tool for Encapsulin classification.
Based on https://napari.org/tutorials/segmentation/annotate_segmentation.html
"""

# TODO:
# - Tiling prediction
# - TTA
# - Fix classification


from pathlib import Path

import imageio.v3 as iio
import napari
import numpy as np
import torch
import yaml
import ubelt as ub
from magicgui import magic_factory, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari.utils.notifications import show_info
from scipy import ndimage
from skimage import data
from skimage.measure import label, regionprops_table, regionprops
from skimage import morphology as sm
from skimage.segmentation import clear_border
from typing_extensions import Annotated

# WARNING: This can quickly lead to OOM on systems with < 16 GB RAM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

DTYPE = torch.float16 if 'cuda' in str(device) else torch.float32

def cc_label(lab, minsize=20, maxsize=600, noborder=True, min_circularity=0.8):
    # remove artifacts connected to image border
    if noborder:
        lab = clear_border(lab)
    cleared = sm.remove_small_objects(lab, minsize)

    # label image regions
    label_image = label(cleared)

    rprops = regionprops(label_image, extra_properties=[circularity])
    for rp in rprops:
        if circularity(rp.perimeter, rp.area) < min_circularity:
            label_image[rp.slice] = 0
        elif rp.area > maxsize:
            label_image[rp.slice] = 0

    # TODO: ^ Don't remove complete slice but only the actually covered area
    # TODO: Relabel?

    return label_image



def circularity(perimeter, area):
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

# Load mapping from class names to class IDs
class_info_path = './emcaps/class_info.yaml'
with open(class_info_path) as f:
    class_info = yaml.load(f, Loader=yaml.FullLoader)
CLASS_IDS = class_info['class_ids_v5']
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}

# load the image and segment it


# TODO: Path updates
# data_root = Path('~/tum/Single-table_database/').expanduser()
# segmenter_path = Path('~/tum/ptsmodels/unet_gdl_v7_hek_160k.pts').expanduser()
# classifier_path = Path('~/tum/ptsmodels/effnet_m_v7_hek_80k.pts').expanduser()

# repo_root = Path(__file__).parents[2]

# segmenter_path = Path('./unet_v7_all.pts')
# classifier_path = Path('./effnet_m_v7_hek_80k.pts')


segmenter_path = ub.grabdata('https://github.com/mdraw/model-test/releases/download/v7/unet_v7_all.pts', hash_prefix='b23bd76dc81')
# classifier_path = ub.grabdata('https://github.com/mdraw/model-test/releases/download/v7/effnet_m_v7_hek_80k.pts', hash_prefix='b8eb59038a')
classifier_path = ub.grabdata('https://github.com/mdraw/model-test/releases/download/v7/effnet_s_v7_hek_80k.pts', hash_prefix='78989aeeb8')

def load_torchscript_model(path):
    model = torch.jit.load(path, map_location=device).eval().to(DTYPE)
    model = torch.jit.optimize_for_inference(model)
    return model


def normalize(image: np.ndarray) -> np.ndarray:
    normalized = (image.astype(np.float32) - 128.) / 128.
    assert normalized.min() >= -1.
    assert normalized.max() <= 1.
    return normalized


classifier_model = load_torchscript_model(classifier_path)
seg_model = load_torchscript_model(segmenter_path)

# image_raw = iio.imread(data_root / '129/129.tif')
# image_normalized = normalize(image_raw)


def segment(image: np.ndarray, thresh: float) -> np.ndarray:
    # return image > 0.9
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
    inp = torch.from_numpy(nobg)[None, None]
    with torch.inference_mode():
        out = classifier_model(inp)
        pred = torch.argmax(out, dim=1)[0].item()
    return pred


def assign_class_names(pred_ids):
    pred_class_names = [CLASS_NAMES[pred] for pred in pred_ids]
    return pred_class_names


def compute_rprops(image, labels, minsize, maxsize):
    instance_labels = cc_label(labels, minsize=minsize, maxsize=maxsize)

    properties = regionprops_table(
        label_image=instance_labels,
        # intensity_image=normalize(image),
        intensity_image=image,  # normalized by caller
        properties=('label', 'bbox', 'perimeter', 'area', 'solidity'),
        extra_properties=[rp_classify]
    )
    properties['pred_classname'] = assign_class_names(properties['rp_classify'])
    properties['circularity'] = circularity(
        properties['perimeter'], properties['area']
    )
    
    return properties


def compute_majority_class_name(class_preds):
    majority_class = np.argmax(np.bincount(class_preds))
    majority_class_name = assign_class_names([majority_class])[0]
    return majority_class_name


# TODO: Make tiling optional
@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Segmenting...'})
def make_seg_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    threshold: Annotated[float, {"min": 0, "max": 1, "step": 0.1}] = 0.5,
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 50}] = 150,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide})
    def seg() -> LayerDataTuple:
        img_normalized = normalize(image)

        pred = segment(img_normalized, thresh=threshold)
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


@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Analyzing regions...'})
def make_regions_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    labels: LabelsData,
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 50}] = 150,
    maxsize: Annotated[int, {"min": 1, "max": 2000, "step": 50}] = 1000,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide})
    def regions() -> LayerDataTuple:
        img_normalized = normalize(image)

        properties = compute_rprops(image=img_normalized, labels=labels, minsize=minsize, maxsize=maxsize)
        bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])
        text_parameters = {
            # 'text': 'id: {label:03d}, circularity: {circularity:.2f}\nclass: {pred_classname}',
            # 'text': 'id: {label:03d}\nclass: {pred_classname}',
            'text': '{pred_classname}',
            'size': 14,
            'color': 'blue',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }

        majority_class_name = compute_majority_class_name(class_preds=properties['rp_classify'])

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

def main():

    viewer = napari.Viewer()
    # viewer = napari.view_image(image_raw[:600, :600].copy(), name='image')

    viewer.window.add_dock_widget(make_seg_widget(), name='Segmentation', area='right')
    viewer.window.add_dock_widget(make_regions_widget(), name='Region analysis', area='right')

    napari.run()


if __name__ == '__main__':
    main()