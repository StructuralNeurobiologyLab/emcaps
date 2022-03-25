"""
Experimental interactive visualization tool for Encapsulin classification.
Based on https://napari.org/tutorials/segmentation/annotate_segmentation.html
"""

from os import major
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
import imageio
import yaml
import napari

from magicgui import magic_factory, widgets
from skimage import data
from skimage.feature import blob_log
from typing_extensions import Annotated

import napari
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import ImageData, LayerDataTuple, LabelsData
from napari.utils.notifications import show_info

from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from tiler import Tiler, Merger

# WARNING: This can quickly lead to OOM on systems with < 16 GB RAM

import torch


INTERACTIVE = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

DTYPE = torch.float16 if 'cuda' in str(device) else torch.float32


def cc_label(lab, minsize=20, noborder=True):
    # remove artifacts connected to image border
    if noborder:
        lab = clear_border(lab)
    cleared = remove_small_objects(lab, minsize)
    # cleared = ndimage.binary_dilation(cleared, iterations=4)  # Enlarge (provoke false mergers)

    # label image regions
    label_image = label(cleared)

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
class_info_path = './class_info.yaml'
with open(class_info_path) as f:
    class_info = yaml.load(f, Loader=yaml.FullLoader)
class_ids = class_info['class_ids']
class_names = {v: k for k, v in class_ids.items()}

# load the image and segment it

# USE_GT = False
USE_GT = True

data_root = Path('~/tum/Single-table_database/').expanduser()
segmenter_path = Path('~/tum/ptsmodels/unet_gdl_uni4_15k.pts').expanduser()
classifier_path = Path('~/tum/ptsmodels/effnet_s_40k_uni4a.pts').expanduser()

def load_torchscript_model(path):
    model = torch.jit.load(path, map_location=device).eval().to(DTYPE)
    model = torch.jit.optimize_for_inference(model)
    return model


def normalize(image: np.ndarray) -> np.ndarray:
    normalized = (image.astype(np.float32) - 128.) / 128.
    return normalized


classifier_model = load_torchscript_model(classifier_path)
seg_model = load_torchscript_model(segmenter_path)

image_raw = imageio.imread(data_root / '129/129.tif')
image_normalized = normalize(image_raw)


def segment(image: np.ndarray, thresh: float) -> np.ndarray:
    # return image > 0.9
    img = torch.from_numpy(image)[None, None].to(DTYPE)
    with torch.inference_mode():
        out = seg_model(img)
        # pred = torch.argmax(out, dim=1)
        pred = out[0, 1] > thresh
        pred = pred.numpy().astype(np.int64)
    return pred


def tiled_segment(image: np.ndarray, thresh: float, pbar=None) -> np.ndarray:
    tiler = Tiler(
        data_shape=image.shape,
        tile_shape=(256, 256),
        overlap=(32, 32),
        channel_dimension=None,
    )
    new_shape, padding = tiler.calculate_padding()
    tiler.recalculate(data_shape=new_shape)
    padded_image = np.pad(image, padding, mode='reflect')
    merger = Merger(tiler=tiler, window='overlap-tile')
    # pbar.max = len(tiler)
    for tile_id, tile in tiler(padded_image):
        # if pbar is not None:
        #     pbar.increment()
        merger.add(tile_id, segment(tile, thresh=thresh))
    pred = merger.merge(unpad=True, extra_padding=padding, dtype=np.int64)

    return pred



# For regionprops extra_properties
def rp_classify(region_mask, img):
    # patch_np = image[region_mask]
    patch_np = img.astype(np.float32)
    patch = torch.from_numpy(patch_np)[None, None]
    with torch.inference_mode():
        out = classifier_model(patch)
        pred = torch.argmax(out, dim=1)[0].item()
    return pred


def assign_class_names(pred_ids):
    pred_class_names = [class_names[pred] for pred in pred_ids]
    return pred_class_names





if not INTERACTIVE:
    viewer = napari.view_image(image_raw, name='image', rgb=False)
    print('Segmenting...')
    if USE_GT:
        sem_label_image = imageio.imread(data_root / '129/129_encapsulins.tif')
    else:
        sem_label_image = segment(image_normalized)

    label_image = cc_label(sem_label_image)

    print('Calculating rprops and classification...')
    # create the properties dictionary
    properties = regionprops_table(
        label_image=label_image,
        intensity_image=image_normalized,
        properties=('label', 'bbox', 'perimeter', 'area', 'solidity'),
        extra_properties=[rp_classify]
    )
    properties['pred_classname'] = assign_class_names(properties['rp_classify'])
    properties['circularity'] = circularity(
        properties['perimeter'], properties['area']
    )


    # create the bounding box rectangles
    bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])

    # specify the display parameters for the text
    text_parameters = {
        'text': 'id: {label:03d}, circularity: {circularity:.2f}, solidity: {solidity:.2f}\nclass: {pred_classname}',
        'size': 14,
        'color': 'black',
        'anchor': 'upper_left',
        'translation': [-3, 0],
    }

    # add the labels
    # TODO: One label layer per patchclassify class, each with different color?
    label_layer = viewer.add_labels(label_image, name='segmentation')

    shapes_layer = viewer.add_shapes(
        bbox_rects,
        face_color='transparent',
        edge_color='green',
        properties=properties,
        text=text_parameters,
        name='bounding box',
    )


# TODO: Use https://github.com/the-lay/tiler

@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Segmenting...'})
def make_seg_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    threshold: Annotated[float, {"min": 0, "max": 1, "step": 0.1}] = 0.5,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide})
    def seg() -> LayerDataTuple:
        # this is the potentially long-running function
        img_normalized = normalize(image)
        pred = segment(img_normalized, thresh=threshold)
        # pred = tiled_segment(img_normalized, thresh=threshold, pbar=pbar)

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
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 10}] = 20,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide})
    def regions() -> LayerDataTuple:
        instance_labels = cc_label(labels, minsize=minsize)

        properties = regionprops_table(
            label_image=instance_labels,
            intensity_image=normalize(image),
            properties=('label', 'bbox', 'perimeter', 'area', 'solidity'),
            extra_properties=[rp_classify]
        )
        properties['pred_classname'] = assign_class_names(properties['rp_classify'])
        properties['circularity'] = circularity(
            properties['perimeter'], properties['area']
        )
        bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])
        text_parameters = {
            'text': 'id: {label:03d}, circularity: {circularity:.2f}\nclass: {pred_classname}',
            'size': 14,
            'color': 'black',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }

        majority_class = np.argmax(np.bincount(properties['rp_classify']))
        majority_class_name = assign_class_names([majority_class])[0]

        text_display =  f'Majority vote: {majority_class_name}'
        print(f'\n{text_display}')
        # resultlabel.label
        show_info(text_display)

        meta = dict(
            name='regions',
            # features={'class': properties['pred_classname'], 'majority_class_name': majority_class_name},
            edge_color='pred_classname',
            face_color='transparent',
            properties=properties,
            text=text_parameters,
            metadata={'majority_class_name': majority_class_name},
        )

        return (bbox_rects, meta, 'shapes')

    pbar.show()
    return regions()


if INTERACTIVE:
    # viewer = napari.Viewer()
    viewer = napari.view_image(image_raw[:600, :600].copy(), name='image')

    viewer.window.add_dock_widget(make_seg_widget(), name='Segmentation', area='right')
    viewer.window.add_dock_widget(make_regions_widget(), name='Region analysis', area='right')

if False and INTERACTIVE:
    viewer = napari.view_image(image_raw, name='image', rgb=False)  # TODO: Don't hardcode initial image

    button_layout = QVBoxLayout()
    process_btn = QPushButton("Full Process")
    process_btn.clicked.connect(action_segment)
    button_layout.addWidget(process_btn)

    action_widget = QWidget()
    action_widget.setLayout(button_layout)
    action_widget.setObjectName("Segmentation")
    viewer.window.add_dock_widget(action_widget)

    viewer.window._status_bar._toggle_activity_dock(True)

napari.run()
