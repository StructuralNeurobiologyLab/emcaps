"""
Experimental interactive visualization tool for Encapsulin classification.
Based on https://napari.org/tutorials/segmentation/annotate_segmentation.html
"""

from mailbox import mbox
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

from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


# WARNING: This can quickly lead to OOM on systems with < 16 GB RAM

import torch


INTERACTIVE = False


def cc_label(lab):
    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(lab), 20)
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
    model = torch.jit.load(path, map_location='cpu').eval()
    model = torch.jit.optimize_for_inference(model)
    return model

classifier_model = load_torchscript_model(classifier_path)
seg_model = load_torchscript_model(segmenter_path)

image_raw = imageio.imread(data_root / '129/129.tif')
image_normalized = (image_raw.astype(np.float32) - 128.) / 128.


def segment(img):
    img = torch.from_numpy(img)[None, None]
    with torch.inference_mode():
        out = seg_model(img)
        pred = torch.argmax(out, dim=1)
        pred = pred[0].numpy().astype(np.int64)
    return pred


def action_segment():
    img_raw = viewer.layers[0].data
    img_normalized = (img_raw.astype(np.float32) - 128.) / 128.
    pred = segment(img_normalized)
    viewer.add_labels(pred)


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


if INTERACTIVE:
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
