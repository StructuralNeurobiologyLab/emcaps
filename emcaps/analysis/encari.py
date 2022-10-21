"""
Experimental interactive visualization tool for Encapsulin classification.
Based on https://napari.org/tutorials/segmentation/annotate_segmentation.html
"""

# TODO:
# - Tiling prediction
# - TTA

import logging
from pathlib import Path
import platform
import tempfile


import imageio.v3 as iio
import napari
import napari.utils
import numpy as np
import torch
import yaml
import pandas as pd
from magicgui import magic_factory, widgets
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari.utils.notifications import show_info
from scipy import ndimage
from skimage import morphology as sm
from typing_extensions import Annotated

from emcaps import utils
from emcaps.utils import inference_utils as iu
from emcaps.utils.colorlabel import color_dict_rgba

TMPPATH = '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()


# Set up logging
logger = logging.getLogger('encari')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{TMPPATH}/encari.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


# WARNING: This can quickly lead to OOM on systems with <= 8 GB RAM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

DTYPE = torch.float16 if 'cuda' in str(device) else torch.float32

repo_root = Path(__file__).parents[2]

# TODO: Replace with utils.* refs
# Load mapping from class names to class IDs
class_info_path = repo_root / 'emcaps/class_info.yaml'
with open(class_info_path) as f:
    class_info = yaml.load(f, Loader=yaml.FullLoader)
CLASS_IDS = class_info['class_ids_v5']
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}


# TODO: Determine color map in class_info.yaml
from emcaps.utils.inference_utils import class_colors, skimage_color_cycle

print(class_colors)
print(skimage_color_cycle)

_global_state = {}


# TODO: Support invalidation for low-confidence predictions


def get_default_xlsx_output_path() -> str:
    if (src_spath := _global_state.get('src_path')) is not None:
        default_path = src_spath.with_stem(f'{src_spath.stem}_cls.xlsx')
    else:
        default_path = f'{TMPPATH}/ec-cls.xlsx'
    return default_path


def get_default_overlay_output_path() -> str:
    if (src_spath := _global_state.get('src_path')) is not None:
        default_path = src_spath.with_stem(f'{src_spath.stem}_cls.png')
    else:
        default_path = f'{TMPPATH}/ec-cls.png'
    return default_path


# TODO: Support optional tiling
@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Segmenting...'})
def make_seg_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    segmenter_variant: Annotated[str, {'choices': list(iu.segmenter_urls.keys())}] = 'unet_all_v13',
    threshold: Annotated[float, {"min": 0, "max": 1, "step": 0.1}] = 0.5,
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 50}] = 60,
    assign_unique_instance_ids: bool = False,
) -> FunctionWorker[LayerDataTuple]:

    @thread_worker(connect={'returned': pbar.hide})
    def seg() -> LayerDataTuple:
        img_normalized = iu.normalize(image)

        pred = iu.segment(img_normalized, thresh=threshold, segmenter_variant=segmenter_variant)

        # Postprocessing:
        pred = sm.remove_small_holes(pred, 2000)
        pred = sm.remove_small_objects(pred, minsize)

        if assign_unique_instance_ids:
            pred, _ = ndimage.label(pred)

        meta = dict(
            name='segmentation',
            color=class_colors.copy(),
            seed=0,
        )
        # return a "LayerDataTuple"
        return (pred, meta, 'labels')

    # show progress bar and return worker
    pbar.show()
    return seg()


# TODO: GUI progress indicator
@magic_factory(pbar={'visible': False, 'max': 0, 'label': 'Analyzing regions...'})
def make_regions_widget(
    pbar: widgets.ProgressBar,
    image: ImageData,
    labels: LabelsData,
    classifier_variant: Annotated[str, {'choices': list(iu.classifier_urls.keys())}] = 'effnet_m_all_v14',
    allowed_classes: Annotated[list[str], {'choices': utils.CLASS_GROUPS['simple_hek'], 'allow_multiple': True}] = utils.CLASS_GROUPS['simple_hek'],
    minsize: Annotated[int, {"min": 0, "max": 1000, "step": 50}] = 60,
    maxsize: Annotated[int, {"min": 1, "max": 2000, "step": 50}] = 1000,
    mincircularity: Annotated[float, {"min": 0.0, "max": 1.0, "step": 0.1}] = 0.8,
    shape_type: Annotated[str, {'choices': ['ellipse', 'rectangle', 'none']}] = 'none',
    inplace_relabel: bool = True,
    xlsx_output_path: str = get_default_xlsx_output_path(),
    # xlsx_output_path: Path = Path(get_default_xlsx_output_path()),  # Path picker always expects existing files, so use str instead:
) -> FunctionWorker[LayerDataTuple]:

    xlp = xlsx_output_path

    @thread_worker(connect={'returned': pbar.hide})
    def regions() -> LayerDataTuple:
        # img_normalized = normalize(image)

        properties = iu.compute_rprops(
            image=image,
            lab=labels,
            classifier_variant=classifier_variant,
            minsize=minsize,
            maxsize=maxsize,
            min_circularity=mincircularity,
            inplace_relabel=inplace_relabel,
            allowed_classes=allowed_classes
        )

        nonlocal xlp
        # Save region info to .xlsx file
        if not isinstance(xlp, Path):
            xlp = Path(xlp)
        iu.save_properties_to_xlsx(properties=properties, xlsx_out_path=xlp)

        # If inplace_relabel is true, this has modified the labels from the
        # caller in place without napari suspecting anything, so we'll refresh manually
        if inplace_relabel:
            for layer in napari.current_viewer().layers:
                layer.refresh()

        if shape_type == 'none':
            # Return early, don't construct a shape layer
            return


        bbox_rects = iu.make_bbox([properties[f'bbox-{i}'] for i in range(4)])
        text_parameters = {
            # 'text': 'id: {label:03d}, circularity: {circularity:.2f}\nclass: {pred_classname}',
            # 'text': 'id: {label:03d}\nclass: {pred_classname}',
            'text': '{class_name}',
            'size': 14,
            'color': 'blue',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        }

        majority_class_name = iu.compute_majority_class_name(class_preds=properties['class_id'])

        text_display =  f'Majority vote: {majority_class_name}'
        print(f'\n{text_display}')
        show_info(text_display)

        meta = dict(
            name='regions',
            shape_type=shape_type,
            # edge_color_cycle=color_cycle,
            # face_color_cycle=color_cycle,
            # face_colormap=napcolormap,
            # edge_colormap=napcolormap,
            # opacity=0.35,
            properties=properties,
            text=text_parameters,
            metadata={'majority_class_name': majority_class_name},
            features=properties['class_id'],
        )

        match shape_type:
            case 'ellipse':
                meta.update({
                    'edge_color': 'white',
                    'face_color': 'transparent',
                })
            case 'rectangle':
                meta.update({
                    'edge_color': 'white',
                    'face_color': 'transparent',
                })
            # case 'none' is already handled by the early return above
            case _:
                raise ValueError(f'Unsupported shape_type {shape_type}')

        return (bbox_rects, meta, 'shapes')

    if labels is None:
        raise ValueError('Please select segmentation labels for region analysis')

    pbar.show()
    return regions()


def render_overlay(
    image: ImageData,
    labels: LabelsData
) -> LayerDataTuple:
    overlay = utils.render_skimage_overlay(img=image, lab=labels, colors=skimage_color_cycle)
    meta = dict(name='overlay')
    return (overlay, meta, 'image')


def export_overlay(
    image: ImageData,
    labels: LabelsData,
    output_path: str = get_default_overlay_output_path(),
) -> None:
    # # TODO: HACK
    # if (segpath := _global_state.get('seg_path')) is not None:
    #     output_path = segpath.with_stem(segpath.stem.replace('thresh', 'cls'))

    overlay = utils.render_skimage_overlay(img=image, lab=labels, colors=skimage_color_cycle)
    iio.imwrite(output_path, overlay)
    show_info(f'Exported overlay to {output_path}')


def main():

    import argparse
    parser = argparse.ArgumentParser(description='Napari emcaps')
    parser.add_argument('paths', nargs='*', help='Path to input file(s)', default=None)
    args = parser.parse_args()
    ipaths = args.paths

    viewer = napari.Viewer(title='EMcapsulin demo')

    if ipaths == ['test136']:
        eip = Path('~/tum/Single-table_database/136/136.tif').expanduser()
        ilp = Path('~/tum/Single-table_database/136/136_encapsulins.tif').expanduser()
        eimg = iio.imread(eip)[600:900, 600:900].copy()
        elab = iio.imread(ilp)[600:900, 600:900].copy() > 0
        viewer.add_image(eimg, name='img')
        viewer.add_labels(elab, name='lab', seed=0, color=class_colors.copy())
        ipaths = []

    if ipaths and len(ipaths) > 0:
        img_path = Path(ipaths[0]).expanduser()
        img = iio.imread(img_path)
        viewer.add_image(img, name=img_path.name)
        # print(img_path.stem)

        _global_state['src_path'] = img_path
        _global_state['src_name'] = img_path.stem
        # Reassign default paths based on updated image source info
        # TODO: WIP, Make this actually reassign paths:
        # make_regions_widget.xlsx_output_path = get_default_xlsx_output_path()
        # export_overlay.output_path = get_default_overlay_output_path()

        # # TEMPORARY HACK for fast segmap opening TODO remove/rewrite
        # seg_path = img_path.with_stem(img_path.stem.replace('raw', 'thresh'))
        # seg = iio.imread(seg_path)
        # seg = (seg > 0).astype(np.int32)
        # viewer.add_labels(seg, name=seg_path.name, seed=0, color=class_colors.copy())
        # _global_state['seg_path'] = seg_path

    if ipaths and len(ipaths) > 1:
        lab_path = Path(ipaths[1]).expanduser()
        lab = iio.imread(lab_path)
        viewer.add_labels(lab > 0, name=lab_path.name, seed=0, color=class_colors.copy())

    viewer.window.add_dock_widget(make_seg_widget(), name='Segmentation', area='right')
    viewer.window.add_dock_widget(make_regions_widget(), name='Region analysis', area='right')
    viewer.window.add_function_widget(render_overlay, name='Render overlay image', area='right')
    viewer.window.add_function_widget(export_overlay, name='Render and export overlay image', area='right')

    napari.run()


if __name__ == '__main__':
    main()