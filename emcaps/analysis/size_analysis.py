"""
Uses binary segmentation results (from segment.py) to estimate
typical size of encapsulin particles, by median or by finding the first KDE
peak.
"""


import os
from pathlib import Path
from os.path import expanduser as eu

import numpy as np
import imageio.v3 as iio
from numpy.core.fromnumeric import ndim
import skimage

from skimage import morphology as sm
from skimage.color import label2rgb
from scipy import ndimage

import matplotlib.pyplot as plt
import seaborn as sns


def eul(paths):
    """Shortcut for expanding all user paths in a list"""
    return [os.path.expanduser(p) for p in paths]


results_path = os.path.expanduser('~/tumresults')

image_numbers = [22, 32, 42]

# img_paths = eul([
#     f'~/tumdata/{i}/{i}.tif' for i in image_numbers
# ])

model_name = '15to54_encapsulins__UNet__21-09-16_03-10-26'

print_stats = {
    'median': False,
    'hpeak': True
}


lab_paths = [
    f'{results_path}/{model_name}/{i}_lab.png' for i in image_numbers
]
pred_paths = [
    f'{results_path}/{model_name}/{i}_{model_name}_thresh127.png' for i in image_numbers
]

# for p in lab_paths + pred_paths:
#     if not Path(p).is_file():
#         raise FileNotFoundError(p)


labs = [iio.imread(p) for p in lab_paths]
preds = [iio.imread(p) for p in pred_paths]

# Horizontally stack all images
# hs_labs = np.hstack([iio.imread(p) for p in lab_paths])
# hs_preds = np.hstack([iio.imread(p) for p in pred_paths])

all_imgs = {'GT': labs, 'Prediction': preds}

# Estimate number of labeled particles by dividing total amount of labeled
#  foreground pixels by median size of connected components in foreground.
for kind in 'GT', 'Prediction':
    imgs = all_imgs[kind]
    for img, i in zip(imgs, image_numbers):
        n_labeled = np.count_nonzero(img)
        cc, n_comps = ndimage.label(img)
        avg_cc_size = n_labeled / n_comps
        cc_sizes = np.bincount(cc.flat)
        cc_sizes = cc_sizes[1:]  # Exclude background cc
        # hist, bin_edges = np.histogram(cc_sizes)
        fig, ax = plt.subplots(tight_layout=True, figsize=(7, 7))
        hp = sns.histplot(cc_sizes, kde=True)
        plt.title(f'{kind}, image {i}')
        plt.xlabel('Connected component size (pixels)')

        if print_stats['hpeak']:
            # Extract maximum (i.e. here: first peak) of KDE
            try:
                kde = hp.lines[0]
                peak_idx = np.argmax(kde.get_ydata())
                hpeak_count = max([p.get_height() for p in hp.patches])
                hpeak = kde.get_xdata()[peak_idx]
                hpeak_estimated_particle_count = n_labeled / hpeak

                print(f'[KDE] Typical particle size of {kind} image {i}: {hpeak:.1f}')
                print(f'[KDE] Estimated particle count of {kind} image {i}: {hpeak_estimated_particle_count:.1f}')

                # plt.scatter(hpeak, hpeak_count, color='red', marker='o')
                plt.text(hpeak, hpeak_count + .1, f'Typical size: {hpeak:.1f}', color='red')
            except IndexError:
                print(f'No KDE available for {kind}, image {i}')

        # if i == 22:
            # import IPython ; IPython.embed()
        plt.savefig(f'{kind}-{i}.png', dpi=300)

        if print_stats['median']:
            median_size = np.median(cc_sizes)
            median_estimated_particle_count = n_labeled / median_size
            print(f'[Median] Typical particle size of {kind} image {i}: {median_size:.1f}')
            print(f'[Median] Estimated particle count of {kind} image {i}: {median_estimated_particle_count:.1f}')


## Median:
# [Median] Typical particle size of GT image 22: 262.0
# [Median] Estimated particle count of GT image 22: 747.1
# [Median] Typical particle size of GT image 32: 272.0
# [Median] Estimated particle count of GT image 32: 6.0
# [Median] Typical particle size of GT image 42: 362.0
# [Median] Estimated particle count of GT image 42: 1.0
# [Median] Typical particle size of Prediction image 22: 289.0
# [Median] Estimated particle count of Prediction image 22: 740.8
# [Median] Typical particle size of Prediction image 32: 285.0
# [Median] Estimated particle count of Prediction image 32: 6.5
# [Median] Typical particle size of Prediction image 42: 336.0
# [Median] Estimated particle count of Prediction image 42: 1.0

## Histogram KDE
# [KDE] Typical particle size of GT image 22: 265.2
# [KDE] Typical particle count of GT image 22: 738.2
# [KDE] Typical particle size of GT image 32: 266.8
# [KDE] Typical particle count of GT image 32: 6.1
# No KDE available for GT, image 42
# [KDE] Typical particle size of Prediction image 22: 291.4
# [KDE] Typical particle count of Prediction image 22: 734.7
# [KDE] Typical particle size of Prediction image 32: 308.2
# [KDE] Typical particle count of Prediction image 32: 6.0
# No KDE available for Prediction, image 42
