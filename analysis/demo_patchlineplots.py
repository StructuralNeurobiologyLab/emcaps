"""
Create line plots of horizontal and vertical axis profiles of encapsulin
image patches.

Superseded by patchlineplots.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import sys

mx_path = os.path.expanduser('~/tumpatches/mx_0000.tif')
qt_path = os.path.expanduser('~/tumpatches/qt_3065.tif')


mx = imageio.imread(mx_path)
qt = imageio.imread(qt_path)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8), tight_layout=True)


img = mx

horizontal_profile = img[img.shape[0] // 2, :]
vertical_profile = img[:, img.shape[1] // 2]

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('MxEnc')
axes[0, 1].plot(horizontal_profile)
axes[0, 1].set_title('Horizontal profile')
axes[0, 2].plot(vertical_profile)
axes[0, 2].set_title('Vertical profile')


img = qt

horizontal_profile = img[img.shape[0] // 2, :]
vertical_profile = img[:, img.shape[1] // 2]

axes[1, 0].imshow(img, cmap='gray')
axes[1, 0].set_title('QtEnc')

axes[1, 1].plot(horizontal_profile)
axes[1, 1].set_title('Horizontal profile')

axes[1, 2].plot(vertical_profile)
axes[1, 2].set_title('Vertical profile')

plt.show()


