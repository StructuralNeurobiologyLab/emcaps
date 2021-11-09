"""
Create line plots of horizontal and vertical axis profiles of encapsulin
image patches.

Requires a directory with patch images in the `patch_path`, which can be
built with eclassify_anaylsis.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import sys
from pathlib import Path
import tqdm


# avg = True  # Plot average profile on top
# avg = False  # Plot demo patch profile instead

mx_demo_path = Path('~/tumpatches/mx_0000.tif').expanduser()
qt_demo_path = Path('~/tumpatches/qt_3065.tif').expanduser()

patch_path = Path('~/tumpatches').expanduser()

demo_paths = {mx_demo_path, qt_demo_path}

profiles = {
    'MxEnc': {
        'horizontal': [],
        'vertical': []
    },
    'QtEnc': {
        'horizontal': [],
        'vertical': []
    },
}

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8), tight_layout=True)

for i, p in tqdm.tqdm(enumerate(patch_path.iterdir()), total=len(list(patch_path.glob('*')))):
    img = imageio.imread(p)
    if p.name.startswith('mx'):
        enctype = 'MxEnc'
        axrow = axes[0]
    elif p.name.startswith('qt'):
        enctype = 'QtEnc'
        axrow = axes[1]
    else:
        raise RuntimeError(p)

    horizontal_profile = img[img.shape[0] // 2, :]
    vertical_profile = img[:, img.shape[1] // 2]

    profiles[enctype]['horizontal'].append(horizontal_profile)
    profiles[enctype]['vertical'].append(vertical_profile)
    if p in demo_paths:
        axrow[0].imshow(img, cmap='gray')
        axrow[0].axhline(y=28//2, c='red', alpha=0.9)
        axrow[0].axvline(x=28//2, c='green', alpha=0.9)
        axrow[1].plot(horizontal_profile, c='red', linewidth=2)
        axrow[2].plot(vertical_profile, c='green', linewidth=2)
    axrow[0].set_title(enctype)
    axrow[1].plot(horizontal_profile, c='gray', linewidth=0.1, alpha=0.5)
    axrow[1].set_title('Horizontal profile')
    axrow[2].plot(vertical_profile, c='gray', linewidth=0.1, alpha=0.5)
    axrow[2].set_title('Vertical profile')


# for k in profiles.keys():
#     for j in profiles[k].keys():
#         prof = np.array(profiles[k][j])
#         avgprof = np.mean(prof, axis=0)
#         profiles[k][j] = avgprof

for enctype, orientations in profiles.items():
    for orientation, profile in orientations.items():
        prof = np.array(profile)
        avgprof = np.mean(prof, axis=0)
        profiles[enctype][orientation] = avgprof
        axrow = axes[0] if enctype == 'MxEnc' else axes[1]
        ax = axrow[1] if orientation == 'horizontal' else axrow[2]
        ax.plot(avgprof, c='blue', linewidth=1, linestyle='--')

fig.suptitle('Axis profile plots. Red: horizontal, green: vertical. Gray: all images, blue: average profiles.', fontsize=16)

plt.savefig('/tmp/patchprofile.png')
plt.show()

# Average plots
fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
ax.plot(profiles['MxEnc']['horizontal'], c='purple', label='MxEnc horizontal')
ax.plot(profiles['QtEnc']['horizontal'], c='orange', label='QtEnc horizontal')
ax.legend()
ax.set_title('Horizontal average profiles')

plt.show()

fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
ax.plot(profiles['MxEnc']['vertical'], c='purple', label='MxEnc vertical')
ax.plot(profiles['QtEnc']['vertical'], c='orange', label='QtEnc vertical')
ax.legend()
ax.set_title('Vertical average profiles')


plt.show()
